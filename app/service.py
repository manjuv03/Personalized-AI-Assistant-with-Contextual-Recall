"""Ties the graph, the memory store and the logs together."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage

from .config import Settings, settings as default_settings
from .graph import build_checkpointer, build_graph
from .llm import get_chat_model, get_embeddings, message_text
from .logs import InteractionLog
from .memory.extractor import extract_facts
from .memory.long_term import LongTermMemory, build_long_term_memory

log = logging.getLogger(__name__)


@dataclass
class ChatResult:
    reply: str
    recalled: list[dict]
    latency_ms: float
    log_id: int
    stored: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)


class Assistant:
    def __init__(
        self,
        settings: Settings = default_settings,
        chat_model=None,
        memory: LongTermMemory | None = None,
        log: InteractionLog | None = None,
    ) -> None:
        self.settings = settings
        self.chat_model = chat_model or get_chat_model(settings)
        self.memory = memory or build_long_term_memory(settings, get_embeddings(settings))
        self.log = log or InteractionLog(settings.log_db_path)
        self.graph = build_graph(
            self.memory,
            self.chat_model,
            settings.keep_recent_messages,
            checkpointer=build_checkpointer(settings.checkpoint_path),
            use_tools=settings.agent_tools,
            max_tool_rounds=settings.max_tool_rounds,
        )

    @staticmethod
    def _config(user_id: str, thread_id: str, use_memory: bool = True) -> dict:
        return {"configurable": {"thread_id": f"{user_id}:{thread_id}", "user_id": user_id, "use_memory": use_memory}}

    @staticmethod
    def _final_reply(messages) -> str:
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                text = message_text(m).strip()
                if text:
                    return text
        return ""

    def chat(self, user_id: str, thread_id: str, message: str, use_memory: bool = True) -> ChatResult:
        """Answer one message. Latency covers recall + reply (+ tools and summary when they run)."""
        config = self._config(user_id, thread_id, use_memory)
        start = time.perf_counter()
        state = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config)
        latency_ms = (time.perf_counter() - start) * 1000
        reply = self._final_reply(state["messages"])
        recalled = state.get("recalled", [])
        tools = state.get("tool_events", [])
        log_id = self.log.add(user_id, thread_id, message, reply, recalled, latency_ms, tools=tools)
        return ChatResult(reply=reply, recalled=recalled, latency_ms=latency_ms, log_id=log_id, tools=tools)

    def chat_stream(self, user_id: str, thread_id: str, message: str, use_memory: bool = True):
        """Answer one message as a stream of events:

        recalled -> token ... (tools ...) token ... -> done
        """
        config = self._config(user_id, thread_id, use_memory)
        start = time.perf_counter()
        first_token_ms = None
        recalled: list[dict] = []
        for mode, payload in self.graph.stream(
            {"messages": [HumanMessage(content=message)]}, config, stream_mode=["messages", "updates"]
        ):
            if mode == "updates":
                for node, update in (payload or {}).items():
                    update = update or {}
                    if node == "recall":
                        recalled = update.get("recalled", [])
                        yield {"type": "recalled", "recalled": recalled}
                    elif node == "tools":
                        yield {"type": "tools", "tools": update.get("tool_events", [])}
            elif mode == "messages":
                chunk, meta = payload
                if (meta or {}).get("langgraph_node") != "respond":
                    continue  # don't stream summary text
                text = message_text(chunk)
                if text:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - start) * 1000
                    yield {"type": "token", "text": text}
        latency_ms = (time.perf_counter() - start) * 1000
        values = self.graph.get_state(config).values or {}
        reply = self._final_reply(values.get("messages", []))
        tools = values.get("tool_events", [])
        log_id = self.log.add(
            user_id, thread_id, message, reply, recalled, latency_ms, tools=tools, first_token_ms=first_token_ms
        )
        yield {
            "type": "done",
            "reply": reply,
            "recalled": recalled,
            "tools": tools,
            "latency_ms": round(latency_ms, 1),
            "first_token_ms": None if first_token_ms is None else round(first_token_ms, 1),
            "log_id": log_id,
        }

    def threads(self, user_id: str) -> list[dict]:
        return self.log.threads(user_id)

    def thread_history(self, user_id: str, thread_id: str) -> list[dict]:
        return self.log.thread_turns(user_id, thread_id)

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        removed = self.log.delete_thread(user_id, thread_id)
        try:
            self.graph.checkpointer.delete_thread(f"{user_id}:{thread_id}")
        except Exception:  # noqa: BLE001 - older checkpointers may not support deletion
            log.warning("Could not delete saved state for thread %s", thread_id)
        return removed > 0

    def remember(self, user_id: str, message: str, log_id: int | None = None) -> list[dict]:
        """Extract facts from the user's message and save them. Safe to run in the background."""
        stored = []
        for fact in extract_facts(self.chat_model, message):
            record = self.memory.add(user_id, fact["text"], fact["category"], fact["importance"])
            stored.append({"id": record.id, "text": record.text, "category": record.category})
        if log_id is not None:
            self.log.set_stored(log_id, stored)
        return stored

    def remember_safely(self, user_id: str, message: str, log_id: int | None = None) -> list[dict]:
        """Background version of remember(): a failure is logged instead of raised."""
        try:
            return self.remember(user_id, message, log_id)
        except Exception:  # noqa: BLE001
            log.exception("Could not save memories for this message")
            return []

    def summary(self, user_id: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
        snapshot = self.graph.get_state(config)
        return (snapshot.values or {}).get("summary", "") if snapshot else ""
