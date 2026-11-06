"""Ties the graph, the memory store and the logs together."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from .config import Settings, settings as default_settings
from .graph import build_graph
from .llm import get_chat_model, get_embeddings
from .logs import InteractionLog
from .memory.extractor import extract_facts
from .memory.long_term import LongTermMemory, build_long_term_memory


@dataclass
class ChatResult:
    reply: str
    recalled: list[dict]
    latency_ms: float
    log_id: int
    stored: list[dict] = field(default_factory=list)


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
        self.graph = build_graph(self.memory, self.chat_model, settings.keep_recent_messages)

    def chat(self, user_id: str, thread_id: str, message: str, use_memory: bool = True) -> ChatResult:
        """Answer one message. Latency covers recall + reply (+ summary when it runs)."""
        config = {"configurable": {"thread_id": f"{user_id}:{thread_id}", "user_id": user_id, "use_memory": use_memory}}
        start = time.perf_counter()
        state = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config)
        latency_ms = (time.perf_counter() - start) * 1000
        reply = str(state["messages"][-1].content)
        recalled = state.get("recalled", [])
        log_id = self.log.add(user_id, thread_id, message, reply, recalled, latency_ms)
        return ChatResult(reply=reply, recalled=recalled, latency_ms=latency_ms, log_id=log_id)

    def remember(self, user_id: str, message: str, log_id: int | None = None) -> list[dict]:
        """Extract facts from the user's message and save them. Safe to run in the background."""
        stored = []
        for fact in extract_facts(self.chat_model, message):
            record = self.memory.add(user_id, fact["text"], fact["category"], fact["importance"])
            stored.append({"id": record.id, "text": record.text, "category": record.category})
        if log_id is not None:
            self.log.set_stored(log_id, stored)
        return stored

    def summary(self, user_id: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
        snapshot = self.graph.get_state(config)
        return (snapshot.values or {}).get("summary", "") if snapshot else ""
