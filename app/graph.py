"""The LangGraph agent.

    START -> recall -> respond --(tool calls?)--> tools -> respond ...
                               \\-(thread long?)--> summarise -> END
                               \\----------------------------> END

recall    searches the user's long-term memory (Chroma DB) with the new message
respond   answers using recalled facts + the running summary + recent messages;
          it may call memory tools (search / save / forget)
tools     runs those tool calls for this user and hands the results back
summarise folds older messages into the summary so the prompt stays small
          (runs once the thread reaches 2 x KEEP_RECENT_MESSAGES)

Facts are also extracted from every message outside the graph (see
service.py), so saving never slows down the reply.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .llm import message_text
from .memory.long_term import LongTermMemory
from .memory.short_term import build_summary_prompt, split_for_summary
from .memory.tools import TOOL_GUIDE, TOOLS, run_tool

SYSTEM_PROMPT = """You are a friendly personal assistant that remembers the people you talk to.
Use what you know about the user when it helps, naturally and without listing it back.
If a remembered fact conflicts with what the user says now, trust the user and go with the new information.
Never invent facts about the user."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    recalled: list[dict]
    tool_events: list[dict]
    tool_rounds: int


def _configurable(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", {})


def build_graph(
    memory: LongTermMemory,
    chat_model,
    keep_recent_messages: int = 8,
    checkpointer=None,
    use_tools: bool = True,
    max_tool_rounds: int = 3,
):
    tool_model = chat_model.bind_tools(TOOLS) if use_tools and hasattr(chat_model, "bind_tools") else None

    def recall(state: AgentState, config: RunnableConfig) -> dict:
        cfg = _configurable(config)
        fresh = {"recalled": [], "tool_events": [], "tool_rounds": 0}
        if not cfg.get("use_memory", True):
            return fresh
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_human is None:
            return fresh
        hits = memory.recall(cfg.get("user_id", "default"), str(last_human.content))
        return {**fresh, "recalled": [h.to_dict() for h in hits]}

    def respond(state: AgentState, config: RunnableConfig) -> dict:
        cfg = _configurable(config)
        tools_on = tool_model is not None and cfg.get("use_memory", True)
        parts = [SYSTEM_PROMPT]
        if tools_on:
            parts.append(TOOL_GUIDE)
        recalled = state.get("recalled") or []
        if recalled:
            facts = "\n".join(f"- {m['text']}" for m in recalled)
            parts.append(f"What you remember about this user:\n{facts}")
        if state.get("summary"):
            parts.append(f"Summary of the conversation so far:\n{state['summary']}")
        prompt = [SystemMessage(content="\n\n".join(parts)), *state["messages"]]
        # After the last allowed tool round, answer without tools so the turn always ends in text.
        model = tool_model if tools_on and state.get("tool_rounds", 0) < max_tool_rounds else chat_model
        return {"messages": [model.invoke(prompt)]}

    def tools(state: AgentState, config: RunnableConfig) -> dict:
        user_id = _configurable(config).get("user_id", "default")
        last = state["messages"][-1]
        results, events = [], list(state.get("tool_events") or [])
        for call in getattr(last, "tool_calls", None) or []:
            output, event = run_tool(memory, user_id, call["name"], call.get("args") or {})
            results.append(ToolMessage(content=output, tool_call_id=call["id"], name=call["name"]))
            events.append(event)
        return {"messages": results, "tool_events": events, "tool_rounds": state.get("tool_rounds", 0) + 1}

    # Summarise in batches: wait until the thread holds twice the recent window,
    # then trim back to the window. That's one extra model call every few turns
    # instead of one on every turn.
    summary_trigger = keep_recent_messages * 2

    def after_respond(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "summarise" if len(state["messages"]) >= summary_trigger else END

    def summarise(state: AgentState) -> dict:
        old, _ = split_for_summary(
            state["messages"], keep_recent_messages, is_start=lambda m: isinstance(m, HumanMessage)
        )
        if not old:
            return {}
        reply = chat_model.invoke(build_summary_prompt(state.get("summary", ""), old))
        return {
            "summary": message_text(reply).strip(),
            "messages": [RemoveMessage(id=m.id) for m in old],
        }

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall)
    graph.add_node("respond", respond)
    graph.add_node("tools", tools)
    graph.add_node("summarise", summarise)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_conditional_edges("respond", after_respond, {"tools": "tools", "summarise": "summarise", END: END})
    graph.add_edge("tools", "respond")
    graph.add_edge("summarise", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def build_checkpointer(path: str):
    """SQLite checkpointer so chats survive restarts; falls back to memory if unavailable."""
    if path and path != ":memory:":
        try:
            import sqlite3
            from pathlib import Path

            from langgraph.checkpoint.sqlite import SqliteSaver

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "langgraph-checkpoint-sqlite is not installed; chats will reset when the server restarts."
            )
    return InMemorySaver()
