"""The LangGraph agent.

    START -> recall -> respond --(conversation too long?)--> summarise -> END
                                \\-----------------------------------------> END

recall    searches the user's long-term memory (Chroma DB) with the new message
respond   answers using recalled facts + the running summary + recent messages
summarise folds older messages into the summary so the prompt stays small

Writing new facts to long-term memory happens outside the graph (see
service.py), so it never slows down the reply.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .llm import message_text
from .memory.long_term import LongTermMemory
from .memory.short_term import build_summary_prompt, split_for_summary

SYSTEM_PROMPT = """You are a friendly personal assistant that remembers the people you talk to.
Use what you know about the user when it helps, naturally and without listing it back.
If a remembered fact conflicts with what the user says now, trust the user and go with the new information.
Never invent facts about the user."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    recalled: list[dict]


def _configurable(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", {})


def build_graph(memory: LongTermMemory, chat_model, keep_recent_messages: int = 8, checkpointer=None):
    def recall(state: AgentState, config: RunnableConfig) -> dict:
        cfg = _configurable(config)
        if not cfg.get("use_memory", True):
            return {"recalled": []}
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_human is None:
            return {"recalled": []}
        hits = memory.recall(cfg.get("user_id", "default"), str(last_human.content))
        return {"recalled": [h.to_dict() for h in hits]}

    def respond(state: AgentState, config: RunnableConfig) -> dict:
        parts = [SYSTEM_PROMPT]
        recalled = state.get("recalled") or []
        if recalled:
            facts = "\n".join(f"- {m['text']}" for m in recalled)
            parts.append(f"What you remember about this user:\n{facts}")
        if state.get("summary"):
            parts.append(f"Summary of the conversation so far:\n{state['summary']}")
        prompt = [SystemMessage(content="\n\n".join(parts)), *state["messages"]]
        reply = chat_model.invoke(prompt)
        return {"messages": [reply]}

    def needs_summary(state: AgentState) -> str:
        return "summarise" if len(state["messages"]) > keep_recent_messages else END

    def summarise(state: AgentState) -> dict:
        old, _ = split_for_summary(state["messages"], keep_recent_messages)
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
    graph.add_node("summarise", summarise)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_conditional_edges("respond", needs_summary, {"summarise": "summarise", END: END})
    graph.add_edge("summarise", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
