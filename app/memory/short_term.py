"""Short-term (contextual) memory helpers.

The LangGraph checkpointer keeps each conversation's messages. To stop the
prompt growing forever, older messages are folded into a running summary and
only the most recent ones are kept word for word.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

from ..llm import message_text

T = TypeVar("T")

SUMMARY_PROMPT = """Summarise this conversation between a user and their assistant so it can
continue later without the full transcript. Keep names, decisions, open
questions and anything the user asked to remember. Be brief.

Existing summary (may be empty):
{summary}

New messages to fold in:
{transcript}

Updated summary:"""


def split_for_summary(
    messages: Sequence[T], keep_recent: int, is_start: Callable[[T], bool] | None = None
) -> tuple[list[T], list[T]]:
    """Return (to_summarise, to_keep). Nothing is summarised until the window is full.

    `is_start` marks messages that may begin the kept part (e.g. a user message).
    It stops the cut from separating a tool call from its tool result, which
    chat APIs reject.
    """
    if keep_recent <= 0 or len(messages) <= keep_recent:
        return [], list(messages)
    cut = len(messages) - keep_recent
    if is_start is not None:
        forward = next((i for i in range(cut, len(messages)) if is_start(messages[i])), None)
        backward = next((i for i in range(cut, 0, -1) if is_start(messages[i])), None)
        cut = forward if forward is not None else backward
        if not cut:
            return [], list(messages)
    return list(messages[:cut]), list(messages[cut:])


def format_transcript(messages) -> str:
    lines = []
    for m in messages:
        role = {"human": "User", "ai": "Assistant", "tool": "Tool result"}.get(getattr(m, "type", ""), "Message")
        text = message_text(m).strip()
        if not text:
            calls = getattr(m, "tool_calls", None) or []
            text = "; ".join(f"used {c.get('name')}" for c in calls)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_summary_prompt(summary: str, messages) -> str:
    return SUMMARY_PROMPT.format(summary=summary or "(none)", transcript=format_transcript(messages))
