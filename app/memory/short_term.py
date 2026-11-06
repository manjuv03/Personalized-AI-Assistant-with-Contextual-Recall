"""Short-term (contextual) memory helpers.

The LangGraph checkpointer keeps each conversation's messages. To stop the
prompt growing forever, older messages are folded into a running summary and
only the most recent ones are kept word for word.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")

SUMMARY_PROMPT = """Summarise this conversation between a user and their assistant so it can
continue later without the full transcript. Keep names, decisions, open
questions and anything the user asked to remember. Be brief.

Existing summary (may be empty):
{summary}

New messages to fold in:
{transcript}

Updated summary:"""


def split_for_summary(messages: Sequence[T], keep_recent: int) -> tuple[list[T], list[T]]:
    """Return (to_summarise, to_keep). Nothing is summarised until the window is full."""
    if keep_recent <= 0 or len(messages) <= keep_recent:
        return [], list(messages)
    cut = len(messages) - keep_recent
    return list(messages[:cut]), list(messages[cut:])


def format_transcript(messages) -> str:
    lines = []
    for m in messages:
        role = {"human": "User", "ai": "Assistant"}.get(getattr(m, "type", ""), "Message")
        lines.append(f"{role}: {getattr(m, 'content', m)}")
    return "\n".join(lines)


def build_summary_prompt(summary: str, messages) -> str:
    return SUMMARY_PROMPT.format(summary=summary or "(none)", transcript=format_transcript(messages))
