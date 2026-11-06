"""Memory tools the assistant can call by itself.

The graph always recalls the top facts before replying. These tools let the
model go further when it decides to: look something up, save a fact the user
asked it to remember, or forget one that's wrong.

The functions below only define each tool's name, description and arguments
for the model; `run_tool` does the actual work for the current user.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .long_term import CATEGORIES, LongTermMemory


@tool
def search_memory(query: str) -> str:
    """Search what you remember about the user. Use it when a question needs a personal
    detail you weren't already given, or to find a memory's id before forgetting it."""
    return ""


@tool
def save_memory(text: str, category: str = "other", importance: float = 0.6) -> str:
    """Save a lasting fact about the user, mainly when they ask you to remember something.
    Write one short third-person sentence, e.g. "The user's sister is called Asha."
    category is one of: profile, preference, goal, project, other. importance is 0 to 1."""
    return ""


@tool
def forget_memory(memory_id: str) -> str:
    """Delete one memory when the user asks you to forget something or says a fact is wrong.
    Call search_memory first to get the memory's id."""
    return ""


TOOLS = [search_memory, save_memory, forget_memory]

TOOL_GUIDE = """You can manage your memory of the user with tools:
- search_memory: look up details you weren't given above.
- save_memory: when the user asks you to remember something, save it as one short fact.
- forget_memory: when the user asks you to forget something or corrects a fact; search first to get the id.
Don't use tools for small talk. After using a tool, tell the user briefly what you did."""


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def run_tool(memory: LongTermMemory, user_id: str, name: str, args: dict) -> tuple[str, dict]:
    """Run one tool call for this user. Returns (result for the model, event for the UI)."""
    try:
        if name == "search_memory":
            hits = memory.recall(user_id, str(args.get("query", "")), k=5)
            if not hits:
                return "No matching memories.", {"name": name, "label": "Searched memory: nothing found"}
            lines = [f"[{h.record.id}] {h.record.text} ({h.record.category})" for h in hits]
            return "\n".join(lines), {"name": name, "label": f"Searched memory: {len(hits)} found"}

        if name == "save_memory":
            text = str(args.get("text", "")).strip()
            if len(text) < 3:
                return "Nothing saved: the fact was empty.", {"name": name, "label": "Save skipped"}
            category = str(args.get("category", "other")).lower()
            category = category if category in CATEGORIES else "other"
            try:
                importance = float(args.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            record = memory.add(user_id, text, category, importance)
            return f"Saved [{record.id}] {record.text}", {"name": name, "label": f"Saved: {_short(record.text)}", "id": record.id}

        if name == "forget_memory":
            memory_id = str(args.get("memory_id", "")).strip().strip("[]")
            row = memory.backend.get(user_id, memory_id)
            if row is None or not memory.delete(user_id, memory_id):
                return "No memory with that id. Search first.", {"name": name, "label": "Forget failed: unknown id"}
            return f"Forgot: {row[0].text}", {"name": name, "label": f"Forgot: {_short(row[0].text)}", "id": memory_id}

        return f"Unknown tool {name}.", {"name": name, "label": f"Unknown tool {name}"}
    except Exception as error:  # noqa: BLE001 - report the failure to the model instead of crashing the turn
        return f"Tool failed: {error}", {"name": name, "label": f"{name} failed"}
