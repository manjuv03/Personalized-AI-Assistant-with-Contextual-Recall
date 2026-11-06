"""Pull durable facts out of a user's message so they can be stored long term."""

from __future__ import annotations

import json
import re

from .long_term import CATEGORIES

EXTRACTION_PROMPT = """You maintain long-term memory for a personal assistant.
Read the user's latest message and list any durable facts worth remembering in
future conversations: who they are, preferences, goals, ongoing projects,
or corrections to earlier facts.

Rules:
- Only facts the user stated about themselves. No guesses.
- One short sentence per fact, written in the third person ("The user ...").
- Skip small talk, questions and one-off requests.
- If there is nothing worth remembering, return [].

Return ONLY a JSON list like:
[{{"text": "The user's name is Priya.", "category": "profile", "importance": 0.9}}]
category is one of: {categories}. importance is 0 to 1.

User message:
{message}
"""


def build_extraction_prompt(message: str) -> str:
    return EXTRACTION_PROMPT.format(categories=", ".join(CATEGORIES), message=message)


def parse_extracted_facts(raw: str) -> list[dict]:
    """Parse the model's reply into clean fact dicts. Tolerates code fences and stray text."""
    if not raw:
        return []
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    facts = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        fact = str(item.get("text", "")).strip()
        if len(fact) < 3:
            continue
        category = str(item.get("category", "other")).lower()
        if category not in CATEGORIES:
            category = "other"
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        facts.append({"text": fact, "category": category, "importance": max(0.0, min(1.0, importance))})
    return facts


def extract_facts(chat_model, message: str) -> list[dict]:
    from ..llm import message_text

    reply = chat_model.invoke(build_extraction_prompt(message))
    return parse_extracted_facts(message_text(reply))
