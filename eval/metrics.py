"""Scoring helpers for the evaluation (no model dependencies)."""

from __future__ import annotations

import re
import statistics


def contains_any(text: str, keywords: list[str]) -> bool:
    """Whole-word match; a trailing * matches any ending ("allerg*" -> allergy, allergic)."""
    for k in keywords:
        k = k.lower()
        pattern = r"\b" + re.escape(k[:-1]) if k.endswith("*") else r"\b" + re.escape(k) + r"\b"
        if re.search(pattern, text.lower()):
            return True
    return False


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "p95": round(percentile(values, 95), 1),
    }
