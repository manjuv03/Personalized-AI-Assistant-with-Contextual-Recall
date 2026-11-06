"""Interaction log: every turn with what was recalled and how long it took."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class InteractionLog:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                reply TEXT NOT NULL,
                recalled TEXT NOT NULL,
                stored TEXT NOT NULL DEFAULT '[]',
                latency_ms REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def add(self, user_id, thread_id, user_message, reply, recalled, latency_ms) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO interactions (ts, user_id, thread_id, user_message, reply, recalled, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), user_id, thread_id, user_message, reply, json.dumps(recalled), latency_ms),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_stored(self, log_id: int, stored: list[dict]) -> None:
        with self._lock:
            self._conn.execute("UPDATE interactions SET stored = ? WHERE id = ?", (json.dumps(stored), log_id))
            self._conn.commit()

    def recent(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, thread_id, user_message, reply, recalled, stored, latency_ms"
                " FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "thread_id": r[2],
                "user_message": r[3],
                "reply": r[4],
                "recalled": json.loads(r[5]),
                "stored": json.loads(r[6]),
                "latency_ms": round(r[7], 1),
            }
            for r in rows
        ]

    def stats(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), AVG(latency_ms) FROM interactions WHERE user_id = ?", (user_id,)
            ).fetchone()
        return {"turns": row[0], "avg_latency_ms": round(row[1] or 0.0, 1)}
