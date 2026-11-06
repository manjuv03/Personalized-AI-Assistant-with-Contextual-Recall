"""Interaction log: every turn with what was recalled, what tools ran and how long it took.

It also powers the chat sidebar: the list of past chats and each chat's history
come from these rows, so nothing is lost when older messages are summarised.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

COLUMNS = "id, ts, thread_id, user_message, reply, recalled, stored, latency_ms, tools, first_token_ms"


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
                latency_ms REAL NOT NULL,
                tools TEXT NOT NULL DEFAULT '[]',
                first_token_ms REAL
            )"""
        )
        # Older databases were created before these columns existed.
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(interactions)")}
        if "tools" not in existing:
            self._conn.execute("ALTER TABLE interactions ADD COLUMN tools TEXT NOT NULL DEFAULT '[]'")
        if "first_token_ms" not in existing:
            self._conn.execute("ALTER TABLE interactions ADD COLUMN first_token_ms REAL")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_user_thread ON interactions (user_id, thread_id)")
        self._conn.commit()

    def add(self, user_id, thread_id, user_message, reply, recalled, latency_ms, tools=None, first_token_ms=None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO interactions (ts, user_id, thread_id, user_message, reply, recalled, latency_ms, tools, first_token_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), user_id, thread_id, user_message, reply,
                    json.dumps(recalled), latency_ms, json.dumps(tools or []), first_token_ms,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_stored(self, log_id: int, stored: list[dict]) -> None:
        with self._lock:
            self._conn.execute("UPDATE interactions SET stored = ? WHERE id = ?", (json.dumps(stored), log_id))
            self._conn.commit()

    @staticmethod
    def _row(r) -> dict:
        return {
            "id": r[0],
            "ts": r[1],
            "thread_id": r[2],
            "user_message": r[3],
            "reply": r[4],
            "recalled": json.loads(r[5]),
            "stored": json.loads(r[6]),
            "latency_ms": round(r[7], 1),
            "tools": json.loads(r[8] or "[]"),
            "first_token_ms": None if r[9] is None else round(r[9], 1),
        }

    def recent(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {COLUMNS} FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def stats(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), AVG(latency_ms) FROM interactions WHERE user_id = ?", (user_id,)
            ).fetchone()
        return {"turns": row[0], "avg_latency_ms": round(row[1] or 0.0, 1)}

    # --- chats ------------------------------------------------------------

    def threads(self, user_id: str) -> list[dict]:
        """One entry per chat, newest first, titled by its first message."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT thread_id, MIN(ts), MAX(ts), COUNT(*),
                          (SELECT user_message FROM interactions f
                           WHERE f.user_id = i.user_id AND f.thread_id = i.thread_id ORDER BY f.id LIMIT 1)
                   FROM interactions i WHERE user_id = ?
                   GROUP BY thread_id ORDER BY MAX(ts) DESC""",
                (user_id,),
            ).fetchall()
        return [
            {"thread_id": r[0], "created": r[1], "updated": r[2], "turns": r[3], "title": (r[4] or r[0])[:60]}
            for r in rows
        ]

    def thread_turns(self, user_id: str, thread_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {COLUMNS} FROM interactions WHERE user_id = ? AND thread_id = ? ORDER BY id",
                (user_id, thread_id),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete_thread(self, user_id: str, thread_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM interactions WHERE user_id = ? AND thread_id = ?", (user_id, thread_id)
            )
            self._conn.commit()
            return cur.rowcount
