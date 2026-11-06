"""Long-term memory: durable facts about each user, stored as vectors.

Facts (name, preferences, goals, projects...) are embedded and kept in Chroma DB,
tagged with the user's id. On every turn the assistant searches this store with
the new message and ranks hits by similarity, importance and recency.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from ..llm import Embedder

CATEGORIES = ("profile", "preference", "goal", "project", "other")
RECENCY_HALF_LIFE_DAYS = 30.0

# Weights for ranking recalled memories. Similarity dominates; importance and
# recency break ties between memories that are about equally relevant.
W_SIMILARITY = 0.75
W_IMPORTANCE = 0.15
W_RECENCY = 0.10


@dataclass
class MemoryRecord:
    id: str
    user_id: str
    text: str
    category: str = "other"
    importance: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    times_recalled: int = 0

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("id")
        data.pop("text")
        return data


@dataclass
class RecalledMemory:
    record: MemoryRecord
    similarity: float
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.record.id,
            "text": self.record.text,
            "category": self.record.category,
            "importance": round(self.record.importance, 2),
            "similarity": round(self.similarity, 3),
            "score": round(self.score, 3),
        }


def recency_weight(updated_at: float, now: float, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """1.0 for a memory updated just now, 0.5 after one half-life, and so on."""
    age_days = max(0.0, now - updated_at) / 86400.0
    return math.pow(0.5, age_days / half_life_days)


def rank_score(similarity: float, importance: float, updated_at: float, now: float) -> float:
    return (
        W_SIMILARITY * similarity
        + W_IMPORTANCE * max(0.0, min(1.0, importance))
        + W_RECENCY * recency_weight(updated_at, now)
    )


class VectorBackend(Protocol):
    def upsert(self, record: MemoryRecord, embedding: list[float]) -> None: ...
    def get(self, user_id: str, memory_id: str) -> tuple[MemoryRecord, list[float]] | None: ...
    def query(self, user_id: str, embedding: list[float], k: int) -> list[tuple[MemoryRecord, float]]: ...
    def list(self, user_id: str) -> list[tuple[MemoryRecord, list[float]]]: ...
    def delete(self, user_id: str, memory_id: str) -> bool: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryBackend:
    """Plain Python store with the same interface as Chroma, used in tests."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[MemoryRecord, list[float]]] = {}

    def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        self._rows[record.id] = (record, embedding)

    def get(self, user_id: str, memory_id: str) -> tuple[MemoryRecord, list[float]] | None:
        row = self._rows.get(memory_id)
        return row if row and row[0].user_id == user_id else None

    def query(self, user_id: str, embedding: list[float], k: int) -> list[tuple[MemoryRecord, float]]:
        hits = [
            (rec, _cosine(embedding, emb)) for rec, emb in self._rows.values() if rec.user_id == user_id
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    def list(self, user_id: str) -> list[tuple[MemoryRecord, list[float]]]:
        return [(rec, emb) for rec, emb in self._rows.values() if rec.user_id == user_id]

    def delete(self, user_id: str, memory_id: str) -> bool:
        row = self._rows.get(memory_id)
        if row and row[0].user_id == user_id:
            del self._rows[memory_id]
            return True
        return False


class ChromaBackend:
    """Persistent Chroma DB collection. One collection, filtered by user_id."""

    def __init__(self, path: str, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _record(mem_id: str, text: str, meta: dict) -> MemoryRecord:
        return MemoryRecord(
            id=mem_id,
            text=text,
            user_id=str(meta.get("user_id", "")),
            category=str(meta.get("category", "other")),
            importance=float(meta.get("importance", 0.5)),
            created_at=float(meta.get("created_at", 0.0)),
            updated_at=float(meta.get("updated_at", 0.0)),
            times_recalled=int(meta.get("times_recalled", 0)),
        )

    def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        self._col.upsert(
            ids=[record.id],
            documents=[record.text],
            embeddings=[embedding],
            metadatas=[record.metadata()],
        )

    def get(self, user_id: str, memory_id: str) -> tuple[MemoryRecord, list[float]] | None:
        res = self._col.get(ids=[memory_id], include=["documents", "metadatas", "embeddings"])
        if not res["ids"] or res["metadatas"][0].get("user_id") != user_id:
            return None
        embeddings = res.get("embeddings")
        embedding = list(embeddings[0]) if embeddings is not None and len(embeddings) else []
        return self._record(res["ids"][0], res["documents"][0], res["metadatas"][0]), embedding

    def query(self, user_id: str, embedding: list[float], k: int) -> list[tuple[MemoryRecord, float]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for mem_id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Cosine distance -> cosine similarity.
            hits.append((self._record(mem_id, doc, meta), 1.0 - float(dist)))
        return hits

    def list(self, user_id: str) -> list[tuple[MemoryRecord, list[float]]]:
        res = self._col.get(where={"user_id": user_id}, include=["documents", "metadatas", "embeddings"])
        embeddings = res.get("embeddings")
        if embeddings is None:
            embeddings = [[] for _ in res["ids"]]
        return [
            (self._record(mem_id, doc, meta), list(emb))
            for mem_id, doc, meta, emb in zip(res["ids"], res["documents"], res["metadatas"], embeddings)
        ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        res = self._col.get(ids=[memory_id], include=["metadatas"])
        if not res["ids"] or res["metadatas"][0].get("user_id") != user_id:
            return False
        self._col.delete(ids=[memory_id])
        return True


class LongTermMemory:
    def __init__(
        self,
        backend: VectorBackend,
        embedder: Embedder,
        top_k: int = 5,
        min_score: float = 0.25,
        duplicate_threshold: float = 0.92,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.duplicate_threshold = duplicate_threshold

    def add(self, user_id: str, text: str, category: str = "other", importance: float = 0.5) -> MemoryRecord:
        """Store a fact, or refresh an existing one if it is a near-duplicate."""
        text = text.strip()
        if category not in CATEGORIES:
            category = "other"
        now = time.time()
        embedding = self.embedder.embed_query(text)

        existing = self.backend.query(user_id, embedding, k=1)
        if existing and existing[0][1] >= self.duplicate_threshold:
            record = existing[0][0]
            record.text = text  # newer wording wins (e.g. a corrected fact)
            record.category = category
            record.importance = max(record.importance, importance)
            record.updated_at = now
        else:
            record = MemoryRecord(
                id=uuid.uuid4().hex,
                user_id=user_id,
                text=text,
                category=category,
                importance=importance,
                created_at=now,
                updated_at=now,
            )
        self.backend.upsert(record, embedding)
        return record

    def recall(self, user_id: str, query: str, k: int | None = None) -> list[RecalledMemory]:
        k = k or self.top_k
        now = time.time()
        embedding = self.embedder.embed_query(query)
        # Over-fetch, then re-rank with importance and recency.
        hits = self.backend.query(user_id, embedding, k=k * 3)
        ranked = [
            RecalledMemory(rec, sim, rank_score(sim, rec.importance, rec.updated_at, now))
            for rec, sim in hits
        ]
        ranked = [r for r in ranked if r.score >= self.min_score]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:k]

    def update(
        self,
        user_id: str,
        memory_id: str,
        text: str | None = None,
        category: str | None = None,
        importance: float | None = None,
    ) -> MemoryRecord | None:
        """Edit a memory by hand. Changing the text re-embeds it."""
        row = self.backend.get(user_id, memory_id)
        if row is None:
            return None
        record, embedding = row
        if text is not None and text.strip() and text.strip() != record.text:
            record.text = text.strip()
            embedding = self.embedder.embed_query(record.text)
        if category is not None:
            record.category = category if category in CATEGORIES else "other"
        if importance is not None:
            record.importance = max(0.0, min(1.0, float(importance)))
        if not embedding:
            embedding = self.embedder.embed_query(record.text)
        record.updated_at = time.time()
        self.backend.upsert(record, embedding)
        return record

    def list(self, user_id: str) -> list[MemoryRecord]:
        records = [rec for rec, _ in self.backend.list(user_id)]
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self.backend.delete(user_id, memory_id)

    def projection(self, user_id: str) -> list[dict]:
        """2-D PCA projection of a user's memory embeddings, for the memory map."""
        rows = [(rec, emb) for rec, emb in self.backend.list(user_id) if emb]
        if not rows:
            return []
        import numpy as np

        matrix = np.array([emb for _, emb in rows], dtype=float)
        if len(rows) == 1:
            coords = np.zeros((1, 2))
        else:
            centred = matrix - matrix.mean(axis=0)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            coords = centred @ vt[:2].T
            if coords.shape[1] == 1:
                coords = np.hstack([coords, np.zeros((len(rows), 1))])
        return [
            {"id": rec.id, "text": rec.text, "category": rec.category, "x": float(x), "y": float(y)}
            for (rec, _), (x, y) in zip(rows, coords)
        ]


def build_long_term_memory(settings, embedder: Embedder) -> LongTermMemory:
    if settings.vector_backend == "memory":
        backend: VectorBackend = InMemoryBackend()
    else:
        backend = ChromaBackend(settings.chroma_path, settings.chroma_collection)
    return LongTermMemory(
        backend,
        embedder,
        top_k=settings.recall_top_k,
        min_score=settings.recall_min_score,
        duplicate_threshold=settings.duplicate_threshold,
    )
