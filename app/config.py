"""Settings, read once from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so python-dotenv is not a hard requirement."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Which chat model to use: "openai", "gemini" or "ollama".
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.3")))

    # Embeddings: "openai", "gemini", "huggingface" or "hash" (offline, for tests/demos).
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "huggingface"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Vector store: "chroma" (persistent) or "memory" (in-process, for tests).
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "chroma"))
    chroma_path: str = field(default_factory=lambda: _env("CHROMA_PATH", "data/chroma"))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "user_memories"))

    # Long-term recall.
    recall_top_k: int = field(default_factory=lambda: int(_env("RECALL_TOP_K", "5")))
    recall_min_score: float = field(default_factory=lambda: float(_env("RECALL_MIN_SCORE", "0.25")))
    duplicate_threshold: float = field(default_factory=lambda: float(_env("DUPLICATE_THRESHOLD", "0.92")))

    # Short-term (contextual) memory: keep this many recent messages verbatim,
    # fold anything older into a running summary.
    keep_recent_messages: int = field(default_factory=lambda: int(_env("KEEP_RECENT_MESSAGES", "8")))

    # Interaction logs.
    log_db_path: str = field(default_factory=lambda: _env("LOG_DB_PATH", "data/interactions.sqlite"))


settings = Settings()
