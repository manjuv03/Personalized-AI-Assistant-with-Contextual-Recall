"""Model factories. Providers are imported lazily so only the one you use needs installing."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddings:
    """Deterministic bag-of-words embeddings using the hashing trick.

    No model download and no API key, so tests and offline demos run anywhere.
    It only captures word overlap, so use a real embedding model for actual use.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "hash":
        return HashEmbeddings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")


def get_chat_model(settings: Settings):
    """The configured chat model, wrapped so busy/rate-limit errors are retried."""
    from .retry import RetryingChatModel

    return RetryingChatModel(_raw_chat_model(settings), max_retries=settings.llm_max_retries)


def _raw_chat_model(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.llm_model, temperature=settings.llm_temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
