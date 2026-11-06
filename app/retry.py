"""Retry model calls when the provider is busy or rate-limiting.

Hosted models sometimes answer with 503 "high demand" or 429 "quota" errors.
Those usually clear within seconds, so each call is retried with exponential
backoff. Anything else (a bad key, a wrong model name) fails straight away.
"""

from __future__ import annotations

import logging
import random
import time

log = logging.getLogger(__name__)

TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "ratelimit",
    "quota",
    "overloaded",
    "high demand",
    "timeout",
    "timed out",
    "temporarily",
    "try again",
)


class ModelBusyError(RuntimeError):
    """The model stayed busy after every retry."""


def is_transient(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


class RetryingChatModel:
    """Wraps a chat model so invoke() retries transient errors."""

    def __init__(self, model, max_retries: int = 4, base_delay: float = 1.5, max_delay: float = 20.0, sleep=time.sleep):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep

    def invoke(self, prompt, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return self.model.invoke(prompt, *args, **kwargs)
            except Exception as error:  # noqa: BLE001 - provider SDKs raise many types
                if not is_transient(error):
                    raise
                if attempt >= self.max_retries:
                    raise ModelBusyError(
                        "The model is busy right now. Please try again in a moment."
                    ) from error
                delay = min(self.max_delay, self.base_delay * (2**attempt)) * random.uniform(0.8, 1.2)
                attempt += 1
                log.warning("Model busy (%s). Retry %d/%d in %.1fs", type(error).__name__, attempt, self.max_retries, delay)
                self._sleep(delay)

    def __getattr__(self, name):
        return getattr(self.model, name)
