import os

import pytest

from app.config import _load_dotenv
from app.retry import ModelBusyError, RetryingChatModel, is_transient


class FlakyModel:
    def __init__(self, failures, error):
        self.failures = failures
        self.error = error
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


BUSY = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")


def test_retries_then_succeeds():
    model = FlakyModel(failures=2, error=BUSY)
    wrapped = RetryingChatModel(model, max_retries=4, sleep=lambda _: None)
    assert wrapped.invoke("hi") == "ok"
    assert model.calls == 3


def test_gives_up_with_busy_error():
    model = FlakyModel(failures=10, error=BUSY)
    wrapped = RetryingChatModel(model, max_retries=2, sleep=lambda _: None)
    with pytest.raises(ModelBusyError):
        wrapped.invoke("hi")
    assert model.calls == 3


def test_does_not_retry_real_errors():
    model = FlakyModel(failures=1, error=ValueError("API key not valid"))
    wrapped = RetryingChatModel(model, max_retries=4, sleep=lambda _: None)
    with pytest.raises(ValueError):
        wrapped.invoke("hi")
    assert model.calls == 1


def test_transient_detection():
    class RemoteProtocolError(Exception):
        pass

    assert is_transient(RemoteProtocolError("Server disconnected without sending a response."))
    assert is_transient(RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded"))
    assert not is_transient(RuntimeError("404 model not found"))


def test_dotenv_ignores_inline_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A_TEST_VAR=4   # a comment\nB_TEST_VAR="x # y"\nC_TEST_VAR=abc#def\n')
    for key in ("A_TEST_VAR", "B_TEST_VAR", "C_TEST_VAR"):
        monkeypatch.delenv(key, raising=False)
    _load_dotenv(env)
    assert os.environ["A_TEST_VAR"] == "4"
    assert os.environ["B_TEST_VAR"] == "x # y"
    assert os.environ["C_TEST_VAR"] == "abc#def"
