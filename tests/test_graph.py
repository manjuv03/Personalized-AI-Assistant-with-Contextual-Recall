"""End-to-end test of the agent with a scripted model (no API key needed)."""

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm import HashEmbeddings  # noqa: E402
from app.logs import InteractionLog  # noqa: E402
from app.memory.long_term import InMemoryBackend, LongTermMemory  # noqa: E402
from app.service import Assistant  # noqa: E402


class ScriptedModel:
    """Returns canned replies and records every prompt it was given."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = prompt if isinstance(prompt, str) else "\n".join(str(m.content) for m in prompt)
        if "maintain long-term memory" in text:
            if "Priya" in text:
                return AIMessage(content='[{"text": "The user\'s name is Priya.", "category": "profile", "importance": 0.9}]')
            return AIMessage(content="[]")
        if "Summarise this conversation" in text:
            return AIMessage(content="Earlier the user chatted about the weather.")
        return AIMessage(content="Noted!")


@pytest.fixture
def assistant():
    memory = LongTermMemory(InMemoryBackend(), HashEmbeddings(), min_score=0.0)
    return Assistant(Settings(keep_recent_messages=4), chat_model=ScriptedModel(), memory=memory, log=InteractionLog(":memory:"))


def test_fact_from_one_thread_is_recalled_in_another(assistant):
    first = assistant.chat("u1", "t1", "Hi, I'm Priya!")
    assistant.remember("u1", "Hi, I'm Priya!", first.log_id)

    result = assistant.chat("u1", "t2", "What's my name?")
    assert any("Priya" in m["text"] for m in result.recalled)
    system_prompt = assistant.chat_model.prompts[-1][0].content
    assert "Priya" in system_prompt


def test_memory_can_be_switched_off(assistant):
    assistant.remember("u1", "Hi, I'm Priya!")
    result = assistant.chat("u1", "t2", "What's my name?", use_memory=False)
    assert result.recalled == []


def test_short_threads_are_not_summarised(assistant):
    for i in range(3):  # 6 messages, below the trigger of 2 x 4
        assistant.chat("u1", "t1", f"message {i}")
    assert assistant.summary("u1", "t1") == ""


def test_long_threads_are_summarised(assistant):
    for i in range(4):  # 8 messages reaches the trigger, trimmed back to 4
        assistant.chat("u1", "t1", f"message {i}")
    assert assistant.summary("u1", "t1") == "Earlier the user chatted about the weather."
    state = assistant.graph.get_state({"configurable": {"thread_id": "u1:t1"}})
    assert len(state.values["messages"]) <= 4
