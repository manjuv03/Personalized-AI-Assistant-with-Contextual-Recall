"""End-to-end tests of the agent with scripted models (no API key needed)."""

import re

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm import HashEmbeddings  # noqa: E402
from app.logs import InteractionLog  # noqa: E402
from app.memory.long_term import InMemoryBackend, LongTermMemory  # noqa: E402
from app.service import Assistant  # noqa: E402


class ScriptedModel:
    """Returns canned replies (and tool calls) and records every prompt it was given."""

    def __init__(self):
        self.prompts = []
        self.tools_bound = False

    def bind_tools(self, tools, **kwargs):
        bound = ScriptedModel()
        bound.prompts = self.prompts
        bound.tools_bound = True
        return bound

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = prompt if isinstance(prompt, str) else "\n".join(str(m.content) for m in prompt)
        if "maintain long-term memory" in text:
            if "Priya" in text:
                return AIMessage(content='[{"text": "The user\'s name is Priya.", "category": "profile", "importance": 0.9}]')
            return AIMessage(content="[]")
        if "Summarise this conversation" in text:
            return AIMessage(content="Earlier the user chatted about the weather.")

        last = prompt[-1]
        if self.tools_bound and isinstance(last, HumanMessage) and "Remember that my sister" in last.content:
            return AIMessage(content="", tool_calls=[{
                "name": "save_memory", "id": "c1",
                "args": {"text": "The user's sister is called Asha.", "category": "profile", "importance": 0.8},
            }])
        if self.tools_bound and isinstance(last, HumanMessage) and "Forget my sister" in last.content:
            return AIMessage(content="", tool_calls=[{"name": "search_memory", "id": "c2", "args": {"query": "sister"}}])
        if self.tools_bound and isinstance(last, ToolMessage) and last.name == "search_memory":
            found = re.search(r"\[([0-9a-f]{32})\] The user's sister", last.content)
            return AIMessage(content="", tool_calls=[{"name": "forget_memory", "id": "c3", "args": {"memory_id": found.group(1)}}])
        if isinstance(last, ToolMessage):
            return AIMessage(content="Done.")
        return AIMessage(content="Noted!")


def make_assistant(**settings):
    memory = LongTermMemory(InMemoryBackend(), HashEmbeddings(), min_score=0.0)
    s = Settings(keep_recent_messages=4, checkpoint_path=":memory:", **settings)
    return Assistant(s, chat_model=ScriptedModel(), memory=memory, log=InteractionLog(":memory:"))


@pytest.fixture
def assistant():
    return make_assistant()


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


def test_tool_saves_a_memory(assistant):
    result = assistant.chat("u1", "t1", "Remember that my sister is called Asha.")
    assert result.reply == "Done."
    assert [t["name"] for t in result.tools] == ["save_memory"]
    assert any("Asha" in m.text for m in assistant.memory.list("u1"))


def test_tools_search_then_forget(assistant):
    assistant.memory.add("u1", "The user's sister is called Asha.", "profile", 0.8)
    result = assistant.chat("u1", "t1", "Forget my sister's name please.")
    assert [t["name"] for t in result.tools] == ["search_memory", "forget_memory"]
    assert not any("Asha" in m.text for m in assistant.memory.list("u1"))


def test_tool_events_reset_each_turn(assistant):
    assistant.chat("u1", "t1", "Remember that my sister is called Asha.")
    result = assistant.chat("u1", "t1", "Thanks!")
    assert result.tools == []


def test_tools_can_be_turned_off():
    a = make_assistant(agent_tools=False)
    result = a.chat("u1", "t1", "Remember that my sister is called Asha.")
    assert result.tools == [] and result.reply == "Noted!"


def test_summary_keeps_tool_results_with_their_calls(assistant):
    for _ in range(3):
        assistant.chat("u1", "t1", "Remember that my sister is called Asha.")
    state = assistant.graph.get_state({"configurable": {"thread_id": "u1:t1"}})
    assert isinstance(state.values["messages"][0], HumanMessage)


def test_chats_are_listed_and_deleted(assistant):
    assistant.chat("u1", "t1", "First chat")
    assistant.chat("u1", "t2", "Second chat")
    assistant.chat("u1", "t2", "Still second")
    threads = assistant.threads("u1")
    assert [t["thread_id"] for t in threads] == ["t2", "t1"]
    assert threads[0]["title"] == "Second chat" and threads[0]["turns"] == 2
    assert [h["user_message"] for h in assistant.thread_history("u1", "t2")] == ["Second chat", "Still second"]
    assert assistant.delete_thread("u1", "t2")
    assert [t["thread_id"] for t in assistant.threads("u1")] == ["t1"]


def test_stream_sends_recall_tokens_and_done():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello there, Priya!")]))
    memory = LongTermMemory(InMemoryBackend(), HashEmbeddings(), min_score=0.0)
    memory.add("u1", "The user's name is Priya.", "profile", 0.9)
    a = Assistant(
        Settings(keep_recent_messages=4, checkpoint_path=":memory:", agent_tools=False),
        chat_model=model, memory=memory, log=InteractionLog(":memory:"),
    )
    events = list(a.chat_stream("u1", "t1", "What's my name?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "recalled" and kinds[-1] == "done"
    assert "token" in kinds
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello there, Priya!"
    done = events[-1]
    assert done["reply"] == "Hello there, Priya!"
    assert done["first_token_ms"] is not None
    assert a.thread_history("u1", "t1")[0]["reply"] == "Hello there, Priya!"
