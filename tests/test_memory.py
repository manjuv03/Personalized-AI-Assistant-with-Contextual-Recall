import time

import pytest

from app.llm import HashEmbeddings
from app.logs import InteractionLog
from app.memory.extractor import parse_extracted_facts
from app.memory.long_term import InMemoryBackend, LongTermMemory, rank_score, recency_weight
from app.memory.short_term import split_for_summary
from eval.metrics import contains_any


def make_memory(**kwargs) -> LongTermMemory:
    return LongTermMemory(InMemoryBackend(), HashEmbeddings(), **kwargs)


# --- long-term memory -------------------------------------------------------

def test_recall_finds_relevant_fact():
    mem = make_memory(min_score=0.0)
    mem.add("u1", "The user's name is Priya.", "profile", 0.9)
    mem.add("u1", "The user is training for a half marathon.", "goal", 0.7)
    hits = mem.recall("u1", "what is my name")
    assert hits and "Priya" in hits[0].record.text


def test_users_are_isolated():
    mem = make_memory(min_score=0.0)
    mem.add("alice", "The user's favourite colour is green.")
    assert mem.recall("bob", "favourite colour") == []
    assert len(mem.list("alice")) == 1


def test_near_duplicate_updates_instead_of_adding():
    mem = make_memory(duplicate_threshold=0.9)
    first = mem.add("u1", "The user lives in Leeds.", importance=0.4)
    second = mem.add("u1", "The user lives in Leeds.", importance=0.8)
    assert first.id == second.id
    assert len(mem.list("u1")) == 1
    assert mem.list("u1")[0].importance == 0.8


def test_delete_only_own_memory():
    mem = make_memory()
    rec = mem.add("u1", "The user has a cat called Biscuit.")
    assert not mem.delete("u2", rec.id)
    assert mem.delete("u1", rec.id)
    assert mem.list("u1") == []


def test_unknown_category_falls_back_to_other():
    mem = make_memory()
    assert mem.add("u1", "The user likes jazz.", category="music").category == "other"


def test_projection_returns_2d_points():
    mem = make_memory()
    for fact in ["The user likes tea.", "The user codes in Go.", "The user lives in Madrid."]:
        mem.add("u1", fact)
    points = mem.projection("u1")
    assert len(points) == 3
    assert all({"x", "y", "text"} <= p.keys() for p in points)


def test_recency_halves_after_half_life():
    now = time.time()
    assert recency_weight(now, now) == 1.0
    assert abs(recency_weight(now - 30 * 86400, now) - 0.5) < 1e-9


def test_importance_breaks_similarity_ties():
    now = time.time()
    assert rank_score(0.6, 0.9, now, now) > rank_score(0.6, 0.1, now, now)


# --- extraction parsing -----------------------------------------------------

def test_parse_handles_code_fences_and_clamps():
    raw = '```json\n[{"text": "The user is vegetarian.", "category": "preference", "importance": 3}]\n```'
    facts = parse_extracted_facts(raw)
    assert facts == [{"text": "The user is vegetarian.", "category": "preference", "importance": 1.0}]


def test_parse_empty_and_garbage():
    assert parse_extracted_facts("") == []
    assert parse_extracted_facts("[]") == []
    assert parse_extracted_facts("no json here") == []
    assert parse_extracted_facts("[{broken") == []


def test_parse_accepts_plain_strings():
    assert parse_extracted_facts('["The user lives in Bristol."]')[0]["category"] == "other"


# --- short-term memory ------------------------------------------------------

def test_split_keeps_recent_window():
    msgs = list(range(10))
    old, recent = split_for_summary(msgs, keep_recent=4)
    assert old == [0, 1, 2, 3, 4, 5] and recent == [6, 7, 8, 9]


def test_split_does_nothing_when_short():
    assert split_for_summary([1, 2, 3], keep_recent=8) == ([], [1, 2, 3])


# --- logs and eval helpers --------------------------------------------------

def test_interaction_log_roundtrip():
    log = InteractionLog(":memory:")
    log_id = log.add("u1", "main", "hi", "hello", [{"text": "x"}], 123.4)
    log.set_stored(log_id, [{"text": "The user said hi."}])
    item = log.recent("u1")[0]
    assert item["reply"] == "hello" and item["stored"][0]["text"] == "The user said hi."
    assert log.stats("u1") == {"turns": 1, "avg_latency_ms": 123.4}


def test_contains_any_whole_words_and_prefix():
    assert contains_any("You write in Go.", ["go"])
    assert not contains_any("That sounds good", ["go"])
    assert contains_any("You're allergic to peanuts", ["allerg*"])


def test_message_text_handles_content_blocks():
    from app.llm import message_text

    class Msg:
        content = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hi "}, "Manju", {"type": "text", "text": "!", "extras": {"signature": "abc"}}]

    assert message_text(Msg()) == "Hi Manju!"
    assert message_text("plain") == "plain"


# --- editing memories, tools and chats --------------------------------------

def test_update_changes_text_category_and_importance():
    mem = make_memory(min_score=0.0)
    rec = mem.add("u1", "The user lives in Leeds.", "profile", 0.5)
    updated = mem.update("u1", rec.id, text="The user lives in London.", category="profile", importance=0.9)
    assert updated.text == "The user lives in London." and updated.importance == 0.9
    assert mem.recall("u1", "where do I live London")[0].record.text == "The user lives in London."
    assert mem.update("u2", rec.id, text="hijack") is None


def test_tools_save_search_and_forget():
    pytest.importorskip("langchain_core")
    from app.memory.tools import run_tool

    mem = make_memory(min_score=0.0)
    out, event = run_tool(mem, "u1", "save_memory", {"text": "The user's sister is called Asha.", "category": "profile"})
    assert event["label"].startswith("Saved") and len(mem.list("u1")) == 1
    out, event = run_tool(mem, "u1", "search_memory", {"query": "sister"})
    memory_id = mem.list("u1")[0].id
    assert memory_id in out
    out, event = run_tool(mem, "u2", "forget_memory", {"memory_id": memory_id})
    assert "failed" in event["label"] and len(mem.list("u1")) == 1  # other users can't delete it
    out, event = run_tool(mem, "u1", "forget_memory", {"memory_id": f"[{memory_id}]"})
    assert event["label"].startswith("Forgot") and mem.list("u1") == []


def test_split_never_starts_with_a_tool_result():
    msgs = ["H", "A", "H", "A", "T", "T", "A", "H", "A", "H"]
    old, recent = split_for_summary(msgs, keep_recent=4, is_start=lambda m: m == "H")
    assert recent[0] == "H" and old + recent == msgs


def test_log_lists_threads_and_migrates_old_tables(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, user_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL, user_message TEXT NOT NULL, reply TEXT NOT NULL, recalled TEXT NOT NULL,"
        " stored TEXT NOT NULL DEFAULT '[]', latency_ms REAL NOT NULL)"
    )
    con.execute("INSERT INTO interactions (ts, user_id, thread_id, user_message, reply, recalled, latency_ms)"
                " VALUES (1, 'u1', 'old', 'Hello from before', 'Hi', '[]', 900)")
    con.commit()
    con.close()

    log = InteractionLog(str(path))
    log.add("u1", "new", "A new chat", "Sure", [], 500, tools=[{"name": "save_memory", "label": "Saved"}], first_token_ms=120)
    threads = log.threads("u1")
    assert [t["thread_id"] for t in threads] == ["new", "old"]
    assert threads[1]["title"] == "Hello from before"
    turn = log.thread_turns("u1", "new")[0]
    assert turn["tools"][0]["name"] == "save_memory" and turn["first_token_ms"] == 120
    assert log.thread_turns("u1", "old")[0]["tools"] == []
    assert log.delete_thread("u1", "old") == 1
