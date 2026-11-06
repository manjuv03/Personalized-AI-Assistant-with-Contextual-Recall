import time

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
