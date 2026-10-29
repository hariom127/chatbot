import json
import logging
import random
import zlib

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, embeddings, llm, small_talk_kb, vector_store
from app.services.conversational import normalize
from app.services.embeddings import EmbeddingError
from app.services.vector_store import VectorStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(conversational, "_rng", random.Random(1234))
    monkeypatch.setattr(conversational, "_last_used", {})
    monkeypatch.setattr(settings, "project_name", "")
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)
    monkeypatch.setattr(settings, "small_talk_kb_enabled", True)
    monkeypatch.setattr(settings, "small_talk_max_words", 8)
    monkeypatch.setattr(settings, "small_talk_fuzzy_threshold", 0.85)
    monkeypatch.setattr(settings, "small_talk_fuzzy_margin", 0.15)


@pytest.fixture
def no_embedding(monkeypatch):
    """Turn the semantic layer on, but fail if anything is embedded."""
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    monkeypatch.setattr(embeddings, "embed_texts", lambda t: pytest.fail("must not embed"))
    monkeypatch.setattr(embeddings, "embed_query", lambda t: pytest.fail("must not embed"))


@pytest.fixture
def no_rag(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))


def key(message: str) -> str:
    return small_talk_kb.fuzzy_key(normalize(message))


def matched(message: str):
    m = conversational._match(message)
    return m and (m.entry_id, m.layer)


# --- fuzzy key ---

@pytest.mark.parametrize("raw,expected", [
    ("hiii", "hi"), ("thankssss", "thanks"), ("hello", "hello"), ("good", "good"), ("see", "see"),
    ("u", "you"), ("hw r u", "how are you"), ("ur", "your"), ("thx", "thanks"), ("gn", "good night"),
    ("wassup", "whats up"), ("gud", "good"), ("plz", "please"),
    ("whts", "whats"), ("helo", "hello"), ("thnks", "thanks"), ("goin", "going"), ("frnd", "friend"),
    ("heyyy whts goin on", "hey whats going on"), ("thank uuu sooo much", "thank you so much"),
])
def test_fuzzy_key(raw, expected):
    assert key(raw) == expected


@pytest.mark.parametrize("word", ["what", "invoice", "invoices", "status", "order", "due", "dude", "payment",
                                  "refund", "tnx", "hy", "neptune"])
def test_words_left_alone(word):
    # vocabulary words, anchor words, words under 4 letters and unrelated words are never "corrected"
    assert key(word) == word


def test_dude_never_becomes_due():
    assert "due" not in key("hey dude").split()


def test_digits_never_corrected():
    assert key("thnks2") == "thnks2"


# --- step 2a decisions ---

@pytest.mark.parametrize("message,entry_id", [
    ("Hiii there we whats going on???", "whats_going_on"),
    ("heyyy whts goin on", "whats_going_on"),
    ("helo ther", "greeting"),
    ("thnks a lott", "gratitude"),
    ("hw r u doing", "how_are_you"),
    ("hows ur day goin", "how_are_you"),
    ("im back again", "returning"),
    ("hahaha", "laughter"),
])
def test_fuzzy_accepts(no_embedding, message, entry_id):
    m = conversational._match(message)
    assert m.entry_id == entry_id and m.layer in ("fuzzy", "fuzzy-rules")


@pytest.mark.parametrize("message,entry_id", [
    ("thank uuu sooo much", "gratitude"), ("hellooo frnd", "greeting"), ("heyy wassup dude", "whats_going_on"),
    ("gud mornin", "time_greeting"),
])
def test_rules_on_fuzzy_key(no_embedding, message, entry_id):
    assert matched(message)[0] == entry_id


def test_fuzzy_key_salutation():
    assert conversational.conversational_reply("gud mornin")["answer"].startswith("Good morning!")


@pytest.mark.parametrize("message", [
    "Hi, what is an invoice?", "whats going on with my order", "hey whats the status", "hey whats the late fee",
    "what is going on", "any update on my request", "Neptune?", "help with invoices", "dude where is my invoice",
    "what is this", "good question", "morning shift timings",
])
def test_fuzzy_rejects_real_questions(message):
    assert conversational._match(message) is None


def test_whats_going_on_reply_comes_from_its_entry(no_rag):
    allowed = {conversational.render_reply(t) for t in conversational.replies("whats_going_on")}
    for _ in range(4):
        body = client.post("/chat", json={"question": "Hiii there we whats going on???"}).json()
        assert body["answer"] in allowed and body["found"] is True and body["sources"] == []


def test_scores_kept_for_near_misses():
    s = small_talk_kb.fuzzy_scores(key("what is going on"))
    assert s.entry_id == "whats_going_on" and s.f_social > s.f_question
    assert not small_talk_kb.fuzzy_accepts(s)


# --- gates ---

@pytest.mark.parametrize("message", ["helo ther 123", "helo ther " + "x" * 55, "helo ther " * 5])
def test_gates_skip_fuzzy(message):
    assert not small_talk_kb.is_eligible(message)
    assert conversational._match(message) is None


def test_kb_flag_off_skips_fuzzy_but_rules_work(monkeypatch):
    monkeypatch.setattr(settings, "small_talk_kb_enabled", False)
    assert conversational._match("helo ther") is None
    assert matched("hello there") == ("greeting", "rules")


def test_max_words_setting(monkeypatch):
    monkeypatch.setattr(settings, "small_talk_max_words", 1)
    assert conversational._match("helo ther") is None


def test_broken_kb_skips_fuzzy(monkeypatch):
    monkeypatch.setattr(conversational, "knowledge_base", lambda: None)
    assert not small_talk_kb.is_eligible("helo ther")
    assert conversational._match("helo ther") is None
    assert matched("hi") == ("greeting", "rules")


def test_fuzzy_error_falls_back_to_rag(monkeypatch):
    def boom(key):
        raise RuntimeError("fuzzy bug")

    monkeypatch.setattr(small_talk_kb, "fuzzy_scores", boom)
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    r = client.post("/chat", json={"question": "hows ur day goin"})
    assert r.status_code == 200 and seen == ["hows ur day goin"]


def test_index_rebuilt_when_kb_changes(monkeypatch):
    kb = conversational.knowledge_base()
    first = small_talk_kb._index()
    assert small_talk_kb._index() is first
    monkeypatch.setattr(conversational, "knowledge_base", lambda: kb._replace(anchors=list(kb.anchors)))
    assert small_talk_kb._index() is not first


# --- routing and logging ---

@pytest.mark.parametrize("question", ["heyyy whts goin on", "helo ther", "thnks a lott", "thank uuu sooo much"])
def test_fuzzy_match_skips_rag(no_rag, no_embedding, question):
    r = client.post("/chat", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []


def test_fuzzy_log_has_scores_not_text(caplog):
    with caplog.at_level(logging.DEBUG):
        conversational.conversational_reply("hows ur day goin")
        conversational.conversational_reply("what is going on")
    assert "layer=fuzzy f_social=" in caplog.text and "near miss" in caplog.text
    assert "ur day" not in caplog.text and "what is going on" not in caplog.text


# --- step 2b: semantic matching with a fake embedder and a temp Chroma dir ---

GREET = {"hey", "heya", "hello", "hi", "folks", "howdy", "hiya", "greetings", "namaste", "hola", "yo"}
THANKS = {"thanks", "thank", "cheers", "appreciate", "appreciated", "grateful"}
DOC = {"invoice", "invoices", "order", "orders", "refund", "policy", "payment", "payments", "statuses", "status"}
HASH_DIMS = 32


def fake_vector(text: str) -> list[float]:
    """Dimensions: greeting-like, gratitude-like, question-like, then one hashed bucket per other word."""
    vector = [0.0] * (3 + HASH_DIMS)
    for w in text.lower().split():
        if w in GREET:
            vector[0] += 1
        elif w in THANKS:
            vector[1] += 1
        elif w in DOC:
            vector[2] += 1
        else:
            vector[3 + zlib.crc32(w.encode()) % HASH_DIMS] += 1
    return vector or [1.0]


class FakeEmbedder:
    def __init__(self, fail=False):
        self.batches = []  # sizes of embed_texts calls
        self.fail = fail

    def __call__(self, texts):
        self.batches.append(len(texts))
        if self.fail:
            raise EmbeddingError("provider down")
        return [fake_vector(t) for t in texts]

    @property
    def builds(self):
        return sum(n > 1 for n in self.batches)


@pytest.fixture
def semantic(tmp_settings, monkeypatch):
    monkeypatch.setattr(small_talk_kb, "_ready", None)
    monkeypatch.setattr(small_talk_kb, "_failed_at", None)
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    monkeypatch.setattr(settings, "small_talk_semantic_threshold", 0.75)
    monkeypatch.setattr(settings, "small_talk_semantic_margin", 0.10)
    embedder = FakeEmbedder()
    monkeypatch.setattr(embeddings, "embed_texts", embedder)  # embed_query goes through it too
    return embedder


def expected_records():
    kb = conversational.knowledge_base()
    return sum(len(e.examples) for e in kb.entries.values() if not e.rules_only) + len(kb.anchors)


def test_semantic_accepts_clear_match(semantic):
    m = conversational._match("heya folks")
    assert m.entry_id == "greeting" and m.layer == "semantic"


def test_semantic_rejects_question_like(semantic):
    for message in ["invoice refund policy", "folks invoice status"]:
        assert conversational._match(message) is None
    s = small_talk_kb.semantic_scores(key("invoice refund policy"))
    assert s.s_question > s.s_social


@pytest.mark.parametrize("scores,accepted", [
    (small_talk_kb.SemanticScores("greeting", 0.90, 0.70), True),
    (small_talk_kb.SemanticScores("greeting", 0.90, 0.85), False),  # inside the margin
    (small_talk_kb.SemanticScores("greeting", 0.70, 0.00), False),  # below the threshold
    (small_talk_kb.SemanticScores(None, 0.0, 0.90), False),  # only anchors nearby
])
def test_semantic_decision(semantic, scores, accepted):
    assert small_talk_kb.semantic_accepts(scores) is accepted


@pytest.mark.parametrize("message", ["heya folks " * 5, "heya folks 2", "heya" + "a" * 60])
def test_gates_skip_embedding(semantic, message):
    assert conversational._match(message) is None
    assert semantic.batches == []


@pytest.mark.parametrize("flag", ["small_talk_semantic_enabled", "small_talk_kb_enabled"])
def test_flags_off_skip_embedding(semantic, monkeypatch, flag):
    monkeypatch.setattr(settings, flag, False)
    assert conversational._match("heya folks") is None
    assert semantic.batches == []


def test_master_flag_off_skips_embedding(semantic, monkeypatch):
    monkeypatch.setattr(settings, "conversational_replies_enabled", False)
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [])
    client.post("/chat", json={"question": "heya folks"})
    assert semantic.batches == []


def test_rule_and_fuzzy_matches_skip_embedding(semantic):
    assert conversational._match("hi").layer == "rules"
    assert conversational._match("helo ther").layer == "fuzzy-rules"
    assert conversational._match("hows ur day goin").layer == "fuzzy"
    assert semantic.batches == []


def test_collection_built_once_with_examples_and_anchors(semantic):
    for _ in range(5):
        conversational._match("heya folks")
    assert semantic.builds == 1 and semantic.batches[0] == expected_records()
    col = small_talk_kb._client().get_collection(small_talk_kb.collection_name())
    assert col.count() == expected_records()
    assert col.metadata["kb_hash"] == conversational.knowledge_base().hash
    rows = col.get(include=["metadatas"])
    assert not any(i.startswith("time_greeting:") for i in rows["ids"])  # rules_only entries stay out
    kinds = [m["kind"] for m in rows["metadatas"]]
    assert kinds.count("anchor") == len(conversational.knowledge_base().anchors)


def test_existing_collection_reused_after_restart(semantic, monkeypatch):
    conversational._match("heya folks")
    monkeypatch.setattr(small_talk_kb, "_ready", None)  # a new process
    conversational._match("heya folks")
    assert semantic.builds == 1


def test_rebuilt_when_file_changes(semantic, tmp_path, monkeypatch):
    conversational._match("heya folks")
    data = json.loads(conversational.KB_PATH.read_text(encoding="utf-8"))
    next(e for e in data["entries"] if e["id"] == "greeting")["examples"].append("hey yall")
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(conversational, "KB_PATH", path)
    conversational.knowledge_base.cache_clear()
    try:
        conversational._match("heya folks")
        assert semantic.builds == 2 and semantic.batches[-2] == expected_records()
    finally:
        conversational.knowledge_base.cache_clear()


def test_rebuilt_when_embedding_model_changes(semantic, monkeypatch):
    conversational._match("heya folks")
    first = small_talk_kb.collection_name()
    monkeypatch.setattr(settings, "embedding_model", "another-local-model")
    conversational._match("heya folks")
    assert small_talk_kb.collection_name() != first and semantic.builds == 2


def test_drive_docs_and_sync_status_untouched(semantic):
    store = VectorStore()
    status = client.get("/sync/status").json()
    conversational._match("heya folks")
    assert VectorStore().count() == store.count() == 0
    assert client.get("/sync/status").json() == status
    names = {c.name for c in small_talk_kb._client().list_collections()}
    assert vector_store.collection_name() in names and small_talk_kb.collection_name() in names


def test_embedder_failure_starts_cooldown_and_fuzzy_keeps_working(semantic, monkeypatch):
    semantic.fail = True
    now = [1000.0]
    monkeypatch.setattr(small_talk_kb, "_clock", lambda: now[0])
    assert conversational._match("heya folks") is None
    assert conversational._match("heya folks") is None
    assert len(semantic.batches) == 1  # paused after the first failure
    assert conversational._match("hows ur day goin").layer == "fuzzy"
    now[0] += small_talk_kb.FAILURE_COOLDOWN_SECONDS + 1
    semantic.fail = False
    assert conversational._match("heya folks").layer == "semantic"


def test_chroma_failure_starts_cooldown(semantic, monkeypatch):
    def broken():
        raise RuntimeError("chroma down")

    monkeypatch.setattr(small_talk_kb, "_client", broken)
    assert conversational._match("heya folks") is None
    assert small_talk_kb._failed_at is not None


def test_semantic_match_skips_rag(semantic, no_rag):
    body = client.post("/chat", json={"question": "heya folks"}).json()
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []


def test_semantic_failure_falls_back_to_rag(semantic, monkeypatch):
    semantic.fail = True
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    r = client.post("/chat", json={"question": "heya folks"})
    assert r.status_code == 200 and seen == ["heya folks"]


# --- real local model (same convention as test_indexing_retrieval.py) ---

@pytest.fixture
def real_model(tmp_settings, monkeypatch):
    """The shipped defaults, whatever .env says, so these tests prove the defaults suit MiniLM."""
    monkeypatch.setattr(small_talk_kb, "_ready", None)
    monkeypatch.setattr(small_talk_kb, "_failed_at", None)
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    for name in ("small_talk_semantic_threshold", "small_talk_semantic_margin", "embedding_model"):
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)


def semantic_entry(message: str) -> str | None:
    scores = small_talk_kb.semantic_scores(key(message))
    return scores.entry_id if scores and small_talk_kb.semantic_accepts(scores) else None


@pytest.mark.usefixtures("real_model")
@pytest.mark.parametrize("message,entry_id", [
    ("many thanks", "gratitude"), ("catch you later", "farewell"), ("sounds good", "acknowledgement"),
    ("how is your day going", "how_are_you"), ("are you a real person", "identity"),
])
def test_real_model_small_talk(message, entry_id):
    assert semantic_entry(message) == entry_id


@pytest.mark.usefixtures("real_model")
@pytest.mark.parametrize("message", [
    "what is an invoice", "hi can you explain invoice statuses", "how do refunds work", "help with invoices",
])
def test_real_model_questions_go_to_rag(message):
    assert semantic_entry(message) is None
