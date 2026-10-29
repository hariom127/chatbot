import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, conversational_semantic, embeddings, llm
from app.services.conversational_semantic import semantic_intent
from app.services.embeddings import EmbeddingError

client = TestClient(app)

GREET = {"hey", "heya", "hello", "hi", "folks"}
THANKS = {"thanks", "thank", "cheers", "appreciate"}
DOC = {"invoice", "invoices", "order", "orders", "refund", "policy", "payment", "payments", "statuses"}


def fake_vector(text: str) -> list[float]:
    """Dimensions: greeting-like, gratitude-like, question-like, anything else."""
    words = text.lower().split()
    known = GREET | THANKS | DOC
    return [sum(w in GREET for w in words), sum(w in THANKS for w in words),
            sum(w in DOC for w in words), sum(w not in known for w in words)]


class FakeEmbedder:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def __call__(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("provider down")
        return [fake_vector(t) for t in texts]


@pytest.fixture(autouse=True)
def fresh_semantic(monkeypatch):
    # the fake-embedder cases are built around these values; real-model tests switch to the defaults
    monkeypatch.setattr(conversational_semantic, "_reference", None)
    monkeypatch.setattr(conversational_semantic, "_failed_at", None)
    monkeypatch.setattr(settings, "semantic_small_talk_enabled", True)
    monkeypatch.setattr(settings, "semantic_small_talk_threshold", 0.75)
    monkeypatch.setattr(settings, "semantic_small_talk_margin", 0.10)
    monkeypatch.setattr(settings, "semantic_small_talk_max_words", 6)
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "project_name", "")


@pytest.fixture
def fake(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr(embeddings, "embed_texts", embedder)
    return embedder


# --- fake embedder ---

def test_clear_match_returns_intent(fake):
    match = semantic_intent("heya folks")
    assert match.intent == "greeting" and match.score == pytest.approx(1.0) and match.margin >= 0.10


def test_match_inside_margin_returns_none(fake):
    # as close to the invoice anchors as to the greeting examples
    assert semantic_intent("heya invoice") is None


def test_below_threshold_returns_none(fake):
    # halfway between greeting and gratitude (0.71 to each) and far from the anchors
    scores = conversational_semantic.score("heya cheers")
    assert scores.s_best < 0.75 and scores.s_best - scores.s_doc >= 0.10
    assert semantic_intent("heya cheers") is None


@pytest.mark.parametrize("message", ["heya folks heya folks heya folks heya", "heya 2", "heya" + "a" * 60])
def test_gates_skip_embedding(fake, message):
    assert semantic_intent(message) is None
    assert fake.calls == 0


def test_flag_off_skips_embedding(fake, monkeypatch):
    monkeypatch.setattr(settings, "semantic_small_talk_enabled", False)
    assert conversational.conversational_reply("heya folks") is None
    assert fake.calls == 0


def test_rule_match_skips_embedding(fake):
    assert conversational.conversational_reply("hi")["found"] is True
    assert fake.calls == 0


def test_reference_cached_per_embedding_model(fake, monkeypatch):
    for _ in range(5):
        semantic_intent("heya folks")
    assert fake.calls == 1 + 5  # one reference batch, then one call per message
    monkeypatch.setattr(settings, "embedding_model", "another-local-model")
    semantic_intent("heya folks")
    assert fake.calls == 6 + 2  # reference rebuilt for the new embedding id


def test_failing_embedder_starts_cooldown(monkeypatch):
    embedder = FakeEmbedder(fail=True)
    monkeypatch.setattr(embeddings, "embed_texts", embedder)
    now = [1000.0]
    monkeypatch.setattr(conversational_semantic, "_clock", lambda: now[0])
    assert semantic_intent("heya folks") is None
    assert semantic_intent("heya folks") is None
    assert embedder.calls == 1
    now[0] += conversational_semantic.FAILURE_COOLDOWN_SECONDS + 1
    embedder.fail = False
    assert semantic_intent("heya folks").intent == "greeting"


def test_project_name_questions_are_not_small_talk(fake, monkeypatch):
    monkeypatch.setattr(settings, "project_name", "Acme Portal")
    for m in ["What is Acme Portal?", "Tell me about Acme Portal"]:
        assert semantic_intent(m) is None
        assert conversational.conversational_reply(m) is None


def test_semantic_runs_on_text_after_greeting(fake):
    assert conversational.conversational_reply("Hello, many thanks")["found"] is True
    assert conversational.conversational_reply("hola, what's the weather?") is None


def test_semantic_match_skips_rag(fake, monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))
    r = client.post("/chat", json={"question": "heya folks"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []


def test_semantic_failure_falls_back_to_rag(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", FakeEmbedder(fail=True))
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    r = client.post("/chat", json={"question": "heya folks"})
    assert r.status_code == 200 and seen == ["heya folks"]


# --- real local model (same convention as test_indexing_retrieval.py) ---

@pytest.fixture
def default_thresholds(monkeypatch):
    """The shipped defaults, whatever .env says, so these tests prove the defaults suit MiniLM."""
    for name in ("semantic_small_talk_threshold", "semantic_small_talk_margin", "embedding_model"):
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)


@pytest.mark.usefixtures("default_thresholds")
@pytest.mark.parametrize("message,intent", [
    ("many thanks", "gratitude"), ("catch you later", "farewell"), ("sounds good", "acknowledgement"),
    ("how is your day going", "how_are_you"),
])
def test_real_model_small_talk(message, intent):
    match = semantic_intent(message)
    assert match is not None and match.intent == intent


@pytest.mark.usefixtures("default_thresholds")
@pytest.mark.parametrize("message", [
    "what is an invoice", "hi can you explain invoice statuses", "how do refunds work", "help with invoices",
    "what is this", "who are the clients", "good question",
    "great wall of china", "morning news headlines", "perfect square definition", "ok corral shootout",
    "what's the weather?",
])
def test_real_model_questions_go_to_rag(message):
    assert semantic_intent(message) is None
