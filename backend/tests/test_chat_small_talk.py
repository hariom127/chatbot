"""Routing and the safety net through POST /chat."""
import logging
import random

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, llm, small_talk_kb
from app.services.chat import NOT_FOUND_ANSWER

client = TestClient(app)

HIT = {"text": "Invoices are monthly.", "title": "Invoices", "url": "http://doc/inv",
       "doc_id": "inv", "chunk_index": 0, "distance": 0.2}
SAMPLE_NAME = "Acme Portal"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(conversational, "_rng", random.Random(1234))
    monkeypatch.setattr(conversational, "_last_used", {})
    monkeypatch.setattr(settings, "project_name", SAMPLE_NAME)
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)
    monkeypatch.setattr(settings, "small_talk_kb_enabled", True)
    monkeypatch.setattr(settings, "small_talk_safety_net_enabled", True)


@pytest.fixture
def no_hits(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))
    return seen


@pytest.fixture
def llm_not_found(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: NOT_FOUND_ANSWER)
    return seen


def ask(question: str) -> dict:
    r = client.post("/chat", json={"question": question})
    assert r.status_code == 200
    return r.json()


def small_talk_answers(entry_id: str) -> set[str]:
    return {conversational.render_reply(t) for t in conversational.replies(entry_id)}


def greeting_not_found() -> set[str]:
    return {conversational.render_reply(t) for t in conversational.SAFETY_NET_NOT_FOUND}


# --- before RAG ---

@pytest.mark.parametrize("question", ["Hi", "thanks, bye!", "heyyy whts goin on", "helo ther", "hows ur day goin"])
def test_small_talk_never_reaches_rag(monkeypatch, question):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))
    body = ask(question)
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []


def test_semantic_match_never_reaches_rag(monkeypatch):
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    monkeypatch.setattr(small_talk_kb, "semantic_scores",
                        lambda key: small_talk_kb.SemanticScores("farewell", 0.90, 0.10))
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    assert ask("catch ya")["answer"] in small_talk_answers("farewell")


def test_greeting_question_reaches_rag_with_original_text(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: "Invoices are monthly.")
    body = ask("Hi, what is an invoice?")
    assert seen == ["Hi, what is an invoice?"]
    assert body["found"] is True and body["sources"] == [{"title": "Invoices", "url": "http://doc/inv"}]


# --- safety net: no chunks retrieved ---

def test_no_hits_loose_fuzzy_match_gets_small_talk(no_hits):
    body = ask("what is going on")
    assert no_hits == ["what is going on"]
    assert body["found"] is True and body["sources"] == []
    assert body["answer"] in small_talk_answers("whats_going_on")


def test_no_hits_greeting_question_gets_friendly_not_found(no_hits):
    body = ask("Hi, what is the late fee?")
    assert body["found"] is False and body["sources"] == [] and body["snippets"] == []
    assert body["answer"] in greeting_not_found()
    assert body["answer"].startswith(("Hi!", "Hello!")) and SAMPLE_NAME in body["answer"]


@pytest.mark.parametrize("question", ["Hello, what is the weather on Mars?", "hey whats the late fee",
                                      "Heyyy, who approves refunds in the northern region office?"])
def test_long_or_typed_greeting_questions_greet_back(no_hits, question):
    assert ask(question)["answer"] in greeting_not_found()


def test_friendly_not_found_rotates(no_hits):
    answers = [ask("Hi, what is the late fee?")["answer"] for _ in range(4)]
    assert all(a != b for a, b in zip(answers, answers[1:]))


@pytest.mark.parametrize("question", ["Neptune?", "x?", "good question about invoices", "high tide times",
                                      "history of the company", "Hiring policy for interns?"])
def test_no_hits_other_questions_get_plain_not_found(no_hits, question):
    body = ask(question)
    assert body == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": [], "snippets": []}


def test_no_hits_loose_semantic_match(no_hits, monkeypatch):
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    monkeypatch.setattr(settings, "small_talk_semantic_threshold", 0.75)
    monkeypatch.setattr(small_talk_kb, "semantic_scores",
                        lambda key: small_talk_kb.SemanticScores("farewell", 0.70, 0.20))
    body = ask("catch ya")
    assert body["found"] is True and body["answer"] in small_talk_answers("farewell")


@pytest.mark.parametrize("fuzzy,semantic,entry_id", [
    (small_talk_kb.FuzzyScores("farewell", 0.75, 0.40), None, "farewell"),
    (small_talk_kb.FuzzyScores("farewell", 0.69, 0.40), None, None),  # below the loose threshold
    (small_talk_kb.FuzzyScores("farewell", 0.80, 0.80), None, None),  # no lead over the anchors
    (None, small_talk_kb.SemanticScores("farewell", 0.66, 0.20), "farewell"),
    (None, small_talk_kb.SemanticScores("farewell", 0.64, 0.20), None),  # below threshold - 0.10
    (None, small_talk_kb.SemanticScores("farewell", 0.70, 0.72), None),  # no lead over the anchors
    (None, small_talk_kb.SemanticScores(None, 0.0, 0.90), None),
    (small_talk_kb.FuzzyScores("gratitude", 0.72, 0.40), small_talk_kb.SemanticScores("farewell", 0.70, 0.2),
     "gratitude"),  # the higher score wins
])
def test_loose_match_rules(monkeypatch, fuzzy, semantic, entry_id):
    monkeypatch.setattr(settings, "small_talk_semantic_threshold", 0.75)
    match = conversational._loose_match(conversational.NearMiss("k", fuzzy, semantic))
    assert (match and match.entry_id) == entry_id


def test_safety_net_never_embeds(no_hits, monkeypatch):
    calls = []
    real = small_talk_kb.semantic_scores
    monkeypatch.setattr(small_talk_kb, "semantic_scores", lambda key: calls.append(key) or real(key))
    ask("what is going on")
    assert len(calls) == 1  # once before RAG, never again in the safety net


# --- safety net: the LLM says not found ---

def test_llm_not_found_loose_match_gets_small_talk(llm_not_found):
    body = ask("what is going on")
    assert llm_not_found == ["what is going on"]
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []
    assert body["answer"] in small_talk_answers("whats_going_on")


def test_llm_not_found_greeting_question(llm_not_found):
    body = ask("Hi, what is the late fee?")
    assert body["found"] is False and body["answer"] in greeting_not_found() and body["sources"] == []


def test_llm_not_found_keeps_llm_text(monkeypatch):
    llm_text = f"{NOT_FOUND_ANSWER} Try the billing team."
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: llm_text)
    body = ask("Neptune?")
    assert body == {"answer": llm_text, "found": False, "sources": [], "snippets": []}


# --- flags and failures ---

@pytest.mark.parametrize("question", ["what is going on", "Hi, what is the late fee?", "Neptune?", "Hi"])
def test_master_flag_off_behaves_as_today(no_hits, monkeypatch, question):
    monkeypatch.setattr(settings, "conversational_replies_enabled", False)
    body = ask(question)
    assert no_hits == [question]
    assert body == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": [], "snippets": []}


@pytest.mark.parametrize("question", ["what is going on", "Hi, what is the late fee?"])
def test_safety_net_flag_off_returns_plain_not_found(no_hits, monkeypatch, question):
    monkeypatch.setattr(settings, "small_talk_safety_net_enabled", False)
    assert ask(question)["answer"] == NOT_FOUND_ANSWER


def test_safety_net_flag_off_llm_path(llm_not_found, monkeypatch):
    monkeypatch.setattr(settings, "small_talk_safety_net_enabled", False)
    body = ask("what is going on")
    assert body == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": [], "snippets": []}


def test_kb_flag_off_still_greets_back(no_hits, monkeypatch):
    monkeypatch.setattr(settings, "small_talk_kb_enabled", False)
    assert ask("what is going on")["answer"] == NOT_FOUND_ANSWER  # no scores to loosen
    assert ask("Hi, what is the late fee?")["answer"] in greeting_not_found()


def test_safety_net_error_returns_plain_not_found(no_hits, monkeypatch, caplog):
    def boom(*args):
        raise RuntimeError("safety net bug")

    monkeypatch.setattr(conversational, "_loose_match", boom)
    with caplog.at_level(logging.ERROR):
        body = ask("what is going on")
    assert body["answer"] == NOT_FOUND_ANSWER and "safety net failed" in caplog.text


def test_classifier_error_returns_rag_answer(monkeypatch):
    def boom(message):
        raise RuntimeError("classifier bug")

    monkeypatch.setattr(conversational, "classify_conversational_intent", boom)
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [])
    assert ask("Hi")["answer"] == NOT_FOUND_ANSWER


def test_blank_input_still_rejected(no_hits):
    assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_safety_net_log_has_no_message_text(no_hits, caplog):
    with caplog.at_level(logging.DEBUG):
        ask("what is going on")
        ask("Hi, what is the late fee?")
    assert "layer=safety-net" in caplog.text and "greeting not-found" in caplog.text
    assert "late fee" not in caplog.text and "what is going on" not in caplog.text
