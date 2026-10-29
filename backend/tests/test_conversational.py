import logging
import random
import re

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, llm
from app.services.conversational import REPLIES, classify_conversational_intent, normalize, render_reply

client = TestClient(app)

HIT = {"text": "Invoices are monthly.", "title": "Invoices", "url": "http://doc/inv",
       "doc_id": "inv", "chunk_index": 0, "distance": 0.2}
SAMPLE_NAME = "Acme Portal"
ALL_TEMPLATES = [(intent, t) for intent, variants in REPLIES.items() for t in variants]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(conversational, "_rng", random.Random(1234))
    monkeypatch.setattr(conversational, "_last_used", {})
    monkeypatch.setattr(conversational, "_logged_once", set())
    monkeypatch.setattr(settings, "project_name", "")
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)


@pytest.fixture
def no_rag(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))


# --- normalization ---

@pytest.mark.parametrize("raw,expected", [
    ("  HELLO!! ", "hello"),
    ("Hi there!!!", "hi there"),
    ("Good   Morning", "good morning"),
    ("hey 👋", "hey"),
    ("thanks,bye", "thanks bye"),
    ("what's up", "whats up"),
    ("what’s up", "whats up"),
    ("hi__there", "hi there"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


# --- rules ---

@pytest.mark.parametrize("intent,messages", [
    ("greeting", ["hi", "Hiiii", "hhiii", "hello", "hellooo", "HELLO!!", "hey", "heyyy", "hiya", "howdy",
                  "greetings", "namaste", "namaskar", "yo", "hello team", "Hi there!!!", "hey 👋", "hi hi"]),
    ("time_greeting", ["good morning", "Good   Morning", "good afternoon", "good evening", "morning",
                       "afternoon", "evening", "good morning everyone"]),
    ("gratitude", ["thanks", "thanksss", "thank you", "thank u", "thanks a lot", "thanks so much",
                   "thanks very much", "thank you so much", "thank you very much", "thx", "ty",
                   "much appreciated", "appreciate it", "Thanks bot!"]),
    ("farewell", ["bye", "byeee", "bye bye", "goodbye", "good bye", "see you", "see ya", "see u",
                  "see you later", "see ya soon", "talk to you later", "ttyl", "good night", "take care"]),
    ("how_are_you", ["how are you", "how r u", "how are you doing", "how's it going", "what's up", "sup",
                     "How are you?"]),
    ("capabilities", ["what can you do", "who are you", "what are you", "help", "what do you do", "Help!"]),
    ("acknowledgement", ["ok", "okay", "k", "cool", "great", "nice", "got it", "alright", "perfect", "OK!"]),
])
def test_phrase_families(intent, messages):
    for m in messages:
        assert classify_conversational_intent(m) == intent, m


@pytest.mark.parametrize("message,intent", [
    ("Hi, what can you do?", "capabilities"),
    ("Thanks, bye!", "farewell"),
    ("ok thanks", "gratitude"),
    ("hi how are you", "how_are_you"),
    ("Hi, good afternoon", "time_greeting"),
])
def test_priority(message, intent):
    assert classify_conversational_intent(message) == intent


@pytest.mark.parametrize("message", [
    "Hi, what is an invoice?", "Hello, explain invoice statuses", "Thanks, can you explain draft invoices?",
    "What does hello mean in this document?", "Explain invoice statuses", "him", "this",
    "history of the company", "thirty", "good", "morning shift policy", "bye-laws of the company",
    "help me with invoices", "", "   ", "hi " * 30,
])
def test_real_questions_are_not_small_talk(message):
    assert classify_conversational_intent(message) is None


def test_project_name_is_not_small_talk(monkeypatch):
    monkeypatch.setattr(settings, "project_name", SAMPLE_NAME)
    for m in [f"What is {SAMPLE_NAME}?", f"Tell me about {SAMPLE_NAME}"]:
        assert classify_conversational_intent(m) is None
        assert conversational.conversational_reply(m) is None


# --- rotation and replies ---

def test_rotation_never_repeats():
    replies = [conversational.conversational_reply("hi")["answer"] for _ in range(20)]
    allowed = {render_reply(t) for t in REPLIES["greeting"]}
    assert set(replies) <= allowed
    assert all(a != b for a, b in zip(replies, replies[1:]))


def test_single_variant_intent(monkeypatch):
    monkeypatch.setitem(REPLIES, "greeting", ["Hello!"])
    assert [conversational.conversational_reply("hi")["answer"] for _ in range(3)] == ["Hello!"] * 3


@pytest.mark.parametrize("message,prefix", [("good evening", "Good evening!"), ("Good Morning!!", "Good morning!"),
                                            ("afternoon", "Good afternoon!")])
def test_time_greeting_echoes_user(message, prefix):
    assert conversational.conversational_reply(message)["answer"].startswith(prefix)


def test_catalog_sanity():
    assert set(REPLIES) == set(conversational.PRIORITY) == set(conversational.PHRASES)
    assert all(REPLIES[intent] for intent in REPLIES)
    assert all("{salutation}" in t for t in REPLIES["time_greeting"])


# --- project name in replies ---

def test_project_placeholder_position():
    for _, t in ALL_TEMPLATES:
        assert t.count("{project}") <= 1, t
        for m in re.finditer(r"\{project\}", t):
            assert t[:m.start()].endswith(("about ", "with ")), t
        assert not re.search(r"(^|[.!?] )\{project\}", t), t


@pytest.mark.parametrize("name,expected", [
    (SAMPLE_NAME, SAMPLE_NAME), ("  Acme   Portal ", SAMPLE_NAME),
    ("", "this project"), ("   ", "this project"), ("x" * 61, "this project"),
])
def test_every_variant_renders_cleanly(monkeypatch, name, expected):
    monkeypatch.setattr(settings, "project_name", name)
    for _, t in ALL_TEMPLATES:
        r = render_reply(t, "Good evening")
        if "{project}" in t:
            assert r.count(expected) == 1, r
        assert "{" not in r and "}" not in r and "salutation" not in r and "project}" not in r, r
        assert not re.search(r"\b(\w+) \1\b", r, re.IGNORECASE), r
        assert "  " not in r and not re.search(r" [.,!?;:]", r), r
        assert r.endswith((".", "!", "?")), r


@pytest.mark.parametrize("name", ["Beta {v2}", "50% Club", "R&D Portal", "{salutation}"])
def test_special_characters_are_literal(monkeypatch, name):
    monkeypatch.setattr(settings, "project_name", name)
    assert name in render_reply("Ask me anything about {project}.")


def test_unset_name_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="app.services.conversational"):
        for _ in range(3):
            conversational.conversational_reply("what can you do")
    assert sum("PROJECT_NAME is not set" in r.message for r in caplog.records) == 1


def test_too_long_name_warned_once(monkeypatch, caplog):
    monkeypatch.setattr(settings, "project_name", "x" * 61)
    with caplog.at_level(logging.INFO, logger="app.services.conversational"):
        for _ in range(3):
            conversational.conversational_reply("what can you do")
    assert sum(r.levelno == logging.WARNING and "PROJECT_NAME" in r.message for r in caplog.records) == 1


def test_name_read_at_reply_time(monkeypatch):
    monkeypatch.setattr(settings, "project_name", "Alpha Hub")
    first = conversational.conversational_reply("what can you do")["answer"]
    monkeypatch.setattr(settings, "project_name", "Beta Desk")
    second = conversational.conversational_reply("what can you do")["answer"]
    assert "Alpha Hub" in first and "Beta Desk" in second


def test_message_text_not_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        conversational.conversational_reply("hellooo team")
    assert "hellooo" not in caplog.text


# --- routing through POST /chat ---

@pytest.mark.parametrize("question", ["Hi", "Good morning", "Thank you", "Bye", "ok", "what can you do"])
def test_small_talk_skips_rag(no_rag, question):
    r = client.post("/chat", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []
    assert body["answer"]


def test_greeting_with_question_goes_to_rag(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: "Invoices are monthly.")
    body = client.post("/chat", json={"question": "Hi, what is an invoice?"}).json()
    assert seen == ["Hi, what is an invoice?"]
    assert body["sources"] == [{"title": "Invoices", "url": "http://doc/inv"}]


def test_master_flag_off_sends_hi_to_rag(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "conversational_replies_enabled", False)
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    body = client.post("/chat", json={"question": "Hi"}).json()
    assert seen == ["Hi"] and body["found"] is False


def test_classifier_error_falls_back_to_rag(monkeypatch):
    def boom(message):
        raise RuntimeError("classifier bug")

    monkeypatch.setattr(conversational, "classify_conversational_intent", boom)
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: "Invoices are monthly.")
    r = client.post("/chat", json={"question": "Hi"})
    assert r.status_code == 200 and r.json()["sources"]


def test_blank_input_still_rejected(no_rag):
    assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_hi_uses_configured_name(monkeypatch, no_rag):
    monkeypatch.setattr(settings, "project_name", SAMPLE_NAME)
    without_project = {t for t in REPLIES["greeting"] if "{project}" not in t}
    for _ in range(4):
        answer = client.post("/chat", json={"question": "Hi"}).json()["answer"]
        assert SAMPLE_NAME in answer or answer in without_project
