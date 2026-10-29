import logging
import random
import re

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, embeddings, llm, moderation
from app.services.moderation import REPLIES, check_message

client = TestClient(app)

HIT = {"text": "Invoices are monthly.", "title": "Invoices", "url": "http://doc/inv",
       "doc_id": "inv", "chunk_index": 0, "distance": 0.2}
SAMPLE_NAME = "Acme Portal"

ABUSIVE_ONLY = [
    "fuck you", "FUCK YOU!!!", "f*ck off", "F**K YOU", "fck this bot", "shut up", "shut the fuck up", "stfu",
    "go to hell", "piss off", "screw you", "you are so stupid", "You're useless", "youre an idiot",
    "u r dumb", "ur a moron", "your so stupid", "you absolute moron", "you idiot", "stupid bot",
    "this bot sucks", "you suck", "useless chatbot!", "you piece of shit", "idiot", "Idiot!!", "hi idiot",
    "hello stupid", "thanks, you idiot", "bye, you useless bot", "bullshit", "you are a stupid bot",
    "this bot is garbage", "sh!t bot",
]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(moderation, "_rng", random.Random(1234))
    monkeypatch.setattr(moderation, "_last_used", None)
    monkeypatch.setattr(conversational, "_logged_once", set())
    monkeypatch.setattr(settings, "project_name", "")
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)
    monkeypatch.setattr(settings, "abusive_language_filter_enabled", True)


@pytest.fixture
def no_rag(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: pytest.fail("retrieve must not be called"))
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))
    monkeypatch.setattr(embeddings, "embed_query", lambda t: pytest.fail("embeddings must not be called"))
    monkeypatch.setattr(embeddings, "embed_texts", lambda t: pytest.fail("embeddings must not be called"))


# --- detection ---

@pytest.mark.parametrize("message", ABUSIVE_ONLY)
def test_abusive_only(message):
    assert check_message(message) == (True, "")


@pytest.mark.parametrize("message,question", [
    ("You stupid bot, what is an invoice?", "what is an invoice?"),
    ("What is an invoice, you idiot?", "What is an invoice?"),
    ("what the fuck is an invoice", "what is an invoice"),
    ("fuck you, how do refunds work", "how do refunds work"),
    ("shit, I forgot my password. How do I reset it?", "I forgot my password. How do I reset it?"),
    ("you are useless. explain draft invoices", "explain draft invoices"),
    ("Hi, you idiot, what is a line item?", "Hi, what is a line item?"),
    ("This bot sucks! Who approves orders?", "Who approves orders?"),
])
def test_abuse_removed_from_question(message, question):
    assert check_message(message) == (False, question)


@pytest.mark.parametrize("message", [
    # ordinary words that contain or resemble abusive ones
    "How do I fix stupid mistakes in invoices?", "idiot-proof setup guide", "Scunthorpe office invoices",
    "class assessment dates", "cocktail party policy", "Dickens and shiitake", "Can you dumb it down?",
    "What is a useless line item?", "How do I empty the trash folder?", "garbage collection settings",
    "is it hell to approve invoices", "the assistant manager role", "Can you explain the fool-proof process?",
    "stupid question, but what is an invoice?", "what is the past due fee", "hello", "Hi, what is an invoice?",
    # educational discussion and quoted text
    "What does 'idiot' mean?", "what does fuck mean", "Is stupid a bad word?", "Is 'shut up' considered rude?",
    'The customer wrote "you are useless" in the review', "The vendor said “you idiot” on the call",
    "how do I handle a customer who says you are useless?", "Is calling someone an idiot an insult?",
    "What is the policy on profanity like fuck in tickets?",
])
def test_not_abusive(message):
    assert check_message(message) == (False, message)


def test_quoted_and_unquoted_abuse_in_one_message():
    assert check_message("What does 'idiot' mean, you idiot?") == (False, "What does 'idiot' mean?")


def test_check_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(moderation, "_tidy", boom)
    assert check_message("you idiot, what is X?") == (False, "you idiot, what is X?")


# --- replies ---

@pytest.mark.parametrize("name,expected", [
    (SAMPLE_NAME, SAMPLE_NAME), ("  Acme   Portal ", SAMPLE_NAME),
    ("", "this project"), ("   ", "this project"), ("x" * 61, "this project"),
])
def test_reply_names_project(monkeypatch, name, expected):
    monkeypatch.setattr(settings, "project_name", name)
    for t in REPLIES:
        r = conversational.render_reply(t)
        assert r.count(expected) == 1, r
        assert "{" not in r and "}" not in r and "document" not in r.lower(), r
        assert "  " not in r and not re.search(r" [.,!?;:]", r) and r.endswith((".", "!", "?")), r


def test_reply_templates_place_project_like_small_talk():
    for t in REPLIES:
        assert t.count("{project}") == 1
        assert t[:t.index("{project}")].endswith(("about ", "with ")), t


def test_rotation_never_repeats():
    replies = [moderation.moderation_reply()["answer"] for _ in range(20)]
    assert set(replies) <= {conversational.render_reply(t) for t in REPLIES}
    assert all(a != b for a, b in zip(replies, replies[1:]))


def test_reply_shape():
    reply = moderation.moderation_reply()
    assert set(reply) == {"answer", "found", "sources", "snippets"}
    assert reply["found"] is True and reply["sources"] == [] and reply["snippets"] == []


def test_message_text_not_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        check_message("you moron, what is X?")
        check_message("you moron")
    assert "moron" not in caplog.text


# --- routing through POST /chat ---

@pytest.mark.parametrize("question", ["fuck you", "you are useless", "stupid bot", "hi idiot", "shut up"])
def test_abusive_message_skips_rag(no_rag, monkeypatch, question):
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", True)
    monkeypatch.setattr(settings, "project_name", SAMPLE_NAME)
    r = client.post("/chat", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert SAMPLE_NAME in body["answer"]
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []


def test_abusive_reply_without_project_name(no_rag):
    body = client.post("/chat", json={"question": "you idiot"}).json()
    assert "this project" in body["answer"]


def test_mixed_message_answers_cleaned_question(monkeypatch):
    seen, prompts = [], []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: prompts.append(m) or "Invoices are monthly.")
    body = client.post("/chat", json={"question": "You stupid bot, what is an invoice?"}).json()
    assert seen == ["what is an invoice?"]
    assert prompts[0][1]["content"].endswith("Question: what is an invoice?")
    assert body == {"answer": "Invoices are monthly.", "found": True,
                    "sources": [{"title": "Invoices", "url": "http://doc/inv"}],
                    "snippets": [{"title": "Invoices", "url": "http://doc/inv", "text": "Invoices are monthly."}]}


def test_mixed_message_keeps_greeting(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    client.post("/chat", json={"question": "Hi, you idiot, what is a line item?"})
    assert seen == ["Hi, what is a line item?"]


@pytest.mark.parametrize("question", ["What does 'idiot' mean?", "How do I fix stupid mistakes in invoices?",
                                      "garbage collection settings"])
def test_false_positives_reach_rag_unchanged(monkeypatch, question):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    body = client.post("/chat", json={"question": question}).json()
    assert seen == [question] and body["found"] is False


def test_filter_off_sends_abuse_to_rag(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "abusive_language_filter_enabled", False)
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    client.post("/chat", json={"question": "you idiot"})
    assert seen == ["you idiot"]


def test_filter_works_with_small_talk_off(no_rag, monkeypatch):
    monkeypatch.setattr(settings, "conversational_replies_enabled", False)
    body = client.post("/chat", json={"question": "fuck you"}).json()
    assert body["answer"] in {conversational.render_reply(t) for t in REPLIES}


# --- regressions: existing behaviour is unchanged ---

@pytest.mark.parametrize("question,intent", [("Hi", "greeting"), ("Good morning", "time_greeting"),
                                             ("Thank you", "gratitude"), ("bye", "farewell")])
def test_small_talk_still_answered(no_rag, question, intent):
    body = client.post("/chat", json={"question": question}).json()
    renders = {conversational.render_reply(t, conversational.salutation(question))
               for t in conversational.replies(intent)}
    assert body["answer"] in renders


def test_normal_question_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: "Invoices are monthly.")
    body = client.post("/chat", json={"question": "Hi, what is an invoice?"}).json()
    assert seen == ["Hi, what is an invoice?"] and body["found"] is True
