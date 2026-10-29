import json
import logging
import random
import re

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import chat as chat_service
from app.services import conversational, llm
from app.services.conversational import classify_conversational_intent, knowledge_base, normalize, render_reply

client = TestClient(app)

HIT = {"text": "Invoices are monthly.", "title": "Invoices", "url": "http://doc/inv",
       "doc_id": "inv", "chunk_index": 0, "distance": 0.2}
SAMPLE_NAME = "Acme Portal"
KB = knowledge_base()
SHIPPED_PATH = conversational.KB_PATH
ALL_TEMPLATES = [(e.id, t) for e in KB.entries.values() for t in e.replies] + \
    [("safety_net", t) for t in conversational.SAFETY_NET_NOT_FOUND]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(conversational, "_rng", random.Random(1234))
    monkeypatch.setattr(conversational, "_last_used", {})
    monkeypatch.setattr(conversational, "_logged_once", set())
    monkeypatch.setattr(settings, "project_name", "")
    monkeypatch.setattr(settings, "conversational_replies_enabled", True)


@pytest.fixture
def kb_file(tmp_path, monkeypatch):
    """Point the loader at a temp copy of the shipped file; returns a function that writes new content."""
    path = tmp_path / "kb.json"
    monkeypatch.setattr(conversational, "KB_PATH", path)
    knowledge_base.cache_clear()

    def write(data):
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        knowledge_base.cache_clear()
        return path

    yield write
    knowledge_base.cache_clear()


def shipped():
    return json.loads(SHIPPED_PATH.read_text(encoding="utf-8"))


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
    ("greeting", ["hi", "Hi", "hii", "Hiiii", "hhiii", "hello", "hellooo", "HELLO!!", "hey", "heyy", "heyyy",
                  "hiya", "howdy", "Howdy!", "hola", "HOLA", "greetings", "namaste", "namaskar", "yo", "yooo",
                  "hello team", "Hi there!!!", "hey 👋", "hi hi", "  hii  ", "hi guys", "hey mate", "hello folks",
                  "yo dude", "hey bro"]),
    ("time_greeting", ["good morning", "Good   Morning", "good afternoon", "good evening", "Good day",
                       "good day!", "morning", "afternoon", "evening", "good morning everyone"]),
    ("gratitude", ["thanks", "thanksss", "thank you", "thank-you", "Thank-You!", "thankyou", "thank u",
                   "thanks a lot", "thanks so much", "thanks very much", "thank you so much",
                   "thank you very much", "thx", "ty", "much appreciated", "appreciate it", "Thanks bot!"]),
    ("farewell", ["bye", "byeee", "bye bye", "goodbye", "good bye", "see you", "see ya", "See ya!", "see u",
                  "see you later", "see ya soon", "talk to you later", "ttyl", "cya", "farewell", "Farewell!",
                  "good night", "Good night!", "take care"]),
    ("how_are_you", ["how are you", "how r u", "how are you doing", "how's it going", "How are you?",
                     "how are you buddy"]),
    ("whats_going_on", ["what's up", "whatsup", "sup", "Sup?", "wassup", "wasssup", "whats going on",
                        "what's new", "what's happening", "hey whats up", "sup dude", "whats up bro"]),
    ("capabilities", ["what can you do", "who are you", "what are you", "help", "what do you do", "Help!"]),
    ("acknowledgement", ["ok", "okay", "k", "cool", "great", "nice", "got it", "alright", "perfect", "OK!",
                         "understood", "sure", "yes", "yeah", "yep", "no", "nope", "hmm", "Hmmm..."]),
])
def test_phrase_families(intent, messages):
    for m in messages:
        assert classify_conversational_intent(m) == intent, m


@pytest.mark.parametrize("message,intent", [
    ("Hi, what can you do?", "capabilities"),
    ("Thanks, bye!", "farewell"),
    ("ok thanks", "gratitude"),
    ("hi how are you", "how_are_you"),
    ("hey whats up", "whats_going_on"),
    ("Hi, good afternoon", "time_greeting"),
])
def test_priority(message, intent):
    assert classify_conversational_intent(message) == intent


@pytest.mark.parametrize("message", [
    "Hi, what is an invoice?", "Hello, explain invoice statuses", "Thanks, can you explain draft invoices?",
    "What does hello mean in this document?", "Explain invoice statuses", "him", "this",
    "history of the company", "thirty", "good", "morning shift policy", "bye-laws of the company",
    "help me with invoices", "", "   ", "hi " * 30,
    "great wall of china", "morning news headlines", "perfect square definition", "ok corral shootout",
    "good day trading", "no deposit policy", "yes bank invoices", "hola, what's the weather?",
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
    allowed = {render_reply(t) for t in conversational.replies("greeting")}
    assert set(replies) <= allowed
    assert all(a != b for a, b in zip(replies, replies[1:]))


def test_single_variant_intent(monkeypatch):
    monkeypatch.setattr(conversational, "replies", lambda entry_id: ["Hello!"])
    assert [conversational.conversational_reply("hi")["answer"] for _ in range(3)] == ["Hello!"] * 3


@pytest.mark.parametrize("message,prefix", [("good evening", "Good evening!"), ("Good Morning!!", "Good morning!"),
                                            ("afternoon", "Good afternoon!"), ("Good day", "Good day!")])
def test_time_greeting_echoes_user(message, prefix):
    assert conversational.conversational_reply(message)["answer"].startswith(prefix)


def test_catalog_sanity():
    assert set(conversational.PRIORITY) == set(conversational.PHRASES) == set(conversational.FALLBACK_REPLIES)
    assert set(conversational.PHRASES) <= set(KB.entries)
    assert all("{salutation}" in t for t in conversational.replies("time_greeting"))
    for entry_id, variants in conversational.FALLBACK_REPLIES.items():
        assert all(conversational.reply_problem(t, entry_id) is None for t in variants), entry_id


def test_rotation_is_per_entry():
    first = conversational.conversational_reply("hi")["answer"]
    conversational.conversational_reply("thanks")
    assert conversational.conversational_reply("hi")["answer"] != first


# --- knowledge-base file ---

def test_shipped_file_is_valid():
    assert KB is not None
    for entry in KB.entries.values():
        if not entry.rules_only:
            assert 8 <= len(entry.examples) <= 15, entry.id
        assert 2 <= len(entry.replies) <= 4, entry.id
    assert 25 <= len(KB.anchors) <= 40
    assert [e.id for e in KB.entries.values() if e.rules_only] == ["time_greeting"]


def test_no_project_name_in_file():
    text = SHIPPED_PATH.read_text(encoding="utf-8").lower()
    assert SAMPLE_NAME.lower() not in text


def test_rule_phrases_are_examples():
    # fuzzy matching needs each rule entry's own phrases to catch their misspellings
    for entry_id in conversational.PHRASES:
        examples = KB.entries[entry_id].examples
        assert all(classify_conversational_intent(e) == entry_id for e in examples
                   if classify_conversational_intent(e)), entry_id


def _break(mutate):
    data = shipped()
    mutate(data)
    return data


def _entry(data, entry_id):
    return next(e for e in data["entries"] if e["id"] == entry_id)


BROKEN = {
    "duplicate id": lambda d: d["entries"].append(dict(_entry(d, "greeting"), examples=[])),
    "unknown intent": lambda d: _entry(d, "laughter").update(intent="jokes"),
    "digit in example": lambda d: _entry(d, "gratitude")["examples"].append("thanks x2"),
    "bad reply grammar": lambda d: _entry(d, "greeting")["replies"].append("Welcome to the {project}!"),
    "example is an anchor": lambda d: _entry(d, "greeting")["examples"].append("what is an invoice"),
    "example in two entries": lambda d: _entry(d, "laughter")["examples"].append("thanks"),
    "too few examples": lambda d: _entry(d, "apology").update(examples=["sorry"]),
    "missing rule entry": lambda d: d["entries"].remove(_entry(d, "farewell")),
    "salutation outside time_greeting": lambda d: _entry(d, "greeting")["replies"].append("{salutation}! Hi."),
}


@pytest.mark.parametrize("case", [*BROKEN, "invalid json", "missing file"])
def test_broken_file_disables_kb_but_rules_still_reply(kb_file, monkeypatch, caplog, case):
    if case == "invalid json":
        kb_file("{not json")
    elif case == "missing file":
        monkeypatch.setattr(conversational, "KB_PATH", kb_file("{}").with_name("nope.json"))
    else:
        kb_file(_break(BROKEN[case]))
    with caplog.at_level(logging.INFO, logger="app.services.conversational"):
        assert knowledge_base() is None
        assert knowledge_base() is None
        reply = conversational.conversational_reply("Hi")
    assert sum(r.levelno == logging.ERROR for r in caplog.records) == 1
    assert reply["found"] is True and reply["answer"] == render_reply(conversational.FALLBACK_REPLIES["greeting"][0])


def test_valid_copy_loads_and_logs_counts(kb_file, caplog):
    kb_file(shipped())
    with caplog.at_level(logging.INFO, logger="app.services.conversational"):
        kb = knowledge_base()
    assert kb.entries == KB.entries and kb.anchors == KB.anchors and kb.hash
    assert any("14 entries" in r.message for r in caplog.records)


def test_reply_problem_rules():
    ok = ["Ask me anything about {project}.", "How can I help you with {project}?", "I'm the assistant for {project}."]
    assert all(conversational.reply_problem(t, "greeting") is None for t in ok)
    for bad in ["{project} is great.", "Welcome to {project}!", "About the {project}.", "Help with your {project}.",
                "Questions about {project}'s docs.", "Questions about {project}s.", "About {project} and {project}.",
                "Ask about {project}", "Hi  there!", "Hi {name}!", "Hello {salutation}!"]:
        assert conversational.reply_problem(bad, "greeting"), bad
    assert conversational.reply_problem("{salutation}! Hi.", "time_greeting") is None
    assert conversational.reply_problem("Hi {salutation}!", "time_greeting")


# --- project name in replies ---

def test_project_placeholder_position():
    for entry_id, t in ALL_TEMPLATES:
        assert conversational.reply_problem(t, entry_id) is None, t
        assert t.count("{project}") <= 1, t
        for m in re.finditer(r"\{project\}", t):
            assert t[:m.start()].endswith(("about ", "with ", "for ")), t
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

@pytest.mark.parametrize("question", ["Hi", "Good morning", "Thank you", "Bye", "ok", "what can you do",
                                      "wassup", "hola", "Good day", "Good night", "cya", "thank-you", "hmm",
                                      "Hey, what's up?", "whats going on"])
def test_small_talk_skips_rag(no_rag, question):
    r = client.post("/chat", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["sources"] == [] and body["snippets"] == []
    assert body["answer"]


def test_question_with_greeting_reaches_rag_unchanged(monkeypatch):
    seen, prompts = [], []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: prompts.append(m) or "Invoices are monthly.")
    body = client.post("/chat", json={"question": "Hi, what is an invoice?"}).json()
    assert seen == ["Hi, what is an invoice?"]
    assert prompts[0][1]["content"].endswith("Question: Hi, what is an invoice?")
    assert body["found"] is True and body["sources"] == [{"title": "Invoices", "url": "http://doc/inv"}]


@pytest.mark.parametrize("question", ["great wall of china", "morning news headlines", "perfect square definition",
                                      "ok corral shootout"])
def test_content_queries_reach_rag_unchanged(monkeypatch, question):
    seen = []
    monkeypatch.setattr(chat_service, "retrieve", lambda q: seen.append(q) or [])
    body = client.post("/chat", json={"question": question}).json()
    assert seen == [question] and body["found"] is False


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
    without_project = {t for t in conversational.replies("greeting") if "{project}" not in t}
    for _ in range(4):
        answer = client.post("/chat", json={"question": "Hi"}).json()["answer"]
        assert SAMPLE_NAME in answer or answer in without_project
