import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import chat as chat_service
from app.services import llm
from app.services.llm import LLMError

client = TestClient(app)

HIT = {"text": "25 days of vacation.", "title": "Vacation Policy", "url": "http://doc/vac",
       "doc_id": "vac", "chunk_index": 0, "distance": 0.2}


def test_successful_answer(monkeypatch):
    seen = {}

    def fake_generate(messages):
        seen["messages"] = messages
        return "You get 25 days."

    monkeypatch.setattr(chat_service, "retrieve", lambda q: [HIT])
    monkeypatch.setattr(llm, "generate", fake_generate)
    r = client.post("/chat", json={"question": "How many days?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["answer"] == "You get 25 days."
    assert body["sources"] == [{"title": "Vacation Policy", "url": "http://doc/vac"}]
    assert body["snippets"][0]["text"].startswith("25 days")
    assert "untrusted" in seen["messages"][0]["content"]
    assert "25 days of vacation." in seen["messages"][1]["content"]


def test_no_relevant_context_skips_llm(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [])
    monkeypatch.setattr(llm, "generate", lambda m: pytest.fail("LLM must not be called"))
    body = client.post("/chat", json={"question": "Neptune?"}).json()
    assert body["found"] is False and body["sources"] == []
    assert "could not find" in body["answer"]


def test_model_says_not_found_hides_sources(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda q: [HIT])
    monkeypatch.setattr(llm, "generate", lambda m: chat_service.NOT_FOUND_ANSWER)
    body = client.post("/chat", json={"question": "x?"}).json()
    assert body["found"] is False and body["sources"] == []


def test_provider_failure(monkeypatch):
    def boom(messages):
        raise LLMError("The AI provider timed out.")

    monkeypatch.setattr(chat_service, "retrieve", lambda q: [HIT])
    monkeypatch.setattr(llm, "generate", boom)
    r = client.post("/chat", json={"question": "x?"})
    assert r.status_code == 502 and r.json()["detail"] == "The AI provider timed out."


def test_unexpected_error_hides_details(monkeypatch):
    def boom(q):
        raise ValueError("secret-internal-detail")

    monkeypatch.setattr(chat_service, "retrieve", boom)
    r = client.post("/chat", json={"question": "x?"})
    assert r.status_code == 500 and "secret-internal-detail" not in r.text


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "a" * 2001}])
def test_validation(payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "modal")
    monkeypatch.setattr(settings, "modal_proxy_token_id", "")
    monkeypatch.setattr(settings, "modal_proxy_token_secret", "")
    with pytest.raises(LLMError, match="Modal credentials"):
        llm.generate([{"role": "user", "content": "hi"}])


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(LLMError, match="Unknown"):
        llm.generate([])


def test_strip_thinking():
    assert llm._strip_thinking("<think>hmm\nok</think>\nFinal.") == "Final."
