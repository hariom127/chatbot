import pytest

from app.config import settings
from app.services import embeddings, indexer, llm
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError
from app.services.vector_store import VectorStore, collection_name
from tests.conftest import doc


class FakeCompletions:
    def __init__(self):
        self.params = None

    def create(self, **params):
        self.params = params
        msg = type("M", (), {"content": "ok"})
        return type("C", (), {"choices": [type("Ch", (), {"message": msg})]})


def fake_client(completions):
    return type("Client", (), {"chat": type("Chat", (), {"completions": completions})})


def test_openai_uses_openai_api_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    assert llm._client().api_key == "sk-test"


def test_openai_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        llm.generate([])


@pytest.mark.parametrize("model, has_temperature", [("gpt-4.1-nano", True), ("gpt-5-mini", False), ("o4-mini", False)])
def test_openai_request_params(monkeypatch, model, has_temperature):
    completions = FakeCompletions()
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", model)
    monkeypatch.setattr(llm, "_client", lambda: fake_client(completions))
    assert llm.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert "max_completion_tokens" in completions.params and "max_tokens" not in completions.params
    assert ("temperature" in completions.params) == has_temperature


def test_embedding_provider_resolution(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small")
    assert settings.embedding_id == "openai:text-embedding-3-small"
    monkeypatch.setattr(settings, "openai_embedding_model", "")
    assert settings.embedding_id.startswith("local:")


def test_openai_embeddings_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        embeddings.embed_texts(["hello"])


def test_changing_embedding_model_reindexes(tmp_settings, monkeypatch):
    d = doc("a", "Alpha", "Alpha project budget is ten thousand dollars.")
    indexer.sync_documents([d], VectorStore())
    old_collection = collection_name()

    # switch to a (fake) different embedding space
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "openai_embedding_model", "fake-model")
    monkeypatch.setattr(indexer, "embed_texts", lambda texts: [[0.1] * 8 for _ in texts])
    assert collection_name() != old_collection
    assert indexer.get_status()["indexed_documents"] == 0

    r = indexer.sync_documents([d], VectorStore())
    assert r["added"] == 1 and r["unchanged"] == 0
    assert indexer.get_status()["embedding"] == "openai:fake-model"
