import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def small_talk_semantic_off(monkeypatch):
    """The semantic small-talk layer embeds text and builds a Chroma collection; tests that need it turn it on."""
    monkeypatch.setattr(settings, "small_talk_semantic_enabled", False)


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Point Chroma and sync state at a temp dir so tests never touch real data."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "sync_state_file", str(tmp_path / "state.json"))
    # tests always use the free local model, whatever .env selects
    monkeypatch.setattr(settings, "embedding_provider", "local")
    return settings


def doc(doc_id, title, text, modified="2024-01-01T00:00:00Z"):
    return {"id": doc_id, "title": title, "url": f"https://docs.google.com/document/d/{doc_id}",
            "modified_time": modified, "text": text}
