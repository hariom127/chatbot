from app.services import indexer
from app.services.retrieval import retrieve
from app.services.vector_store import VectorStore
from tests.conftest import doc

VACATION = doc("vac", "Vacation Policy", "Employees receive 25 days of paid vacation per year. "
               "Unused vacation days can be carried over until March of the next year.")
COOKING = doc("cook", "Pasta Recipe", "Boil the spaghetti in salted water for ten minutes, "
              "then toss with garlic, olive oil and fresh basil.")


def test_index_search_and_idempotent(tmp_settings):
    store = VectorStore()
    r1 = indexer.sync_documents([VACATION, COOKING], store)
    assert r1["added"] == 2
    n = store.count()
    assert n >= 2

    hits = retrieve("How many vacation days do I get?", store)
    assert hits and hits[0]["title"] == "Vacation Policy"
    assert hits[0]["url"].endswith("/vac")

    r2 = indexer.sync_documents([VACATION, COOKING], store)  # nothing changed
    assert r2["unchanged"] == 2 and r2["added"] == 0 and r2["updated"] == 0
    assert store.count() == n


def test_irrelevant_question_returns_nothing(tmp_settings):
    store = VectorStore()
    indexer.sync_documents([VACATION, COOKING], store)
    assert retrieve("What is the orbital period of Neptune?", store) == []


def test_modified_and_deleted_docs(tmp_settings):
    store = VectorStore()
    indexer.sync_documents([VACATION, COOKING], store)

    changed = doc("vac", "Vacation Policy", "Employees receive 30 days of paid vacation per year.",
                  modified="2024-06-01T00:00:00Z")
    r = indexer.sync_documents([changed, COOKING], store)
    assert r["updated"] == 1 and r["unchanged"] == 1
    texts = " ".join(h["text"] for h in retrieve("vacation days", store))
    assert "30 days" in texts and "25 days" not in texts  # no stale content

    r = indexer.sync_documents([changed], store)  # COOKING deleted from Drive
    assert r["removed"] == 1
    assert "cook" not in store.indexed_doc_ids()
    assert indexer.get_status()["indexed_documents"] == 1


def test_empty_doc_and_read_error_are_handled(tmp_settings):
    store = VectorStore()
    indexer.sync_documents([VACATION], store)
    empty = doc("empty", "Blank", "   ")
    broken = {**VACATION, "text": "", "error": "Permission denied"}
    r = indexer.sync_documents([empty, broken], store)
    assert r["skipped"] == 1 and len(r["errors"]) == 1
    assert "vac" in store.indexed_doc_ids()  # a transient read error must not wipe the index


def test_persistence_across_restart(tmp_settings):
    indexer.sync_documents([VACATION], VectorStore())
    assert VectorStore().count() > 0  # fresh client, same directory
    assert retrieve("paid vacation", VectorStore())
