"""Sync Drive docs into ChromaDB: only changed docs are re-embedded, deleted docs are purged."""
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.services.chunking import chunk_document
from app.services.embeddings import embed_texts
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)
_sync_lock = threading.Lock()


def _load_state() -> dict:
    path = Path(settings.sync_state_file)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Sync state file is corrupt; starting fresh")
    return {"docs": {}, "last_sync": None, "last_result": None}


def _save_state(state: dict) -> None:
    path = Path(settings.sync_state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def get_status() -> dict:
    state = _load_state()
    return {
        "last_sync": state["last_sync"],
        "last_result": state["last_result"],
        "indexed_documents": len(state["docs"]),
    }


def _fingerprint(doc: dict) -> str:
    # content hash + title: a rename changes chunk metadata, so it must trigger re-indexing too
    return hashlib.sha256((doc["title"] + "\n" + doc.get("text", "")).encode()).hexdigest()


def sync_documents(docs: list[dict], store: VectorStore | None = None) -> dict:
    """docs: dicts with id, title, url, modified_time, text (optional 'error' = could not be read)."""
    if not _sync_lock.acquire(blocking=False):
        raise RuntimeError("A sync is already running.")
    try:
        store = store or VectorStore()
        state = _load_state()
        known = state["docs"]
        result = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "skipped": 0, "errors": []}

        seen = set()
        for doc in docs:
            seen.add(doc["id"])
            if doc.get("error"):
                # keep any existing index for this doc; don't delete on a transient read failure
                result["errors"].append(f"{doc['title']}: {doc['error']}")
                continue
            fp = _fingerprint(doc)
            prev = known.get(doc["id"])
            if prev and prev["hash"] == fp:
                result["unchanged"] += 1
                continue

            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
            store.delete_document(doc["id"])  # drop stale chunks (e.g. doc got shorter)
            if not chunks:
                result["skipped"] += 1  # empty doc: nothing to index
                known.pop(doc["id"], None)
                continue
            store.add_chunks(chunks, embed_texts([c.text for c in chunks]))
            known[doc["id"]] = {"hash": fp, "modified_time": doc["modified_time"], "chunks": len(chunks)}
            result["updated" if prev else "added"] += 1

        # remove docs deleted from Drive (or moved out of the folder)
        for doc_id in (set(known) | store.indexed_doc_ids()) - seen:
            store.delete_document(doc_id)
            known.pop(doc_id, None)
            result["removed"] += 1

        state["docs"] = known
        state["last_sync"] = datetime.now(timezone.utc).isoformat()
        state["last_result"] = result
        _save_state(state)
        return result
    finally:
        _sync_lock.release()
