"""Persistent ChromaDB wrapper. Embeddings are passed in, so Chroma never loads its own model."""
import hashlib

import chromadb

from app.config import settings
from app.services.chunking import Chunk


def collection_name() -> str:
    # one collection per embedding model: vector sizes differ, so they must never be mixed
    return "drive_docs_" + hashlib.sha256(settings.embedding_id.encode()).hexdigest()[:12]


def _collection(path: str | None = None):
    client = chromadb.PersistentClient(
        path=path or settings.chroma_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        collection_name(), metadata={"hnsw:space": "cosine", "embedding_id": settings.embedding_id}
    )


class VectorStore:
    def __init__(self, path: str | None = None):
        self.col = _collection(path)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def indexed_doc_ids(self) -> set[str]:
        res = self.col.get(include=["metadatas"])
        return {m["doc_id"] for m in res["metadatas"]}

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.col.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return self.col.count()
