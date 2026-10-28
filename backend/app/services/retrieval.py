"""Retrieval only: question in, relevant chunks out. Knows nothing about the LLM."""
from app.config import settings
from app.services.embeddings import embed_query
from app.services.vector_store import VectorStore


def retrieve(question: str, store: VectorStore | None = None, top_k: int | None = None,
             max_distance: float | None = None) -> list[dict]:
    store = store or VectorStore()
    top_k = top_k or settings.top_k
    max_distance = settings.max_distance if max_distance is None else max_distance
    hits = store.query(embed_query(question), top_k)
    return [
        {
            "text": h["text"],
            "title": h["metadata"]["title"],
            "url": h["metadata"]["url"],
            "doc_id": h["metadata"]["doc_id"],
            "chunk_index": h["metadata"]["chunk_index"],
            "distance": h["distance"],
        }
        for h in hits
        if h["distance"] <= max_distance
    ]
