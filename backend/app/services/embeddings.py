"""Embeddings (local Sentence Transformers or OpenAI). Kept separate from answer generation."""
import logging
from functools import lru_cache

import openai
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

OPENAI_BATCH_SIZE = 100


class EmbeddingError(Exception):
    """User-presentable embedding failure (never contains keys)."""


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _embed_local(texts: list[str]) -> list[list[float]]:
    return _local_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def _embed_openai(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY is missing (needed for OpenAI embeddings).")
    if not settings.openai_embedding_model:
        raise EmbeddingError("OPENAI_EMBEDDING_MODEL is not set.")
    client = OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), OPENAI_BATCH_SIZE):
            resp = client.embeddings.create(
                model=settings.openai_embedding_model, input=texts[i:i + OPENAI_BATCH_SIZE]
            )
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    except openai.AuthenticationError as exc:
        raise EmbeddingError("OpenAI rejected the API key (embeddings).") from exc
    except openai.RateLimitError as exc:
        raise EmbeddingError("OpenAI rate limit or quota reached (embeddings).") from exc
    except openai.NotFoundError as exc:
        raise EmbeddingError(f"Embedding model '{settings.openai_embedding_model}' was not found.") from exc
    except (openai.APITimeoutError, openai.APIConnectionError) as exc:
        raise EmbeddingError("Could not reach OpenAI for embeddings.") from exc
    except openai.APIError as exc:
        logger.warning("Embedding API error: %s", type(exc).__name__)
        raise EmbeddingError("OpenAI returned an error while creating embeddings.") from exc
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    provider = settings.resolved_embedding_provider
    if provider == "openai":
        return _embed_openai(texts)
    if provider == "local":
        return _embed_local(texts)
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'.")


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
