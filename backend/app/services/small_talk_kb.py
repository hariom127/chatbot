"""Small-talk knowledge-base matching: fuzzy keys for typos and shorthand (2a), then a Chroma collection (2b)."""
import difflib
import hashlib
import logging
import re
import threading
import time
from typing import NamedTuple

import chromadb

from app.config import settings
from app.services import conversational, embeddings

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 60
MIN_TYPO_WORD_LENGTH = 4
TYPO_CUTOFF = 0.8
SEMANTIC_TOP_K = 10
FAILURE_COOLDOWN_SECONDS = 300

# Whole-word chat shorthand, applied after stretched letters are squeezed.
SHORTHAND = {
    "u": "you", "r": "are", "ur": "your", "ya": "you", "plz": "please", "pls": "please", "thx": "thanks",
    "gud": "good", "gn": "good night", "wassup": "whats up", "wazzup": "whats up", "hw": "how",
}
_STRETCHED = re.compile(r"([^\W\d_])\1{2,}")


class FuzzyScores(NamedTuple):
    entry_id: str  # entry of the closest small-talk example
    f_social: float  # best ratio against any small-talk example
    f_question: float  # best ratio against any real-question anchor


class SemanticScores(NamedTuple):
    entry_id: str | None  # entry of the nearest small-talk example among the top results
    s_social: float  # its cosine similarity (0 when no example is among the top results)
    s_question: float  # best similarity to a real-question anchor (0 when none is among the top results)


class _Index(NamedTuple):
    social_words: frozenset[str]  # typo corrections may only land on these
    protected_words: frozenset[str]  # anchor words: never corrected, so questions keep their meaning
    by_first_letter: dict[str, list[str]]
    examples: list[tuple[str, str]]  # (entry id, fuzzy key), excluding rules_only entries
    anchors: list[str]  # fuzzy keys


_lock = threading.Lock()
_index_cache: tuple[object, _Index] | None = None
_collection_lock = threading.Lock()
_ready: tuple[tuple, object] | None = None  # (embedding id, chroma dir, kb hash) -> verified collection
_failed_at: float | None = None
_clock = time.monotonic


def _basic_key(normalized: str) -> str:
    """Squeeze stretched letters ("hiii" -> "hi"; doubles like "hello" stay) and expand shorthand."""
    words = _STRETCHED.sub(r"\1", normalized).split()
    return " ".join(SHORTHAND.get(w, w) for w in words)


def _build_index(kb) -> _Index:
    social = {w for e in kb.entries.values() for x in e.examples for w in _basic_key(conversational.normalize(x)).split()}
    anchor_keys = [_basic_key(conversational.normalize(a)) for a in kb.anchors]
    protected = {w for a in anchor_keys for w in a.split()}
    by_first: dict[str, list[str]] = {}
    for w in sorted(social):
        by_first.setdefault(w[0], []).append(w)
    index = _Index(frozenset(social), frozenset(protected), by_first, [], anchor_keys)
    index.examples.extend((e.id, fuzzy_key(conversational.normalize(x), index))
                          for e in kb.entries.values() if not e.rules_only for x in e.examples)
    return index


def _index() -> _Index | None:
    global _index_cache
    kb = conversational.knowledge_base()
    if kb is None:
        return None
    with _lock:
        if _index_cache is None or _index_cache[0] is not kb:
            _index_cache = (kb, _build_index(kb))
        return _index_cache[1]


def _correct(word: str, index: _Index) -> str:
    if (len(word) < MIN_TYPO_WORD_LENGTH or re.search(r"\d", word) or word in index.social_words
            or word in index.protected_words):
        return word
    # same first letter: "whts" -> "whats", but never "dude" -> "done"-like jumps to another word family
    close = difflib.get_close_matches(word, index.by_first_letter.get(word[0], []), n=1, cutoff=TYPO_CUTOFF)
    return close[0] if close else word


def fuzzy_key(normalized: str, index: _Index | None = None) -> str:
    """Matching-only form of a normalized message: squeezed, shorthand expanded, typos fixed toward small talk."""
    index = index or _index()
    words = _basic_key(normalized).split()
    if index is None:
        return " ".join(words)
    return " ".join(_correct(w, index) for w in words)


def is_eligible(message: str) -> bool:
    """Gates for the knowledge-base layers: short, few words, no digits (order numbers and amounts go to RAG)."""
    if not settings.small_talk_kb_enabled or len(message) > MAX_MESSAGE_LENGTH or re.search(r"\d", message):
        return False
    if not 1 <= len(conversational.normalize(message).split()) <= settings.small_talk_max_words:
        return False
    return conversational.knowledge_base() is not None


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def fuzzy_scores(key: str) -> FuzzyScores | None:
    """Closest small-talk example and closest anchor by character similarity. None without a knowledge base."""
    index = _index()
    if index is None or not index.examples:
        return None
    entry_id, f_social = max(((e, _ratio(key, x)) for e, x in index.examples), key=lambda p: p[1])
    f_question = max((_ratio(key, a) for a in index.anchors), default=0.0)
    return FuzzyScores(entry_id, f_social, f_question)


def fuzzy_accepts(scores: FuzzyScores) -> bool:
    return (scores.f_social >= settings.small_talk_fuzzy_threshold
            and scores.f_social - scores.f_question >= settings.small_talk_fuzzy_margin)


# --- step 2b: semantic matching in the small_talk_* Chroma collection ---

def collection_name() -> str:
    # same scheme as drive_docs_*: one collection per embedding model, never mixed with the documents
    return "small_talk_" + hashlib.sha256(settings.embedding_id.encode()).hexdigest()[:12]


def _client():
    return chromadb.PersistentClient(path=settings.chroma_dir, settings=chromadb.Settings(anonymized_telemetry=False))


def _records(kb, index: _Index) -> tuple[list[str], list[str], list[dict]]:
    """One record per example and per anchor; the stored text is its fuzzy key. Replies stay in the file."""
    ids, docs, metas = [], [], []
    for entry in kb.entries.values():
        if entry.rules_only:
            continue
        for n, example in enumerate(entry.examples):
            ids.append(f"{entry.id}:{n}")
            docs.append(fuzzy_key(conversational.normalize(example), index))
            metas.append({"kind": "example", "entry_id": entry.id, "intent": entry.intent})
    for n, anchor_key in enumerate(index.anchors):
        ids.append(f"anchor:{n}")
        docs.append(anchor_key)
        metas.append({"kind": "anchor", "entry_id": "", "intent": ""})
    return ids, docs, metas


def _collection():
    """The small-talk collection, rebuilt when missing or out of date with the file. Raises on failure."""
    global _ready
    kb, index = conversational.knowledge_base(), _index()
    if kb is None or index is None:
        return None
    state = (settings.embedding_id, settings.chroma_dir, kb.hash)
    if _ready is not None and _ready[0] == state:
        return _ready[1]
    with _collection_lock:
        if _ready is not None and _ready[0] == state:
            return _ready[1]
        ids, docs, metas = _records(kb, index)
        client, name = _client(), collection_name()
        try:
            col = client.get_collection(name)
        except Exception:  # missing
            col = None
        if col is None or (col.metadata or {}).get("kb_hash") != kb.hash or col.count() != len(ids):
            vectors = embeddings.embed_texts(docs)  # embed first: a failure leaves the old collection alone
            if col is not None:
                client.delete_collection(name)
            col = client.create_collection(
                name, metadata={"hnsw:space": "cosine", "embedding_id": settings.embedding_id, "kb_hash": kb.hash})
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
            logger.info("Small-talk collection %s rebuilt: %d records", name, len(ids))
        _ready = (state, col)
        return col


def _cooling_down() -> bool:
    return _failed_at is not None and _clock() - _failed_at < FAILURE_COOLDOWN_SECONDS


def _start_cooldown() -> None:
    global _failed_at
    _failed_at = _clock()
    logger.warning("Small-talk semantic matching failed; paused for %d minutes", FAILURE_COOLDOWN_SECONDS // 60,
                   exc_info=True)


def semantic_scores(key: str) -> SemanticScores | None:
    """Nearest small-talk example and anchor for the fuzzy key. None when off, cooling down or failing."""
    if not settings.small_talk_semantic_enabled or _cooling_down():
        return None
    try:
        col = _collection()
        if col is None:
            return None
        res = col.query(query_embeddings=[embeddings.embed_query(key)], n_results=min(SEMANTIC_TOP_K, col.count()),
                        include=["metadatas", "distances"])
    except Exception:
        _start_cooldown()
        return None
    entry_id, s_social, s_question = None, 0.0, 0.0
    for meta, distance in zip(res["metadatas"][0], res["distances"][0]):
        similarity = 1.0 - distance
        if meta["kind"] == "anchor":
            s_question = max(s_question, similarity)
        elif similarity > s_social:
            entry_id, s_social = meta["entry_id"], similarity
    return SemanticScores(entry_id, s_social, s_question)


def semantic_accepts(scores: SemanticScores) -> bool:
    return (scores.entry_id is not None and scores.s_social >= settings.small_talk_semantic_threshold
            and scores.s_social - scores.s_question >= settings.small_talk_semantic_margin)
