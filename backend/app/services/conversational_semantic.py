"""Semantic small-talk fallback: catches natural variants the rules miss, using the RAG embedding model."""
import logging
import math
import re
import threading
import time
from typing import NamedTuple

from app.config import settings
from app.services import conversational, embeddings

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 60
FAILURE_COOLDOWN_SECONDS = 300

# Natural variations, not copies of the rule phrases. time_greeting has none: its reply echoes the user's words.
POSITIVE_EXAMPLES = {
    "greeting": [
        "hey there friend", "hello hello", "heya", "hi hi", "good to see you", "hey folks", "hello chatbot",
        "hi again", "greetings friend", "hey hey",
    ],
    "gratitude": [
        "many thanks", "thanks a ton", "that helps thanks", "cheers", "thanks so much for the help",
        "really appreciate it", "great help thank you", "thank you kindly",
    ],
    "farewell": [
        "catch you later", "I am done for now", "gotta go", "see you tomorrow", "that is all for today",
        "have a good day", "until next time", "signing off",
    ],
    "how_are_you": [
        "how is your day going", "how have you been", "how are things", "hope you are doing well",
        "how is it going today", "you doing ok",
    ],
    "capabilities": [
        "what are you able to help with", "what kind of questions can I ask", "how can you help me",
        "what do you know", "what is this bot for", "what can I ask you",
    ],
    "acknowledgement": [
        "sounds good", "makes sense", "understood", "fine", "all good", "noted", "that works", "fair enough",
    ],
}

# Real questions. A message must look clearly more like small talk than like any of these.
NEGATIVE_ANCHORS = [
    "what is an invoice", "how does invoice generation work", "can one invoice have multiple orders",
    "what are the invoice statuses", "how is the total calculated",
    "how do I create an order", "where can I find the policy", "explain the refund process",
    "what does this document say about payments",
    "help me understand draft invoices", "what happens after approval", "who approves the request",
    "when is payment due",
    "how many vacation days do I get", "what is the process for onboarding",
    # topics of the indexed invoice documents
    "what is a line item", "what is ready to invoice", "what is a past due invoice",
    "how do invoice reminders work", "what are additional charges", "what is minimum billable",
    "can a vendor edit an invoice after sending it", "what does group by mean", "what are RT OT and DT",
    "what happens when a service is cancelled",
]


class Scores(NamedTuple):
    intent: str
    s_best: float  # highest similarity to that intent's examples
    s_doc: float  # highest similarity to any real-question anchor


class SemanticMatch(NamedTuple):
    intent: str
    score: float
    margin: float


class _Reference(NamedTuple):
    embedding_id: str
    positives: list[tuple[str, list[float]]]
    anchors: list[list[float]]


_lock = threading.Lock()
_reference: _Reference | None = None
_failed_at: float | None = None
_clock = time.monotonic


def _unit(vector: list[float]) -> list[float]:
    # local vectors are normalized already; OpenAI ones are not assumed to be
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def is_eligible(message: str) -> bool:
    if not settings.semantic_small_talk_enabled or len(message) > MAX_MESSAGE_LENGTH:
        return False
    if re.search(r"\d", message):  # order numbers, invoice ids and amounts belong to RAG
        return False
    return 1 <= len(conversational.normalize(message).split()) <= settings.semantic_small_talk_max_words


def _cooling_down() -> bool:
    return _failed_at is not None and _clock() - _failed_at < FAILURE_COOLDOWN_SECONDS


def _start_cooldown() -> None:
    global _failed_at
    _failed_at = _clock()
    logger.warning("Small-talk embedding failed; semantic matching is paused for %d minutes",
                   FAILURE_COOLDOWN_SECONDS // 60, exc_info=True)


def _reference_vectors() -> _Reference | None:
    """Embed the examples and anchors once per embedding model (lazily, in one batch)."""
    global _reference
    embedding_id = settings.embedding_id
    if _reference is not None and _reference.embedding_id == embedding_id:
        return _reference
    with _lock:
        if _reference is not None and _reference.embedding_id == embedding_id:
            return _reference
        if _cooling_down():
            return None
        labelled = [(intent, text) for intent, texts in POSITIVE_EXAMPLES.items() for text in texts]
        texts = [conversational.normalize(t) for _, t in labelled] + \
                [conversational.normalize(t) for t in NEGATIVE_ANCHORS]
        try:
            vectors = [_unit(v) for v in embeddings.embed_texts(texts)]
        except Exception:
            _start_cooldown()
            return None
        n = len(labelled)
        _reference = _Reference(embedding_id, [(intent, v) for (intent, _), v in zip(labelled, vectors[:n])],
                                vectors[n:])
        return _reference


def score(message: str) -> Scores | None:
    """Best small-talk intent and its scores, ignoring threshold and margin. None if ineligible or failing."""
    if not is_eligible(message) or _cooling_down():
        return None
    reference = _reference_vectors()
    if reference is None:
        return None
    try:
        vector = _unit(embeddings.embed_query(conversational.normalize(message)))
    except Exception:
        _start_cooldown()
        return None
    best: dict[str, float] = {}
    for intent, example in reference.positives:
        best[intent] = max(best.get(intent, -1.0), _cosine(vector, example))
    intent = max(best, key=best.get)
    s_doc = max((_cosine(vector, a) for a in reference.anchors), default=-1.0)
    return Scores(intent, best[intent], s_doc)


def accepts(scores: Scores) -> bool:
    return (scores.s_best >= settings.semantic_small_talk_threshold
            and scores.s_best - scores.s_doc >= settings.semantic_small_talk_margin)


def semantic_intent(message: str) -> SemanticMatch | None:
    """The small-talk intent when the message clearly looks like small talk, else None. Never raises."""
    try:
        scores = score(message)
        if scores is None:
            return None
        if not accepts(scores):
            logger.debug("Semantic small-talk near miss: intent=%s s_best=%.2f s_doc=%.2f",
                         scores.intent, scores.s_best, scores.s_doc)
            return None
        return SemanticMatch(scores.intent, scores.s_best, scores.s_best - scores.s_doc)
    except Exception:
        logger.exception("Semantic small-talk check failed")
        return None
