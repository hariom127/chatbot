"""Small talk (greetings, thanks, goodbyes, ...) answered instantly, without retrieval or the LLM."""
import logging
import random
import re
import threading
import unicodedata
from functools import cache

from app.config import settings
from app.services import conversational_semantic

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 60
MAX_PROJECT_NAME_LENGTH = 60
FALLBACK_PROJECT = "this project"

# Highest priority first: a message mixing intents gets the reply of the first one found.
PRIORITY = ["capabilities", "farewell", "gratitude", "how_are_you", "time_greeting", "greeting", "acknowledgement"]

# Each pattern must match a whole run of normalized words; repeated letters are allowed where shown.
PHRASES = {
    "greeting": [r"h+i+", r"he+l+o+", r"he+y+", "hiya", "howdy", "greetings", "namaste", "namaskar", "yo"],
    "time_greeting": [r"(?:good )?(?:morning|afternoon|evening)"],
    "gratitude": [
        r"thanks+(?: (?:a lot|so much|very much))?", r"thank (?:you|u)(?: (?:so much|very much))?",
        "thx", "ty", "much appreciated", "appreciate it",
    ],
    "farewell": [
        r"bye+(?: bye+)?", r"good ?bye+", r"see (?:you|ya|u)(?: (?:later|soon))?",
        "talk to you later", "ttyl", "good night", "take care",
    ],
    "how_are_you": ["how are you(?: doing)?", "how r u", "hows it going", "whats up", "sup"],
    "capabilities": ["what can you do", "who are you", "what are you", "help", "what do you do"],
    "acknowledgement": ["ok", "okay", "k", "cool", "great", "nice", "got it", "alright", "perfect"],
}
FILLERS = {"there", "all", "everyone", "team", "bot", "chatbot", "assistant", "buddy", "friend", "again"}
_MAX_PHRASE_WORDS = 4
_PATTERNS = {intent: re.compile("|".join(f"(?:{p})" for p in pats)) for intent, pats in PHRASES.items()}

# {project} only ever follows "about " or "with ", at most once per reply (tests enforce this).
REPLIES = {
    "greeting": [
        "Hello! I can answer questions about {project}. What would you like to know?",
        "Hi there! Ask me anything about {project}.",
        "Hey! What would you like to know about {project}?",
        "Hello! How can I help you with {project} today?",
    ],
    "time_greeting": [
        "{salutation}! I can answer questions about {project}. What would you like to know?",
        "{salutation}! What would you like to know about {project}?",
        "{salutation}! How can I help you with {project} today?",
    ],
    "gratitude": [
        "You're welcome! Ask me anything else about {project}.",
        "Happy to help! Anything else you'd like to know?",
        "Glad that helped! Let me know if you have more questions.",
        "Anytime! What else can I help you with?",
    ],
    "farewell": [
        "Goodbye! Come back any time you have a question about {project}.",
        "See you later! I'll be here when you need me.",
        "Take care! Feel free to come back with more questions.",
    ],
    "how_are_you": [
        "I'm doing well, thanks for asking! What would you like to know about {project}?",
        "All good here! How can I help you with {project}?",
        "Doing great! Ask me anything about {project}.",
    ],
    "capabilities": [
        "I answer questions about {project} using its knowledge base synced from Google Drive, "
        "and I link the sources I used.",
        "Ask me a question about {project} and I'll answer from its synced Google Docs, with links to the sources.",
        "I can help with questions about {project}. If the answer isn't in the synced content, "
        "I'll tell you instead of guessing.",
    ],
    "acknowledgement": [
        "Great! Anything else you'd like to know?",
        "Sounds good. Ask me anything else about {project}.",
        "Okay! Let me know if you have another question.",
    ],
}
_PLACEHOLDER = re.compile(r"\{(project|salutation)\}")
_TIME_WORDS = ("morning", "afternoon", "evening")

# Rotation state is per process: the backend has no conversation id.
_rng = random.Random()
_last_used: dict[str, str] = {}
_lock = threading.Lock()
_logged_once: set[str] = set()


def normalize(message: str) -> str:
    """Matching-only copy of the message; never sent to RAG or logged."""
    text = unicodedata.normalize("NFKC", message).lower()
    text = re.sub(r"['‘’ʼ]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"[\s_]+", " ", text).strip()


def _match_intents(words: list[str]) -> frozenset[str] | None:
    """Intents found if the words are entirely a sequence of known phrases (each with an optional filler)."""
    n = len(words)

    @cache
    def parse(i: int) -> frozenset[str] | None:
        if i == n:
            return frozenset()
        for j in range(min(n, i + _MAX_PHRASE_WORDS), i, -1):
            span = " ".join(words[i:j])
            for intent, pattern in _PATTERNS.items():
                if not pattern.fullmatch(span):
                    continue
                nexts = [j + 1, j] if j < n and words[j] in FILLERS else [j]
                for k in nexts:
                    rest = parse(k)
                    if rest is not None:
                        return rest | {intent}
        return None

    return parse(0)


def classify_conversational_intent(message: str) -> str | None:
    """Rules only: the small-talk intent of the message, or None."""
    if len(message) > MAX_MESSAGE_LENGTH:
        return None
    words = normalize(message).split()
    if not words:
        return None
    intents = _match_intents(words)
    if not intents:
        return None
    return next(i for i in PRIORITY if i in intents)


def _log_once(key: str, level: int, msg: str, *args) -> None:
    if key not in _logged_once:
        _logged_once.add(key)
        logger.log(level, msg, *args)


def project_name() -> str:
    name = " ".join(settings.project_name.split())
    if len(name) > MAX_PROJECT_NAME_LENGTH:
        _log_once("too_long", logging.WARNING, "PROJECT_NAME is longer than %d characters; using \"%s\" instead.",
                  MAX_PROJECT_NAME_LENGTH, FALLBACK_PROJECT)
        return FALLBACK_PROJECT
    if not name:
        _log_once("unset", logging.INFO, "PROJECT_NAME is not set; small-talk replies say \"%s\".", FALLBACK_PROJECT)
        return FALLBACK_PROJECT
    return name


def salutation(message: str) -> str:
    """Echo the user's own time greeting; the server clock may be in another time zone."""
    words = normalize(message).split()
    for word in _TIME_WORDS:
        if word in words:
            return f"Good {word}"
    return "Hello"


def render_reply(template: str, salutation_text: str = "Hello") -> str:
    # the name is data substituted into a fixed template, so braces or % in it are shown literally
    values = {"project": project_name(), "salutation": salutation_text}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _pick_variant(intent: str) -> str:
    variants = REPLIES[intent]
    with _lock:
        choices = [v for v in variants if v != _last_used.get(intent)] or variants
        choice = _rng.choice(choices)
        _last_used[intent] = choice
    return choice


def conversational_reply(message: str) -> dict | None:
    """A full chat response for small talk, or None to let RAG answer. Never raises."""
    try:
        intent = classify_conversational_intent(message)
        match = None
        if intent is None:
            match = conversational_semantic.semantic_intent(message)
            if match is None:
                return None
            intent = match.intent
        answer = render_reply(_pick_variant(intent), salutation(message))
    except Exception:
        logger.exception("Small-talk check failed; falling back to RAG")
        return None
    if match is None:
        logger.info("Small-talk reply: intent=%s layer=rule", intent)
    else:
        logger.info("Small-talk reply: intent=%s layer=semantic score=%.2f margin=%.2f",
                    intent, match.score, match.margin)
    return {"answer": answer, "found": True, "sources": [], "snippets": []}
