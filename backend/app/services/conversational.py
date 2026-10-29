"""Small talk (greetings, thanks, goodbyes, ...) answered instantly, without retrieval or the LLM."""
import hashlib
import json
import logging
import random
import re
import threading
import unicodedata
from functools import cache
from pathlib import Path
from typing import NamedTuple

from app.config import settings
from app.services import small_talk_kb

logger = logging.getLogger(__name__)

# Version-controlled small-talk knowledge base: examples, replies and real-question anchors.
KB_PATH = Path(__file__).resolve().parent.parent / "small_talk" / "small_talk_kb.json"

MAX_MESSAGE_LENGTH = 60
MAX_EXAMPLE_WORDS = 8
MIN_EXAMPLES = 8
MIN_REPLIES = 2
MAX_PROJECT_NAME_LENGTH = 60
FALLBACK_PROJECT = "this project"

INTENTS = {
    "greeting", "time_greeting", "gratitude", "farewell", "how_are_you", "whats_going_on", "capabilities",
    "acknowledgement", "identity", "returning", "compliment", "apology", "presence_check", "laughter",
}

# Highest priority first: a message mixing entries gets the reply of the first one found.
PRIORITY = ["capabilities", "farewell", "gratitude", "how_are_you", "whats_going_on", "time_greeting", "greeting",
            "acknowledgement"]

# Rule phrases per knowledge-base entry id. Each pattern must match a whole run of normalized words;
# repeated letters are allowed where shown.
PHRASES = {
    "greeting": [r"h+i+", r"he+l+o+", r"he+y+", "hiya", "howdy", "hola", "greetings", "namaste", "namaskar",
                 r"yo+"],
    "time_greeting": [r"good (?:morning|afternoon|evening|day)", "morning|afternoon|evening"],
    "gratitude": [
        r"thanks+(?: (?:a lot|so much|very much))?", r"thank ?(?:you|u)(?: (?:so much|very much))?",
        "thx", "ty", "much appreciated", "appreciate it",
    ],
    "farewell": [
        r"bye+(?: bye+)?", r"good ?bye+", r"see (?:you|ya|u)(?: (?:later|soon))?", "cya", "farewell",
        "talk to you later", "ttyl", "good night", "take care",
    ],
    "how_are_you": ["how are you(?: doing)?", "how r u", "hows it going"],
    "whats_going_on": ["whats ?up", r"wa+s+u+p+", r"su+p+", "whats going on", "whats new", "whats happening"],
    "capabilities": ["what can you do", "who are you", "what are you", "help", "what do you do"],
    "acknowledgement": [
        "ok", "okay", "k", "cool", "great", "nice", "got it", "alright", "perfect", "understood", "sure",
        "yes", "yeah", "yep", "no", "nope", r"h+m+",
    ],
}
FILLERS = {"there", "all", "everyone", "team", "bot", "chatbot", "assistant", "buddy", "friend", "again", "dude",
           "bro", "mate", "guys", "folks"}
_MAX_PHRASE_WORDS = 4
_PATTERNS = {entry: re.compile("|".join(f"(?:{p})" for p in pats)) for entry, pats in PHRASES.items()}

# Used only when the knowledge-base file is missing or invalid, so rule matches still get a reply.
FALLBACK_REPLIES = {
    "greeting": ["Hello! What would you like to know about {project}?"],
    "time_greeting": ["{salutation}! What would you like to know about {project}?"],
    "gratitude": ["You're welcome! Anything else you'd like to know?"],
    "farewell": ["Goodbye! Come back any time you have a question."],
    "how_are_you": ["I'm doing well, thanks for asking! How can I help?"],
    "whats_going_on": ["All good here! What would you like to know about {project}?"],
    "capabilities": ["I answer questions about {project} using its synced knowledge base."],
    "acknowledgement": ["Okay! Let me know if you have another question."],
}

# Layer 4: a question that starts with a greeting but has no answer gets a not-found that greets back.
SAFETY_NET_NOT_FOUND = [
    "Hi! I don't have enough information to answer that about {project}. Could you try rephrasing it?",
    "Hello! I don't have enough information on that one. Try asking it another way, "
    "or ask me something else about {project}.",
]
SAFETY_NET_FUZZY_THRESHOLD = 0.70
SAFETY_NET_SEMANTIC_SLACK = 0.10  # below the semantic threshold

_PLACEHOLDER = re.compile(r"\{(project|salutation)\}")
_ANY_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_TIME_GREETING = re.compile(r"\bgood (day)\b|\b(morning|afternoon|evening)\b")

# Rotation state is per process: the backend has no conversation id.
_rng = random.Random()
_last_used: dict[str, str] = {}
_lock = threading.Lock()
_logged_once: set[str] = set()


class Entry(NamedTuple):
    id: str
    intent: str
    examples: list[str]
    replies: list[str]
    rules_only: bool


class KnowledgeBase(NamedTuple):
    entries: dict[str, Entry]
    anchors: list[str]
    hash: str = ""  # SHA-256 of the file bytes; a change rebuilds the small-talk Chroma collection


def normalize(message: str) -> str:
    """Matching-only copy of the message; never sent to RAG or logged."""
    text = unicodedata.normalize("NFKC", message).lower()
    text = re.sub(r"['‘’ʼ]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"[\s_]+", " ", text).strip()


# --- knowledge base ---

def reply_problem(template: str, entry_id: str) -> str | None:
    """Why a reply template breaks the grammar rules, or None if it is fine."""
    if any(m.group() not in ("{project}", "{salutation}") for m in _ANY_PLACEHOLDER.finditer(template)):
        return "unknown placeholder"
    if template.count("{project}") > 1:
        return "{project} used more than once"
    for m in re.finditer(r"\{project\}", template):
        # after "about/with/for" it reads right with a real name and with "this project"
        if not template[:m.start()].endswith(("about ", "with ", "for ")):
            return "{project} must follow 'about', 'with' or 'for'"
        if re.match(r"['’\w]", template[m.end():]):
            return "{project} must not be possessive or plural"
    if "{salutation}" in template and (entry_id != "time_greeting" or not template.startswith("{salutation}")
                                       or template.count("{salutation}") > 1):
        return "{salutation} only at the start of time_greeting replies"
    if not template.endswith((".", "!", "?")):
        return "must end with '.', '!' or '?'"
    if "  " in template or template != template.strip():
        return "extra whitespace"
    return None


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _example_problem(example: str) -> str | None:
    if len(example) > MAX_MESSAGE_LENGTH or not 1 <= len(normalize(example).split()) <= MAX_EXAMPLE_WORDS:
        return f"must be 1-{MAX_EXAMPLE_WORDS} words, at most {MAX_MESSAGE_LENGTH} characters"
    if re.search(r"\d", example):
        return "contains a digit"
    return None


def _parse_entry(raw) -> Entry:
    """One validated entry, or ValueError naming it and the broken rule."""
    if not isinstance(raw, dict):
        raise ValueError("entries: every entry must be an object")
    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", entry_id):
        raise ValueError(f"entry {entry_id!r}: id must be a lowercase slug")
    examples, replies_, rules_only = raw.get("examples", []), raw.get("replies"), raw.get("rules_only", False)
    problems = [
        (raw.get("intent") not in INTENTS, f"unknown intent {raw.get('intent')!r}"),
        (not isinstance(rules_only, bool), "rules_only must be true or false"),
        (not _is_str_list(examples), "examples must be a list of strings"),
        (not _is_str_list(replies_), "replies must be a list of strings"),
    ]
    if not any(bad for bad, _ in problems):
        problems += [
            (not rules_only and len(examples) < MIN_EXAMPLES, f"needs at least {MIN_EXAMPLES} examples"),
            (len(replies_) < MIN_REPLIES, f"needs at least {MIN_REPLIES} replies"),
        ]
        problems += [(True, f"example {e!r} {p}") for e in examples if (p := _example_problem(e))]
        problems += [(True, f"reply {r!r}: {p}") for r in replies_ if (p := reply_problem(r, entry_id))]
    for bad, rule in problems:
        if bad:
            raise ValueError(f"entry {entry_id}: {rule}")
    return Entry(entry_id, raw["intent"], examples, replies_, rules_only)


def _validate(data: dict) -> KnowledgeBase:
    """The parsed knowledge base, or ValueError naming the entry and the broken rule."""
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError("file: missing 'entries' list")
    anchors = data.get("question_anchors")
    if not _is_str_list(anchors) or not all(a.strip() for a in anchors):
        raise ValueError("file: 'question_anchors' must be a list of strings")

    entries: dict[str, Entry] = {}
    for raw in data["entries"]:
        entry = _parse_entry(raw)
        if entry.id in entries:
            raise ValueError(f"entry {entry.id}: duplicate id")
        entries[entry.id] = entry
    if missing := sorted(set(PHRASES) - set(entries)):
        raise ValueError(f"file: rule entries missing: {', '.join(missing)}")

    # an example must point to exactly one place: one entry, never also a real question
    seen: dict[str, str] = {normalize(a): "question_anchors" for a in anchors}
    for entry in entries.values():
        for example in entry.examples:
            key = normalize(example)
            if key in seen:
                raise ValueError(f"entry {entry.id}: example {example!r} also appears in {seen[key]}")
            seen[key] = f"entry {entry.id}"
    return KnowledgeBase(entries, anchors)


@cache
def knowledge_base() -> KnowledgeBase | None:
    """The validated knowledge base, loaded once; None (logged as one ERROR) when missing or invalid."""
    try:
        raw = KB_PATH.read_bytes()
        kb = _validate(json.loads(raw.decode("utf-8")))._replace(hash=hashlib.sha256(raw).hexdigest())
    except Exception as exc:
        logger.error("Small-talk knowledge base %s is unusable (%s); only rule replies are active", KB_PATH.name, exc)
        return None
    logger.info("Small-talk knowledge base loaded: %d entries, %d examples, %d anchors", len(kb.entries),
                sum(len(e.examples) for e in kb.entries.values()), len(kb.anchors))
    return kb


def replies(entry_id: str) -> list[str]:
    kb = knowledge_base()
    if kb is not None and entry_id in kb.entries:
        return kb.entries[entry_id].replies
    return FALLBACK_REPLIES[entry_id]


# --- rules ---

def _match_entries(words: list[str]) -> frozenset[str] | None:
    """Entries found if the words are entirely a sequence of known phrases (each with an optional filler)."""
    n = len(words)

    @cache
    def parse(i: int) -> frozenset[str] | None:
        if i == n:
            return frozenset()
        for j in range(min(n, i + _MAX_PHRASE_WORDS), i, -1):
            span = " ".join(words[i:j])
            for entry, pattern in _PATTERNS.items():
                if not pattern.fullmatch(span):
                    continue
                nexts = [j + 1, j] if j < n and words[j] in FILLERS else [j]
                for k in nexts:
                    rest = parse(k)
                    if rest is not None:
                        return rest | {entry}
        return None

    return parse(0)


def classify_conversational_intent(message: str) -> str | None:
    """Rules only: the knowledge-base entry id of the message, or None."""
    if len(message) > MAX_MESSAGE_LENGTH:
        return None
    words = normalize(message).split()
    if not words:
        return None
    entries = _match_entries(words)
    if not entries:
        return None
    return next(e for e in PRIORITY if e in entries)


# --- replies ---

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
    m = _TIME_GREETING.search(normalize(message))
    return f"Good {m.group(1) or m.group(2)}" if m else "Hello"


def render_reply(template: str, salutation_text: str = "Hello") -> str:
    # the name is data substituted into a fixed template, so braces or % in it are shown literally
    values = {"project": project_name(), "salutation": salutation_text}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _pick_variant(entry_id: str, variants: list[str] | None = None) -> str:
    variants = variants or replies(entry_id)
    with _lock:
        choices = [v for v in variants if v != _last_used.get(entry_id)] or variants
        choice = _rng.choice(choices)
        _last_used[entry_id] = choice
    return choice


class _Match(NamedTuple):
    entry_id: str
    layer: str
    matched_text: str  # what the salutation is read from: the message, or its fuzzy key
    detail: str = ""


class NearMiss(NamedTuple):
    """Layer 2 scores kept for the safety net, so it never embeds again."""
    key: str = ""
    fuzzy: small_talk_kb.FuzzyScores | None = None
    semantic: small_talk_kb.SemanticScores | None = None


class SmallTalkCheck(NamedTuple):
    reply: dict | None = None
    near_miss: NearMiss | None = None  # set when the message passed the Layer 2 gates but was not accepted


def _kb_match(message: str) -> tuple[_Match | None, NearMiss | None]:
    """Layer 2: rules on the fuzzy key, fuzzy similarity (2a), then the Chroma collection (2b)."""
    if not small_talk_kb.is_eligible(message):
        return None, None
    key = small_talk_kb.fuzzy_key(normalize(message))
    if entry_id := classify_conversational_intent(key):
        return _Match(entry_id, "fuzzy-rules", key), None
    fuzzy = small_talk_kb.fuzzy_scores(key)
    if fuzzy is not None:
        detail = f"f_social={fuzzy.f_social:.2f} f_question={fuzzy.f_question:.2f}"
        if small_talk_kb.fuzzy_accepts(fuzzy):
            return _Match(fuzzy.entry_id, "fuzzy", key, detail), None
        logger.debug("Fuzzy small-talk near miss: entry=%s %s", fuzzy.entry_id, detail)
    semantic = small_talk_kb.semantic_scores(key)
    if semantic is not None:
        detail = f"s_social={semantic.s_social:.2f} s_question={semantic.s_question:.2f}"
        if small_talk_kb.semantic_accepts(semantic):
            return _Match(semantic.entry_id, "semantic", key, detail), None
        logger.debug("Semantic small-talk near miss: entry=%s %s", semantic.entry_id, detail)
    return None, NearMiss(key, fuzzy, semantic)


def _match(message: str) -> _Match | None:
    return _classify(message)[0]


def _classify(message: str) -> tuple[_Match | None, NearMiss | None]:
    if entry_id := classify_conversational_intent(message):
        return _Match(entry_id, "rules", message), None
    return _kb_match(message)


def _small_talk_response(match: _Match) -> dict:
    answer = render_reply(_pick_variant(match.entry_id), salutation(match.matched_text))
    logger.info("Small-talk reply: entry=%s layer=%s %s", match.entry_id, match.layer, match.detail)
    return {"answer": answer, "found": True, "sources": [], "snippets": []}


def check(message: str) -> SmallTalkCheck | None:
    """Layers 1-2: a small-talk reply, or the near-miss scores for the safety net.

    None when the check itself failed: the request then behaves exactly as RAG alone. Never raises.
    """
    try:
        match, near_miss = _classify(message)
        if match is None:
            return SmallTalkCheck(near_miss=near_miss)
        return SmallTalkCheck(reply=_small_talk_response(match))
    except Exception:
        logger.exception("Small-talk check failed; falling back to RAG")
        return None


def conversational_reply(message: str) -> dict | None:
    """A full chat response for small talk, or None to let RAG answer. Never raises."""
    result = check(message)
    return result.reply if result else None


# --- Layer 4: safety net after RAG found nothing ---

def _loose_match(near_miss: NearMiss) -> _Match | None:
    """Looser thresholds than before RAG: by now the only alternative is a not-found message."""
    candidates = []
    fuzzy, semantic = near_miss.fuzzy, near_miss.semantic
    if fuzzy and fuzzy.f_social >= SAFETY_NET_FUZZY_THRESHOLD and fuzzy.f_social > fuzzy.f_question:
        candidates.append((fuzzy.f_social, fuzzy.entry_id, f"f_social={fuzzy.f_social:.2f} "
                                                            f"f_question={fuzzy.f_question:.2f}"))
    if (semantic and semantic.entry_id
            and semantic.s_social >= settings.small_talk_semantic_threshold - SAFETY_NET_SEMANTIC_SLACK
            and semantic.s_social > semantic.s_question):
        candidates.append((semantic.s_social, semantic.entry_id, f"s_social={semantic.s_social:.2f} "
                                                                 f"s_question={semantic.s_question:.2f}"))
    if not candidates:
        return None
    _, entry_id, detail = max(candidates)
    return _Match(entry_id, "safety-net", near_miss.key, detail)


def _starts_with_greeting(message: str) -> bool:
    words = small_talk_kb.fuzzy_key(normalize(message)).split()
    if not words:
        return False
    kb = knowledge_base()
    single_word = {e for e in kb.entries["greeting"].examples if " " not in e} if kb else set()
    return words[0] in single_word or classify_conversational_intent(words[0]) == "greeting"


def safety_net_reply(message: str, near_miss: NearMiss | None) -> dict | None:
    """A friendlier answer than the plain not-found, or None to keep it. Never raises, never embeds."""
    try:
        if near_miss is not None and (match := _loose_match(near_miss)):
            return _small_talk_response(match)
        if _starts_with_greeting(message):
            answer = render_reply(_pick_variant("safety_net", SAFETY_NET_NOT_FOUND))
            logger.info("Safety net: greeting not-found reply")
            return {"answer": answer, "found": False, "sources": [], "snippets": []}
    except Exception:
        logger.exception("Small-talk safety net failed; using the plain not-found answer")
    return None
