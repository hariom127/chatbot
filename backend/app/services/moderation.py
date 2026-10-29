"""Abusive messages answered politely, without retrieval or the LLM; abuse mixed into a real question is removed."""
import logging
import random
import re
import threading
from typing import NamedTuple

from app.services import conversational

logger = logging.getLogger(__name__)

# Spellings like "f*ck", "fck", "f**king", "sh!t" are covered; matches must be whole words.
_FUCK = r"(?:mother)?f+(?:[u*#@]+c*|c+)k+(?:ing|in|er|ers|ed|s|face|head|wit)?"
_PROFANITY = [
    _FUCK, r"s+h+[i1!*]+t+(?:s|ty|head|heads|hole)?", r"bullshit", r"bitch(?:es)?", r"bastards?",
    r"(?:ass|arse)holes?", r"dumbass(?:es)?", r"jackass(?:es)?", r"cunts?", r"dickheads?", r"wankers?", r"twats?",
    r"stfu", r"gtfo", r"wtf",
]
_PHRASES = [
    rf"{_FUCK} (?:you|u|off|this(?: bot)?)", rf"the {_FUCK}", rf"shut (?:the (?:{_FUCK}|hell) )?up", r"go to hell",
    r"go (?:and )?die", r"drop dead", r"kill yourself", r"kys", r"screw (?:you|u|this|off)", r"piss off",
]
# Insults count only when aimed at the assistant ("you idiot", "this bot is useless"), so
# "stupid mistakes in invoices" or "idiot-proof setup" are left alone.
_NOUNS = r"(?:idiots?|morons?|imbeciles?|losers?|clowns?|fools?|dumbass|jackass|retards?)"
_ADJS = (r"(?:stupid|dumb|useless|worthless|pathetic|idiotic|moronic|brainless|retarded|trash|garbage|rubbish"
         r"|crap|shitty|crappy)")
_BOT = r"(?:bot|chatbot|assistant|machine|thing|app)"
_INTENSIFIERS = (rf"(?:(?:so|such|really|very|totally|total|completely|complete|absolutely|absolute|utter|truly|the most|a|an|the|one|"
                 rf"{_FUCK}|freaking|damn|bloody|big|little)\s+)*")
_YOU_ARE = rf"(?:(?:you|u)\s*(?:are|r|'re|’re)|youre|ur|your(?=\s+(?:so|such|really|very)\s)|this {_BOT} is)"
_TARGETED = [
    rf"{_YOU_ARE}\s+{_INTENSIFIERS}(?:{_ADJS}|{_NOUNS})(?:\s+{_BOT})?",
    rf"(?:you|u|this {_BOT})\s+(?:really\s+|so\s+)?sucks?",
    rf"(?:(?:you|u|ya)\s+)?{_INTENSIFIERS}{_ADJS}\s+{_BOT}s?",
    rf"(?:you|u|ya)\s+{_INTENSIFIERS}{_NOUNS}",
    rf"(?:(?:you|u)(?: are)?\s+)?(?:a\s+)?piece of (?:shit|crap)",
]
_ABUSE = re.compile(r"(?<![\w*@#])(?:" + "|".join(_PHRASES + _TARGETED + _PROFANITY) + r")(?![\w*@#])",
                    re.IGNORECASE)
# Bare insults ("idiot", "hi stupid") count only when nothing else is left of the message.
_BARE_INSULT = re.compile(r"\b(?:idiots?|morons?|imbeciles?|stupid|dumb|losers?|useless|worthless|pathetic"
                          r"|retard(?:ed|s)?)\b", re.IGNORECASE)

# A match inside quotes, or in a clause that talks about the words themselves, is a mention, not abuse.
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|`[^`]*`|(?<!\w)'[^']*'(?!\w)")
_MENTION_CUES = re.compile(
    r"\b(?:mean|means|meaning|definition|define|etymology|spell(?:ed|ing)?|pronounce|translat\w*|synonyms?"
    r"|swear(?:ing)?|curse|cuss|profan\w*|slurs?|offensive|abusive|insults?|insulting|vulgar|rude"
    r"|(?:the|a|this|that|bad) (?:word|term|phrase)|says?|said|wrote|writes|quoted?|quotes|told)\b"
    r"|\bhow (?:do|should|can|to) (?:i |we )?(?:respond|reply|react|handle|deal)\b",
    re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"([,.!?;:\n]+)")
# Words that carry no question once the abuse is gone ("you, what?").
FILLERS = {
    "you", "u", "ya", "ur", "your", "youre", "are", "r", "re", "is", "so", "such", "a", "an", "the", "this",
    "that", "really", "very", "just", "and", "or", "but", "bot", "chatbot", "assistant", "ai", "thing", "man",
    "dude", "bro", "lol", "lmao", "omg", "seriously", "ugh", "oh", "wow", "again", "now", "totally", "what",
    "why", "huh", "even", "for", "nothing",
}

REPLIES = [
    "I'm here to help with questions about {project}. Let's keep our conversation respectful. "
    "What would you like to know?",
    "I understand things can be frustrating. I'm happy to help with {project} if you have a question.",
    "Let's keep things friendly. Is there something about {project} I can help you with?",
]

# Rotation state is per process, like small-talk replies.
_rng = random.Random()
_last_used: str | None = None
_lock = threading.Lock()


class ModerationResult(NamedTuple):
    abusive_only: bool
    question: str  # the message with abusive language removed (unchanged when there was none)


def _in_quotes(text: str, pos: int) -> bool:
    return any(m.start() < pos < m.end() for m in _QUOTED.finditer(text))


def _clause_at(text: str, pos: int) -> str:
    start = max(text.rfind(c, 0, pos) for c in ",.!?;:\n") + 1
    ends = [i for i in (text.find(c, pos) for c in ",.!?;:\n") if i != -1]
    return text[start:min(ends, default=len(text))]


def _is_mention(text: str, m: re.Match) -> bool:
    return _in_quotes(text, m.start()) or bool(_MENTION_CUES.search(_clause_at(text, m.start())))


def _has_content(text: str) -> bool:
    return any(w not in FILLERS for w in conversational.normalize(text).split())


def _tidy(text: str) -> str:
    """Drop clauses left empty by the removal, keeping the rest as written ("What is X, ?" -> "What is X?")."""
    parts = _CLAUSE_SPLIT.split(text)
    kept = [chunk + delim for chunk, delim in zip(parts[::2], parts[1::2] + [""]) if _has_content(chunk)]
    out = re.sub(r"\s+", " ", "".join(kept)).strip().rstrip(",;: ")
    end = re.search(r"[.!?]+(?=\s*$)", text)
    if out and end and not re.search(r"[.!?]$", out):
        out += end.group()
    return out


def _is_empty_or_small_talk(text: str) -> bool:
    return not _has_content(text) or conversational.classify_conversational_intent(text) is not None


def check_message(message: str) -> ModerationResult:
    """Whether the message is only abuse, and the message with abuse removed. Rules only; never raises."""
    try:
        spans = [m for m in _ABUSE.finditer(message) if not _is_mention(message, m)]
        cleaned = message
        for m in reversed(spans):
            cleaned = cleaned[:m.start()] + " " + cleaned[m.end():]
        if spans:
            cleaned = _tidy(cleaned)
            if _is_empty_or_small_talk(cleaned):
                return ModerationResult(True, "")
        bare = [m for m in _BARE_INSULT.finditer(cleaned) if not _is_mention(cleaned, m)]
        if bare and _is_empty_or_small_talk(_BARE_INSULT.sub(" ", cleaned)):
            return ModerationResult(True, "")
        if spans:
            logger.info("Abusive language removed; answering the rest of the question")
        return ModerationResult(False, cleaned)
    except Exception:
        logger.exception("Abusive-language check failed; using the full message")
        return ModerationResult(False, message)


def _pick_variant() -> str:
    global _last_used
    with _lock:
        choices = [v for v in REPLIES if v != _last_used] or REPLIES
        _last_used = _rng.choice(choices)
        return _last_used


def moderation_reply() -> dict:
    logger.info("Moderation reply: abusive message")
    return {"answer": conversational.render_reply(_pick_variant()), "found": True, "sources": [], "snippets": []}
