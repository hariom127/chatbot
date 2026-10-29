"""Calibrate the small-talk fuzzy and semantic thresholds for the configured embedding model.

Run: cd backend && python -m app.scripts.calibrate_small_talk
Uses the app's settings, knowledge-base file and embedding provider, but builds the small-talk collection in a
temporary directory. Pick the lowest values with zero false positives; misses fall through to the safety net.
"""
import argparse
import sys
import tempfile
from typing import NamedTuple

from app.config import settings
from app.services import conversational, small_talk_kb

# None of these may be copied from the knowledge base's examples or anchors (checked at start).
SHOULD_BE_SMALL_TALK = [
    "hey hows your day", "many many thanks", "catch ya", "whats this bot about", "hiya everyone",
    "heyyy whts goin on", "hw r u doing", "im back again", "hahaha", "appreciate the help", "thanks a bunch",
    "hello friends", "nice to meet you", "have a nice day", "thank you for your help", "awesome thanks",
    "bye for now", "what can this bot do", "what questions can you answer", "alright then", "cheers mate",
    "im off now", "u there?", "r u a robot", "my apologies", "thats hilarious", "great answer thanks",
    "yo whats good", "hey there buddy", "thnx a ton", "gud nite", "hows ur week going",
]
MUST_GO_TO_RAG = [
    "what does invoice mean", "hi can you explain invoice statuses", "thanks, what about draft invoices?",
    "whats going on with my refund", "how do refunds work", "hey whats the late fee", "who approves orders",
    "what is a draft invoice", "how are due dates set", "can I edit a sent invoice", "what is overtime",
    "what is per diem", "hello what does past due mean", "explain minimum billable", "what is the rush fee",
    "show me outstanding invoices", "how do I add mileage", "who are the clients", "great wall of china",
    "morning news headlines", "perfect square definition", "ok corral shootout", "hola, what's the weather?",
    "whats new in billing", "any news on my payment", "is my invoice approved", "how are you billing overtime",
    "thanks for the invoice, when is it due", "who are you sending invoices to", "what are you able to export",
]
TRICKY = [
    "help with invoices", "ok so what is the total", "good question", "morning shift timings", "what is going on",
    "what is this", "tell me more", "why", "is it free", "test invoice", "sorry what is a line item",
]
SMALL_TALK, RAG, TRICKY_GROUP = "should be small talk", "must go to RAG", "tricky"
GROUPS = [(SMALL_TALK, SHOULD_BE_SMALL_TALK), (RAG, MUST_GO_TO_RAG), (TRICKY_GROUP, TRICKY)]
PRE_RAG = {"rules", "2a", "2b"}
FUZZY_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
FUZZY_MARGINS = [0.05, 0.10, 0.15, 0.20, 0.25]
SEMANTIC_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SEMANTIC_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]


class Row(NamedTuple):
    probe: str
    key: str
    rule: str | None  # entry from the rules, on the message or on its fuzzy key
    rule_layer: str  # "rules" or "2a"
    fuzzy: small_talk_kb.FuzzyScores | None
    semantic: small_talk_kb.SemanticScores | None
    greets: bool


class Values(NamedTuple):
    fuzzy_threshold: float
    fuzzy_margin: float
    semantic_threshold: float
    semantic_margin: float


def measure(probe: str) -> Row:
    """Every score for the probe, whether or not an earlier layer would already have decided."""
    greets = conversational._starts_with_greeting(probe)
    if rule := conversational.classify_conversational_intent(probe):
        return Row(probe, "", rule, "rules", None, None, greets)
    if not small_talk_kb.is_eligible(probe):
        return Row(probe, "", None, "", None, None, greets)
    key = small_talk_kb.fuzzy_key(conversational.normalize(probe))
    if rule := conversational.classify_conversational_intent(key):
        return Row(probe, key, rule, "2a", None, None, greets)
    return Row(probe, key, None, "", small_talk_kb.fuzzy_scores(key), small_talk_kb.semantic_scores(key), greets)


def decide(row: Row, v: Values) -> tuple[str, str]:
    """(decision, entry): the same order as the app, with the given values."""
    if row.rule:
        return row.rule_layer, row.rule
    f, s = row.fuzzy, row.semantic
    if f and f.f_social >= v.fuzzy_threshold and f.f_social - f.f_question >= v.fuzzy_margin:
        return "2a", f.entry_id
    if s and s.entry_id and s.s_social >= v.semantic_threshold and s.s_social - s.s_question >= v.semantic_margin:
        return "2b", s.entry_id
    # after RAG found nothing: the safety net's loose match, then the greeting not-found
    loose = []
    if f and f.f_social >= conversational.SAFETY_NET_FUZZY_THRESHOLD and f.f_social > f.f_question:
        loose.append((f.f_social, f.entry_id))
    if (s and s.entry_id and s.s_social >= v.semantic_threshold - conversational.SAFETY_NET_SEMANTIC_SLACK
            and s.s_social > s.s_question):
        loose.append((s.s_social, s.entry_id))
    if loose:
        return "safety-net", max(loose)[1]
    return ("RAG, greets back" if row.greets else "RAG"), "-"


def counts(results: dict[str, list[Row]], v: Values) -> tuple[int, int, int]:
    false_pos = sum(decide(r, v)[0] in PRE_RAG for r in results[RAG])
    misses = sum(decide(r, v)[0] not in PRE_RAG for r in results[SMALL_TALK])
    tricky = sum(decide(r, v)[0] in PRE_RAG for r in results[TRICKY_GROUP])
    return false_pos, misses, tricky


def _fmt(value: float | None) -> str:
    return f"{value:6.2f}" if value is not None else f"{'':>6}"


def print_table(results: dict[str, list[Row]], v: Values) -> None:
    for group, rows in results.items():
        print(f"== {group} ==")
        print(f"{'message':40} {'fuzzy key':32} {'f_soc':>6} {'f_q':>6} {'s_soc':>6} {'s_q':>6}  "
              f"{'entry':16} decision")
        for r in rows:
            decision, entry = decide(r, v)
            f, s = r.fuzzy, r.semantic
            print(f"{r.probe:40} {r.key:32} {_fmt(f and f.f_social)} {_fmt(f and f.f_question)} "
                  f"{_fmt(s and s.s_social)} {_fmt(s and s.s_question)}  {entry:16} {decision}")
        print()


def print_problems(results: dict[str, list[Row]], v: Values) -> None:
    false_pos = [(r.probe, *decide(r, v)) for r in results[RAG] if decide(r, v)[0] in PRE_RAG]
    late = [(r.probe, *decide(r, v)) for r in results[RAG] if decide(r, v)[0] == "safety-net"]
    misses = [(r.probe, decide(r, v)[0]) for r in results[SMALL_TALK] if decide(r, v)[0] not in PRE_RAG]
    print(f"False positives (real questions answered as small talk before RAG): {len(false_pos)}")
    for probe, decision, entry in false_pos:
        print(f"  {probe!r} -> {entry} ({decision})")
    print(f"Real questions the safety net would answer as small talk if RAG found nothing: {len(late)}")
    for probe, _, entry in late:
        print(f"  {probe!r} -> {entry}")
    print(f"Misses before RAG (small talk sent to RAG): {len(misses)}")
    for probe, decision in misses:
        print(f"  {probe!r} -> {decision}")
    print()


def print_sweeps(results: dict[str, list[Row]], v: Values) -> None:
    print("Fuzzy sweep (semantic values fixed): false positives / misses / tricky accepted")
    print("threshold " + "".join(f"  margin {m:.2f}" for m in FUZZY_MARGINS))
    for t in FUZZY_THRESHOLDS:
        cells = [counts(results, v._replace(fuzzy_threshold=t, fuzzy_margin=m)) for m in FUZZY_MARGINS]
        print(f"{t:9.2f} " + "".join(f"  {'%d / %d / %d' % c:>11}" for c in cells))
    print("\nSemantic sweep (fuzzy values fixed): false positives / misses / tricky accepted")
    print("threshold " + "".join(f"  margin {m:.2f}" for m in SEMANTIC_MARGINS))
    for t in SEMANTIC_THRESHOLDS:
        cells = [counts(results, v._replace(semantic_threshold=t, semantic_margin=m)) for m in SEMANTIC_MARGINS]
        print(f"{t:9.2f} " + "".join(f"  {'%d / %d / %d' % c:>11}" for c in cells))


def copied_probes() -> list[str]:
    kb = conversational.knowledge_base()
    known = {conversational.normalize(x) for e in kb.entries.values() for x in e.examples}
    known |= {conversational.normalize(a) for a in kb.anchors}
    return [p for _, probes in GROUPS for p in probes if conversational.normalize(p) in known]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fuzzy-threshold", type=float, default=settings.small_talk_fuzzy_threshold)
    parser.add_argument("--fuzzy-margin", type=float, default=settings.small_talk_fuzzy_margin)
    parser.add_argument("--semantic-threshold", type=float, default=settings.small_talk_semantic_threshold)
    parser.add_argument("--semantic-margin", type=float, default=settings.small_talk_semantic_margin)
    args = parser.parse_args()

    if conversational.knowledge_base() is None:
        sys.exit("The small-talk knowledge base is invalid; see the error above.")
    if copied := copied_probes():
        sys.exit(f"These probes copy knowledge-base examples or anchors; replace them: {copied}")
    # measure every layer, whatever .env turns off, without touching the live collection
    settings.small_talk_kb_enabled = settings.small_talk_semantic_enabled = True
    settings.chroma_dir = tempfile.mkdtemp(prefix="small_talk_calibration_")

    v = Values(args.fuzzy_threshold, args.fuzzy_margin, args.semantic_threshold, args.semantic_margin)
    print(f"Embedding model: {settings.embedding_id}")
    print(f"fuzzy threshold={v.fuzzy_threshold:.2f} margin={v.fuzzy_margin:.2f}   "
          f"semantic threshold={v.semantic_threshold:.2f} margin={v.semantic_margin:.2f}\n")
    results = {group: [measure(p) for p in probes] for group, probes in GROUPS}
    if all(r.semantic is None for rows in results.values() for r in rows if r.key and not r.rule):
        print("WARNING: no semantic scores; check the embedding provider (see the warning above).\n")

    print_table(results, v)
    print_problems(results, v)
    fp, misses, tricky = counts(results, v)
    print(f"Summary: {fp} false positives (of {len(MUST_GO_TO_RAG)}), {misses} misses before RAG (of "
          f"{len(SHOULD_BE_SMALL_TALK)}), {tricky} tricky messages answered as small talk (of {len(TRICKY)})\n")
    print_sweeps(results, v)


if __name__ == "__main__":
    main()
