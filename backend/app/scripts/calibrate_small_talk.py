"""Calibrate the semantic small-talk threshold and margin for the configured embedding model.

Run: cd backend && python -m app.scripts.calibrate_small_talk [--threshold 0.75] [--margin 0.10]
Pick the lowest values with zero false positives; misses only mean the message goes to RAG as before.
"""
import argparse

from app.config import settings
from app.services import conversational, conversational_semantic

SHOULD_BE_SMALL_TALK = [
    "hey how's your day", "many many thanks", "ok cool", "catch ya", "what's this bot about", "hiya everyone",
    "appreciate the help", "thanks a bunch", "hello friends", "nice to meet you", "have a nice day",
    "thank you for your help", "awesome thanks", "bye for now", "how are you today", "how's everything",
    "what can this bot do", "what questions can you answer", "alright then", "cheers mate", "i'm off now",
]
MUST_GO_TO_RAG = [
    "what is an invoice", "hi can you explain invoice statuses", "thanks, what about draft invoices?",
    "how do refunds work", "who approves orders", "what is the late fee", "what is a draft invoice",
    "how are due dates set", "can I edit a sent invoice", "what is overtime", "what is per diem",
    "hello what does past due mean", "explain minimum billable", "what is the rush fee",
    "show me outstanding invoices", "how do I add mileage", "what is group by", "who are the clients",
]
TRICKY = [
    "help with invoices", "ok so what is the total", "good question", "morning shift timings", "what is this",
    "tell me more", "why", "is it free",
]
GROUPS = [("small talk", SHOULD_BE_SMALL_TALK), ("must go to RAG", MUST_GO_TO_RAG), ("tricky", TRICKY)]
SWEEP_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SWEEP_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]


def _accepted(scores, threshold: float, margin: float) -> bool:
    return scores is not None and scores.s_best >= threshold and scores.s_best - scores.s_doc >= margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=settings.semantic_small_talk_threshold)
    parser.add_argument("--margin", type=float, default=settings.semantic_small_talk_margin)
    args = parser.parse_args()
    settings.semantic_small_talk_enabled = True  # measure even if .env turns the layer off

    print(f"Embedding model: {settings.embedding_id}   threshold={args.threshold:.2f}  margin={args.margin:.2f}\n")
    results = {}  # group -> list of (rule intent, scores)
    for group, probes in GROUPS:
        print(f"== {group} ==")
        print(f"{'message':40} {'layer':9} {'intent':16} {'s_best':>6} {'s_doc':>6} {'margin':>6}  accepted")
        rows = []
        for probe in probes:
            rule = conversational.classify_conversational_intent(probe)
            scores = None if rule else conversational_semantic.score(probe)
            rows.append((rule, scores))
            if rule:
                print(f"{probe:40} {'rule':9} {rule:16} {'':>6} {'':>6} {'':>6}  yes")
            elif scores is None:
                print(f"{probe:40} {'skipped':9} {'-':16} {'':>6} {'':>6} {'':>6}  no")
            else:
                ok = "yes" if _accepted(scores, args.threshold, args.margin) else "no"
                print(f"{probe:40} {'semantic':9} {scores.intent:16} {scores.s_best:6.2f} {scores.s_doc:6.2f} "
                      f"{scores.s_best - scores.s_doc:6.2f}  {ok}")
        results[group] = rows
        print()

    def counts(threshold: float, margin: float) -> tuple[int, int, int]:
        def taken(rule, scores):
            return bool(rule) or _accepted(scores, threshold, margin)
        false_pos = sum(taken(*r) for r in results["must go to RAG"])
        misses = sum(not taken(*r) for r in results["small talk"])
        tricky = sum(taken(*r) for r in results["tricky"])
        return false_pos, misses, tricky

    fp, misses, tricky = counts(args.threshold, args.margin)
    print(f"Summary: {fp} false positives (of {len(MUST_GO_TO_RAG)}), {misses} misses (of "
          f"{len(SHOULD_BE_SMALL_TALK)}), {tricky} tricky messages treated as small talk (of {len(TRICKY)})\n")

    print("Sweep: false positives / misses / tricky accepted")
    print("threshold " + "".join(f"  margin {m:.2f}" for m in SWEEP_MARGINS))
    for t in SWEEP_THRESHOLDS:
        print(f"{t:9.2f} " + "".join(f"  {'%d / %d / %d' % counts(t, m):>11}" for m in SWEEP_MARGINS))


if __name__ == "__main__":
    main()
