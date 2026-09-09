#!/usr/bin/env python3
"""Evaluate the Stage 1 pre-filter against the labelled corpus.

    python scripts/evaluate_stage1.py            # full report at the configured threshold
    python scripts/evaluate_stage1.py --sweep    # threshold sweep on the training split

The point of this phase is to establish how far heuristics alone get, so that
Stage 2's contribution can later be stated as a difference rather than asserted.
Everything here exists to produce that number honestly.

The sweep runs on the training split only. Selecting a threshold on the test
split would borrow information from the set the reported figure is meant to
measure -- the same error as reporting validation metrics as a result, and an
easier one to make, because the sweep is fast and the test data is right there.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import config  # noqa: E402
from app.detection import stage1  # noqa: E402
from datagen import DATASET_VERSION  # noqa: E402

SWEEP_POINTS = [round(0.05 * i, 2) for i in range(1, 20)]

#: A false-positive rate an ingestion filter could plausibly live with. Used for
#: the second candidate threshold; the evaluation chapter should argue the point
#: rather than let this constant settle it.
FPR_BUDGET = 0.05


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_records(records: list[dict]) -> None:
    """Attach a Stage 1 score and latency to each record, in place."""
    for record in records:
        result = stage1.scan(record["text"], record.get("metadata", {}))
        record["_score"] = result.score
        record["_latency_ms"] = result.latency_ms
        record["_rules"] = result.rules_fired


def confusion(records: list[dict], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for record in records:
        predicted = record["_score"] >= threshold
        actual = record["label"] == 1
        if predicted and actual:
            counts["tp"] += 1
        elif predicted and not actual:
            counts["fp"] += 1
        elif not predicted and actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def print_sweep(train: list[dict]) -> tuple[float, float]:
    """Print the sweep and return the best-F1 and FPR-budget thresholds."""
    print("Threshold sweep — TRAINING SPLIT ONLY")
    print(f"{'thr':>6}{'precision':>11}{'recall':>9}{'F1':>8}{'FPR':>8}{'TP':>6}{'FP':>5}{'FN':>5}")
    print("-" * 58)

    best_f1 = (0.0, 0.0)
    fpr_pick = None

    for threshold in SWEEP_POINTS:
        counts = confusion(train, threshold)
        scores = metrics(counts)
        marker = ""
        if scores["f1"] > best_f1[1]:
            best_f1 = (threshold, scores["f1"])
        if fpr_pick is None and scores["fpr"] <= FPR_BUDGET:
            fpr_pick = threshold
            marker = f"  <- first FPR <= {FPR_BUDGET:.0%}"
        print(
            f"{threshold:>6.2f}{scores['precision']:>11.3f}{scores['recall']:>9.3f}"
            f"{scores['f1']:>8.3f}{scores['fpr']:>8.3f}"
            f"{counts['tp']:>6}{counts['fp']:>5}{counts['fn']:>5}{marker}"
        )

    print(f"\n  best F1 at threshold {best_f1[0]:.2f} (F1 {best_f1[1]:.3f})")
    if fpr_pick is not None:
        print(f"  lowest threshold holding FPR <= {FPR_BUDGET:.0%}: {fpr_pick:.2f}")
    return best_f1[0], fpr_pick if fpr_pick is not None else best_f1[0]


def print_split_metrics(records: list[dict], name: str, threshold: float) -> None:
    counts = confusion(records, threshold)
    scores = metrics(counts)
    print(
        f"{name:<12}{len(records):>6}"
        f"{scores['precision']:>11.3f}{scores['recall']:>9.3f}{scores['f1']:>8.3f}"
        f"{scores['fpr']:>8.3f}"
        f"{counts['tp']:>6}{counts['fp']:>5}{counts['fn']:>5}{counts['tn']:>5}"
    )


def print_per_technique(records: list[dict], threshold: float) -> None:
    """Recall by technique, reported as fractions.

    At roughly ten test examples per technique, a percentage implies a precision
    the sample size cannot support. Nine of ten is honest; ninety per cent is
    not.
    """
    caught: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for record in records:
        if record["label"] != 1:
            continue
        total[record["technique"]] += 1
        if record["_score"] >= threshold:
            caught[record["technique"]] += 1

    print(f"\n{'technique':<24}{'detected':>12}{'recall':>9}")
    print("-" * 45)
    for technique in sorted(total, key=lambda t: caught[t] / total[t]):
        print(
            f"{technique:<24}{caught[technique]:>5} / {total[technique]:<4}"
            f"{caught[technique] / total[technique]:>9.2f}"
        )


def print_per_benign_class(records: list[dict], threshold: float) -> None:
    """False positives by benign class.

    Locates the cost. A false positive on security training material means
    something quite different from one on an ordinary runbook.
    """
    flagged: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for record in records:
        if record["label"] != 0:
            continue
        total[record["benign_class"]] += 1
        if record["_score"] >= threshold:
            flagged[record["benign_class"]] += 1

    print(f"\n{'benign class':<24}{'false pos':>12}{'rate':>9}")
    print("-" * 45)
    for benign_class in sorted(total, key=lambda c: -flagged[c]):
        print(
            f"{benign_class:<24}{flagged[benign_class]:>5} / {total[benign_class]:<4}"
            f"{flagged[benign_class] / total[benign_class]:>9.2f}"
        )


def print_precision_by_band(records: list[dict]) -> None:
    """Precision within score bands — the evidence for the fast-reject question.

    The question the project reference leaves open is empirical: is there a score
    above which Stage 1 has never been wrong, and does a useful share of
    malicious documents reach it?
    """
    bands = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80), (0.50, 0.70)]
    print(f"\n{'score band':<16}{'documents':>11}{'malicious':>11}{'precision':>11}")
    print("-" * 49)
    for low, high in bands:
        inside = [r for r in records if low <= r["_score"] < high]
        if not inside:
            print(f"{f'{low:.2f} - {high:.2f}':<16}{0:>11}{'-':>11}{'-':>11}")
            continue
        malicious = sum(1 for r in inside if r["label"] == 1)
        print(
            f"{f'{low:.2f} - {high:.2f}':<16}{len(inside):>11}{malicious:>11}"
            f"{malicious / len(inside):>11.3f}"
        )

    for floor in (0.95, 0.90, 0.85, 0.80):
        above = [r for r in records if r["_score"] >= floor]
        if not above:
            continue
        malicious = sum(1 for r in above if r["label"] == 1)
        share = malicious / sum(1 for r in records if r["label"] == 1)
        print(
            f"  at or above {floor:.2f}: {len(above):>4} documents, "
            f"precision {malicious / len(above):.3f}, "
            f"covering {share:.1%} of all malicious documents"
        )


def print_latency(records: list[dict]) -> None:
    """Stage 1 latency.

    The two-stage architecture is only justified if the pre-filter is cheap, so
    this is where that claim stops being an assertion.
    """
    latencies = sorted(r["_latency_ms"] for r in records)
    p95 = latencies[int(len(latencies) * 0.95)]
    print(
        f"\nLatency over {len(latencies)} documents: "
        f"mean {statistics.mean(latencies):.3f} ms, "
        f"median {statistics.median(latencies):.3f} ms, "
        f"p95 {p95:.3f} ms, max {latencies[-1]:.3f} ms"
    )


def print_worst_false_positives(records: list[dict], threshold: float, limit: int = 5) -> None:
    """The highest-scoring benign documents, with the rules that flagged them."""
    false_positives = sorted(
        (r for r in records if r["label"] == 0 and r["_score"] >= threshold),
        key=lambda r: -r["_score"],
    )[:limit]

    if not false_positives:
        print("\nNo false positives at this threshold.")
        return

    print(f"\nHighest-scoring false positives ({len(false_positives)} shown):")
    for record in false_positives:
        print(f"  {record['id']}  {record['benign_class']:<22} score {record['_score']:.3f}")
        print(f"      rules: {', '.join(record['_rules'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / DATASET_VERSION)
    parser.add_argument("--sweep", action="store_true", help="print the threshold sweep")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    corpus = read_jsonl(args.data / "corpus.jsonl")
    challenge = read_jsonl(args.data / "challenge.jsonl")
    novel_path = args.data / "novel_phrasings.jsonl"
    novel = read_jsonl(novel_path) if novel_path.exists() else []

    started = time.perf_counter()
    score_records(corpus)
    score_records(challenge)
    score_records(novel)
    elapsed = time.perf_counter() - started
    print(f"Scored {len(corpus) + len(challenge) + len(novel)} documents in {elapsed:.2f}s\n")

    splits = {name: [r for r in corpus if r["split"] == name] for name in ("train", "val", "test")}

    if args.sweep:
        print_sweep(splits["train"])
        return 0

    threshold = args.threshold if args.threshold is not None else config.STAGE1_THRESHOLD
    print(f"Decision threshold: {threshold:.2f}  (selected on the training split)\n")

    print(f"{'split':<12}{'n':>6}{'precision':>11}{'recall':>9}{'F1':>8}{'FPR':>8}"
          f"{'TP':>6}{'FP':>5}{'FN':>5}{'TN':>5}")
    print("-" * 75)
    for name in ("train", "val", "test"):
        print_split_metrics(splits[name], name, threshold)
    print_split_metrics(challenge, "challenge", threshold)
    if novel:
        print_split_metrics(novel, "novel", threshold)

    print("\n" + "=" * 60)
    print("TEST SPLIT — the figures that count")
    print("=" * 60)
    print_per_technique(splits["test"], threshold)
    print_per_benign_class(splits["test"], threshold)
    print_worst_false_positives(splits["test"], threshold)

    print("\n" + "=" * 60)
    print("CHALLENGE SET — held-out document types")
    print("=" * 60)
    print_per_technique(challenge, threshold)
    print_per_benign_class(challenge, threshold)

    if novel:
        print("\n" + "=" * 60)
        print("NOVEL-PHRASING PROBE — the number that is not flattered")
        print("=" * 60)
        print(
            "Payloads phrased unlike anything in the training pools. The rules were\n"
            "written by someone who could read those pools, so every figure above is\n"
            "partly a measure of rule-to-generator fit. This one is not."
        )
        print_per_technique(novel, threshold)
        caught = sum(1 for r in novel if r["_score"] >= threshold)
        print(f"\n  overall recall on novel phrasings: {caught} / {len(novel)} "
              f"({caught / len(novel):.2f})")

    print("\n" + "=" * 60)
    print("FAST-REJECT EVIDENCE — training split only")
    print("=" * 60)
    print_precision_by_band(splits["train"])

    print_latency(corpus + challenge + novel)

    fired = Counter(rule for r in corpus for rule in r["_rules"])
    print("\nRule firing counts across the main corpus:")
    for rule, count in fired.most_common():
        print(f"  {rule:<26}{count:>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
