"""Stage 1 measured against the labelled corpus.

This is the suite that answers the question Phase 3 exists to answer: how far do
heuristics alone get, before Stage 2 exists to close the gap?

Two kinds of assertion live here. Metric floors guard against a later change
quietly degrading detection -- they sit below measured performance by a margin,
because a floor set at the measured value fails on any harmless variation and
one set far below never fails at all. The novel-phrasing floors are different:
they pin a *finding*, not a performance level, and the finding is that rules
detecting a mechanism generalise while rules detecting a phrase do not.

Run with output visible::

    pytest tests/test_stage1_evaluation.py -s
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

from app import config
from app.detection import stage1

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v1"

# Floors sit below measured performance by a margin. Measured values at the time
# of writing are recorded beside each, so a future drift is visible rather than
# merely a failure.
TEST_PRECISION_FLOOR = 0.90   # measured 1.000
TEST_RECALL_FLOOR = 0.90      # measured 1.000
TEST_FPR_CEILING = 0.10       # measured 0.000
CHALLENGE_RECALL_FLOOR = 0.85  # measured 0.967
NOVEL_RECALL_FLOOR = 0.25     # measured 0.333
LATENCY_P95_CEILING_MS = 10.0  # measured 1.21

#: Techniques whose detection rests on a mechanism -- an invisible character, a
#: style attribute, an encoding -- rather than on a phrase. These are the ones
#: that should survive rephrasing.
MECHANISM_TECHNIQUES = ("hidden_text", "encoded_payload")


def _read(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not built; run scripts/build_dataset.py")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _scored(records: list[dict]) -> list[dict]:
    for record in records:
        result = stage1.scan(record["text"], record.get("metadata", {}))
        record["_score"] = result.score
        record["_latency_ms"] = result.latency_ms
        record["_flagged"] = result.triggered
    return records


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _scored(_read("corpus.jsonl"))


@pytest.fixture(scope="module")
def challenge() -> list[dict]:
    return _scored(_read("challenge.jsonl"))


@pytest.fixture(scope="module")
def novel() -> list[dict]:
    return _scored(_read("novel_phrasings.jsonl"))


def confusion(records: list[dict]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for record in records:
        predicted, actual = record["_flagged"], record["label"] == 1
        key = (
            "tp" if predicted and actual
            else "fp" if predicted
            else "fn" if actual
            else "tn"
        )
        counts[key] += 1
    return counts


def scores(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def recall_by_technique(records: list[dict]) -> dict[str, tuple[int, int]]:
    caught: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for record in records:
        if record["label"] != 1:
            continue
        total[record["technique"]] += 1
        caught[record["technique"]] += int(record["_flagged"])
    return {technique: (caught[technique], total[technique]) for technique in total}


# --- the headline figures ---------------------------------------------------


def test_report_metrics_for_every_split(corpus, challenge, novel, capsys) -> None:
    """Print the report. Not an assertion -- this is the phase's deliverable."""
    with capsys.disabled():
        print(f"\n\nStage 1 at threshold {config.STAGE1_THRESHOLD:.2f}")
        print(f"{'split':<12}{'n':>6}{'precision':>11}{'recall':>9}{'F1':>8}{'FPR':>8}")
        print("-" * 54)

        for name in ("train", "val", "test"):
            split = [r for r in corpus if r["split"] == name]
            metric = scores(confusion(split))
            print(
                f"{name:<12}{len(split):>6}{metric['precision']:>11.3f}"
                f"{metric['recall']:>9.3f}{metric['f1']:>8.3f}{metric['fpr']:>8.3f}"
            )

        for name, records in (("challenge", challenge), ("novel", novel)):
            metric = scores(confusion(records))
            print(
                f"{name:<12}{len(records):>6}{metric['precision']:>11.3f}"
                f"{metric['recall']:>9.3f}{metric['f1']:>8.3f}{metric['fpr']:>8.3f}"
            )

        print("\nRecall by technique — test / challenge / novel phrasing")
        test_split = [r for r in corpus if r["split"] == "test"]
        by_test = recall_by_technique(test_split)
        by_challenge = recall_by_technique(challenge)
        by_novel = recall_by_technique(novel)
        print(f"{'technique':<24}{'test':>10}{'challenge':>12}{'novel':>10}")
        print("-" * 56)
        for technique in sorted(by_test):
            print(
                f"{technique:<24}"
                f"{by_test[technique][0]:>5}/{by_test[technique][1]:<4}"
                f"{by_challenge[technique][0]:>7}/{by_challenge[technique][1]:<4}"
                f"{by_novel[technique][0]:>6}/{by_novel[technique][1]:<4}"
            )
        print()


def test_test_split_precision_holds(corpus) -> None:
    split = [r for r in corpus if r["split"] == "test"]
    assert scores(confusion(split))["precision"] >= TEST_PRECISION_FLOOR


def test_test_split_recall_holds(corpus) -> None:
    split = [r for r in corpus if r["split"] == "test"]
    assert scores(confusion(split))["recall"] >= TEST_RECALL_FLOOR


def test_false_positive_rate_stays_within_budget(corpus) -> None:
    split = [r for r in corpus if r["split"] == "test"]
    assert scores(confusion(split))["fpr"] <= TEST_FPR_CEILING


def test_every_technique_is_detected_at_all(corpus) -> None:
    """A technique with zero recall is a hole, not a weak spot."""
    split = [r for r in corpus if r["split"] == "test"]
    for technique, (caught, total) in recall_by_technique(split).items():
        assert caught > 0, f"{technique}: {caught}/{total} detected"


def test_challenge_set_recall_holds(challenge) -> None:
    """Held-out document types. Rules do not learn templates, so this should hold."""
    assert scores(confusion(challenge))["recall"] >= CHALLENGE_RECALL_FLOOR


# --- the finding ------------------------------------------------------------


def test_novel_phrasings_are_detected_above_the_floor(novel) -> None:
    """The number that is not flattered by rule-to-generator fit.

    Every other figure in this suite is measured against payloads drawn from the
    pool the rules were written against. This one is not, and it is far lower.
    """
    assert scores(confusion(novel))["recall"] >= NOVEL_RECALL_FLOOR


def test_mechanism_rules_survive_rephrasing(novel) -> None:
    """Detecting an invisible character does not depend on what it spells.

    Hidden text and encoded payloads are caught by the concealment itself, so
    rephrasing the instruction inside them changes nothing. This is the half of
    Stage 1 that genuinely generalises, and it should stay perfect.
    """
    by_technique = recall_by_technique(novel)
    for technique in MECHANISM_TECHNIQUES:
        caught, total = by_technique[technique]
        assert caught == total, f"{technique}: {caught}/{total} on novel phrasings"


def test_phrase_rules_do_not_survive_rephrasing(novel) -> None:
    """The other half does not, and the suite says so rather than hiding it.

    Recall on rephrased lexical attacks is far below recall on the corpus. That
    gap is the case for Stage 2, stated as a measurement rather than an
    assumption, and it is asserted here so that a future change cannot quietly
    erase the finding without someone noticing.
    """
    by_technique = recall_by_technique(novel)
    lexical = [t for t in by_technique if t not in MECHANISM_TECHNIQUES]

    caught = sum(by_technique[t][0] for t in lexical)
    total = sum(by_technique[t][1] for t in lexical)
    mechanism_caught = sum(by_technique[t][0] for t in MECHANISM_TECHNIQUES)
    mechanism_total = sum(by_technique[t][1] for t in MECHANISM_TECHNIQUES)

    assert caught / total < mechanism_caught / mechanism_total, (
        "Lexical rules now generalise as well as mechanism rules. Either the "
        "rules genuinely improved or the novel phrasings drifted towards the "
        "training pool — check which before updating this test."
    )


# --- the cheapness the architecture depends on ------------------------------


def test_stage_1_is_cheap(corpus, challenge) -> None:
    """The two-stage design is only justified if the pre-filter is cheap.

    Asserted rather than assumed, because the argument for running heuristics
    before the transformer rests entirely on this number.
    """
    latencies = sorted(r["_latency_ms"] for r in corpus + challenge)
    p95 = latencies[int(len(latencies) * 0.95)]

    assert p95 < LATENCY_P95_CEILING_MS, (
        f"p95 latency {p95:.2f} ms, mean {statistics.mean(latencies):.2f} ms"
    )


def test_scanning_is_deterministic(corpus) -> None:
    """The same document always scores the same. No randomness, no state."""
    record = corpus[0]
    first = stage1.scan(record["text"], record.get("metadata", {}))
    second = stage1.scan(record["text"], record.get("metadata", {}))

    assert first.score == second.score
    assert first.rules_fired == second.rules_fired
