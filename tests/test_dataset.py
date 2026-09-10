"""Integrity tests for the labelled corpus.

Data deserves tests as much as code does, and for a sharper reason: every
failure these catch is silent. A corpus with the same document in train and
test produces excellent metrics and no error at all, and the first sign of
trouble would be a thesis result that does not survive scrutiny.

These run against the built corpus in ``data/v1``. Build it first::

    python scripts/build_dataset.py
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import pytest

from datagen.generator import (
    BENIGN_COUNTS,
    HARD_WORD_CAP,
    TECHNIQUE_COUNTS,
    assign_splits,
    generate_corpus,
)
from datagen.scaffolds import HELD_OUT_SCAFFOLDS, MAIN_SCAFFOLDS
from datagen.tokens import MAX_TOKENS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v1"

EXPECTED_FIELDS = {
    "id",
    "text",
    "label",
    "technique",
    "technique_note",
    "benign_class",
    "metadata",
    "payload_location",
    "payload_visibility",
    "scaffold",
    "word_count",
    "token_count",
    "split",
    "source",
    "schema_version",
}


def _read(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not built; run scripts/build_dataset.py")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _read("corpus.jsonl")


@pytest.fixture(scope="module")
def challenge() -> list[dict]:
    return _read("challenge.jsonl")


@pytest.fixture(scope="module")
def novel() -> list[dict]:
    """The novel-phrasing probe.

    Shipped data, gated on both caps by the build, and previously exempt from
    every check in this module -- no schema check, no word cap, no leakage
    check. A file that ships without them is exactly the silent failure this
    suite exists to catch.
    """
    return _read("novel_phrasings.jsonl")


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        pytest.skip("manifest not built; run scripts/build_dataset.py")
    return json.loads(path.read_text(encoding="utf-8"))


# --- schema -----------------------------------------------------------------


def test_every_record_has_exactly_the_expected_fields(corpus, challenge, novel) -> None:
    for record in corpus + challenge + novel:
        assert set(record) == EXPECTED_FIELDS, f"{record['id']} has the wrong fields"


def test_field_types_are_consistent(corpus, challenge, novel) -> None:
    for record in corpus + challenge + novel:
        assert isinstance(record["text"], str) and record["text"].strip()
        assert isinstance(record["metadata"], dict)
        assert isinstance(record["word_count"], int)
        assert record["label"] in (0, 1)


# --- labels and classes -----------------------------------------------------


def test_malicious_records_carry_a_technique_and_no_benign_class(corpus, challenge, novel) -> None:
    """A record cannot be both, and a malicious one must say which technique.

    Without this, a mislabelled record would quietly land in the wrong stratum
    and the per-technique breakdown would be wrong in a way nothing else checks.
    """
    for record in corpus + challenge + novel:
        if record["label"] == 1:
            assert record["technique"] in TECHNIQUE_COUNTS, record["id"]
            assert record["technique_note"], record["id"]
            assert record["benign_class"] is None, record["id"]


def test_benign_records_carry_a_class_and_no_technique(corpus, challenge, novel) -> None:
    for record in corpus + challenge + novel:
        if record["label"] == 0:
            assert record["benign_class"] in BENIGN_COUNTS, record["id"]
            assert record["technique"] is None, record["id"]
            assert record["payload_location"] is None, record["id"]


def test_metadata_payloads_are_located_in_metadata(corpus) -> None:
    """The technique is defined by where the payload sits, so check it does."""
    for record in corpus:
        if record["technique"] == "metadata_payload":
            assert record["payload_location"] == "metadata", record["id"]


def test_hidden_text_records_declare_a_concealment_form(corpus) -> None:
    for record in corpus:
        if record["technique"] == "hidden_text":
            assert record["payload_visibility"] in ("zero_width", "hidden_markup"), record["id"]


# --- duplication and leakage ------------------------------------------------


def test_no_duplicate_document_bodies(corpus, challenge, novel) -> None:
    """Duplicates inflate the corpus without adding information."""
    texts = [record["text"] for record in corpus + challenge + novel]
    duplicates = [text for text, count in Counter(texts).items() if count > 1]

    assert not duplicates, f"{len(duplicates)} duplicated document bodies"


def test_no_text_leaks_across_splits(corpus) -> None:
    """The failure that would inflate every reported metric and show no symptom.

    If the same document body appears in train and in test, the model has seen
    the answer. Nothing in the training output would look wrong, and the thesis
    would report a number that does not mean what it says.
    """
    by_split: dict[str, set[str]] = {}
    for record in corpus:
        by_split.setdefault(record["split"], set()).add(record["text"])

    splits = sorted(by_split)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            overlap = by_split[first] & by_split[second]
            assert not overlap, f"{len(overlap)} document(s) shared by {first} and {second}"


def test_challenge_set_shares_no_text_with_the_corpus(corpus, challenge) -> None:
    corpus_texts = {record["text"] for record in corpus}
    overlap = corpus_texts & {record["text"] for record in challenge}

    assert not overlap, f"{len(overlap)} document(s) shared with the challenge set"


def test_probe_shares_no_text_with_the_corpus_or_challenge_set(corpus, challenge, novel) -> None:
    """The probe composes from the same held-out scaffolds as the challenge set.

    A byte-identical body is therefore possible, and the same document sitting in
    two evaluation sets would make both figures wrong with no visible symptom.
    """
    existing = {record["text"] for record in corpus + challenge}
    overlap = existing & {record["text"] for record in novel}

    assert not overlap, f"{len(overlap)} probe document(s) duplicate corpus text"


# --- length limits ----------------------------------------------------------


def test_no_record_exceeds_the_word_cap(corpus, challenge, novel) -> None:
    for record in corpus + challenge + novel:
        assert record["word_count"] <= HARD_WORD_CAP, (
            f"{record['id']}: {record['word_count']} words"
        )


def test_no_record_exceeds_the_token_ceiling(corpus, challenge, novel) -> None:
    """Holds whether the counts were measured or estimated.

    When they are estimates the bar is if anything stricter, since the estimator
    reads high on purpose.
    """
    for record in corpus + challenge + novel:
        assert record["token_count"] <= MAX_TOKENS, (
            f"{record['id']} ({record['technique'] or record['benign_class']}): "
            f"{record['token_count']} tokens from {record['word_count']} words"
        )


def test_word_count_matches_the_text(corpus) -> None:
    for record in corpus:
        assert record["word_count"] == len(record["text"].split()), record["id"]


# --- balance and coverage ---------------------------------------------------


def test_class_balance_is_even(corpus) -> None:
    counts = Counter(record["label"] for record in corpus)

    assert counts[0] == counts[1] == 400


def test_technique_and_class_counts_match_the_specification(corpus) -> None:
    assert Counter(r["technique"] for r in corpus if r["label"] == 1) == TECHNIQUE_COUNTS
    assert Counter(r["benign_class"] for r in corpus if r["label"] == 0) == BENIGN_COUNTS


def test_every_split_contains_every_category(corpus) -> None:
    """A category absent from a split makes its metric there undefined."""
    categories = set(TECHNIQUE_COUNTS) | set(BENIGN_COUNTS)
    for split in ("train", "val", "test"):
        present = {
            r["technique"] or r["benign_class"] for r in corpus if r["split"] == split
        }
        assert present == categories, f"{split} is missing {categories - present}"


def test_splits_are_the_expected_sizes(corpus) -> None:
    assert Counter(r["split"] for r in corpus) == {"train": 560, "val": 120, "test": 120}


def test_each_split_is_close_to_balanced(corpus) -> None:
    """Stratification should carry the even class balance into every split."""
    for split in ("train", "val", "test"):
        labels = Counter(r["label"] for r in corpus if r["split"] == split)
        assert abs(labels[0] - labels[1]) <= 2, f"{split} is skewed: {dict(labels)}"


# --- the shortcuts a classifier must not be able to take --------------------


def test_every_scaffold_appears_under_both_labels(corpus) -> None:
    """The corpus's most important structural property.

    A scaffold used by only one label would let a classifier learn the document
    template as a proxy for the label and score well while detecting nothing.
    """
    for scaffold in MAIN_SCAFFOLDS:
        labels = {r["label"] for r in corpus if r["scaffold"] == scaffold.key}
        assert labels == {0, 1}, f"{scaffold.key} appears under labels {labels} only"


def test_scaffold_usage_is_balanced_across_labels(corpus) -> None:
    """Not merely present under both labels, but present in similar numbers."""
    for scaffold in MAIN_SCAFFOLDS:
        records = [r for r in corpus if r["scaffold"] == scaffold.key]
        malicious = sum(1 for r in records if r["label"] == 1)
        benign = len(records) - malicious
        assert abs(malicious - benign) <= 4, (
            f"{scaffold.key}: {malicious} malicious vs {benign} benign"
        )


def test_document_length_does_not_signal_the_label(corpus) -> None:
    """Length must not be a usable proxy for the label.

    If injected documents were systematically longer, a classifier could learn
    that instead of learning anything about injection -- and the corpus would
    flatter it.
    """
    malicious = [r["word_count"] for r in corpus if r["label"] == 1]
    benign = [r["word_count"] for r in corpus if r["label"] == 0]

    assert abs(statistics.mean(malicious) - statistics.mean(benign)) < 25


def test_held_out_scaffolds_never_appear_in_the_main_corpus(corpus) -> None:
    """Otherwise the challenge set is not measuring generalisation at all."""
    held_out = {scaffold.key for scaffold in HELD_OUT_SCAFFOLDS}
    used = {record["scaffold"] for record in corpus}

    assert not (used & held_out), f"held-out scaffolds leaked into training: {used & held_out}"


def test_challenge_set_uses_only_held_out_scaffolds(challenge) -> None:
    held_out = {scaffold.key for scaffold in HELD_OUT_SCAFFOLDS}

    assert {record["scaffold"] for record in challenge} <= held_out


# --- reproducibility --------------------------------------------------------


def test_the_build_is_deterministic() -> None:
    """Same seed, same corpus.

    This is what allows the methodology chapter to describe the dataset as
    reconstructible from the repository, so it is worth asserting rather than
    assuming.
    """
    first_main, first_challenge = generate_corpus(7)
    second_main, second_challenge = generate_corpus(7)
    assign_splits(first_main, 7)
    assign_splits(second_main, 7)

    assert [r.to_dict() for r in first_main] == [r.to_dict() for r in second_main]
    assert [r.to_dict() for r in first_challenge] == [r.to_dict() for r in second_challenge]


def test_a_different_seed_produces_a_different_corpus() -> None:
    """Guards against a seed that is accepted but never actually used."""
    first, _ = generate_corpus(7)
    second, _ = generate_corpus(8)

    assert [r.text for r in first] != [r.text for r in second]


# --- manifest ---------------------------------------------------------------


def test_manifest_counts_match_the_corpus(corpus, challenge, novel, manifest) -> None:
    assert manifest["counts"]["main_corpus"] == len(corpus)
    assert manifest["counts"]["challenge"] == len(challenge)
    assert manifest["counts"]["novel_phrasing_probe"] == len(novel)
    assert manifest["counts"]["total"] == len(corpus) + len(challenge) + len(novel)
    assert manifest["counts"]["malicious"] == sum(1 for r in corpus if r["label"] == 1)


def test_manifest_records_whether_tokens_were_verified(manifest) -> None:
    """The flag must exist and be a boolean, whichever way it reads.

    Its value is not asserted here: a corpus built without hub access is legally
    unverified, and this suite should not fail for that. What must never happen
    is the flag going missing, which would let an unverified corpus reach
    fine-tuning with nothing recording the fact.
    """
    assert isinstance(manifest["token_counts_verified"], bool)
