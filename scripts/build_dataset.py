#!/usr/bin/env python3
"""Build the labelled corpus.

    python scripts/build_dataset.py

Deterministic: the same seed rebuilds the identical corpus. Writes the canonical
JSONL, the derived split files, a manifest with checksums, and a dataset card.

The export files the Colab fine-tuning script expects are produced separately by
``scripts/export_for_colab.py``, so that the canonical corpus and the flat
model-ready view are never confused for one another.
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.serialization import serialize_document  # noqa: E402
from datagen import DATASET_VERSION  # noqa: E402
from datagen.generator import (  # noqa: E402
    HARD_WORD_CAP,
    Record,
    assign_splits,
    generate_corpus,
)
from datagen.scaffolds import HELD_OUT_SCAFFOLDS, MAIN_SCAFFOLDS  # noqa: E402
from datagen.tokens import MAX_TOKENS, MODEL_NAME, count_tokens, estimate_tokens, load_tokenizer  # noqa: E402

DEFAULT_SEED = 20260909


def write_jsonl(path: Path, records: list[Record]) -> None:
    """Write records one per line.

    ``ensure_ascii`` is deliberately left on. The corpus contains Unicode Tag
    characters that are invisible by design; escaping them keeps the file pure
    ASCII, so a diff is readable and no editor or tool silently mangles them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(records: list[Record], attribute: str) -> dict[str, int]:
    counts = Counter(getattr(r, attribute) for r in records if getattr(r, attribute))
    return dict(sorted(counts.items()))


def build_manifest(
    main: list[Record],
    challenge: list[Record],
    seed: int,
    tokenizer_used: bool,
    data_dir: Path,
) -> dict:
    everything = main + challenge
    words = [r.word_count for r in everything]
    tokens = [r.token_count for r in everything if r.token_count is not None]

    manifest = {
        "dataset_version": DATASET_VERSION,
        "schema_version": main[0].schema_version,
        "built_on": date.today().isoformat(),
        "generator_seed": seed,
        "generated_by": "scripts/build_dataset.py",
        "language": "en",
        "word_cap": HARD_WORD_CAP,
        "token_ceiling": MAX_TOKENS,
        # The single most important field in this file. False means the token
        # counts are pessimistic estimates, not measurements, and the corpus has
        # not actually been cleared against the Stage 2 tokenizer.
        "token_counts_verified": tokenizer_used,
        "tokenizer": MODEL_NAME if tokenizer_used else None,
        "counts": {
            "total": len(everything),
            "main_corpus": len(main),
            "malicious": sum(1 for r in main if r.label == 1),
            "benign": sum(1 for r in main if r.label == 0),
            "challenge": len(challenge),
        },
        "splits": dict(sorted(Counter(r.split for r in everything).items())),
        "techniques": distribution(main, "technique"),
        "benign_classes": distribution(main, "benign_class"),
        "scaffolds": distribution(main, "scaffold"),
        "challenge_scaffolds": distribution(challenge, "scaffold"),
        "payload_location": distribution(main, "payload_location"),
        "payload_visibility": distribution(main, "payload_visibility"),
        "word_count": {
            "min": min(words),
            "max": max(words),
            "mean": round(statistics.mean(words), 1),
        },
        "token_count": (
            {
                "min": min(tokens),
                "max": max(tokens),
                "mean": round(statistics.mean(tokens), 1),
                "over_ceiling": sum(1 for t in tokens if t > MAX_TOKENS),
            }
            if tokens
            else None
        ),
        "files": {},
    }

    for path in sorted(data_dir.rglob("*.jsonl")):
        manifest["files"][str(path.relative_to(data_dir))] = {
            "sha256": sha256_of(path),
            "records": sum(1 for _ in path.open(encoding="utf-8")),
        }

    return manifest


DATASET_CARD = """# Dataset card — AI Prompt Firewall corpus {version}

Labelled corpus of benign and indirect-prompt-injection-bearing documents, for
training and evaluating a two-stage classifier that screens documents before
they enter a RAG pipeline's context window.

Built by `scripts/build_dataset.py` at seed `{seed}` on {built_on}. The build is
deterministic: the same seed reproduces this corpus exactly.

## Composition

| | Count |
|---|---|
| Main corpus | {main_total} |
| — malicious (label 1) | {malicious} |
| — benign (label 0) | {benign} |
| Challenge set | {challenge} |
| **Total** | **{total}** |

Splits: {split_line}

### Malicious techniques

{technique_table}

### Benign classes

{benign_table}

## Construction

Documents are composed by a seeded generator from hand-authored pools: document
scaffolds, paragraph banks, payload phrasings, and entity slot values. The
corpus is **not** generated by prompting a language model. The reason is
reproducibility — a seeded generator can be re-run from this repository and
produce byte-identical output, which is what allows the methodology to describe
the dataset as reconstructible. The cost is that linguistic diversity comes from
the size of the authored pools rather than from a model.

Payload phrasings are modelled on patterns documented in the promptware
kill-chain literature and published incidents. What is synthetic is the document
built around them.

### Scaffold parity

Every scaffold carries both benign and malicious documents, in roughly equal
numbers. This is the corpus's most important structural property: if malicious
documents were all of one document type and benign ones another, a classifier
could score near-perfectly by recognising the template while learning nothing
about injection. `tests/test_dataset.py` asserts the parity holds.

Ten scaffolds build the main corpus. Four are held out entirely and appear only
in the challenge set, which asks whether a detector generalises to document
types it has never seen.

## Field schema

| Field | Notes |
|---|---|
| `id` | Stable identifier. |
| `text` | Document body, as a text extractor would deliver it. |
| `label` | 0 clean, 1 injected. The only label the classifier sees. |
| `technique` | One of six for a malicious record; null for benign. |
| `technique_note` | One sentence on what the payload asks for and how it hides. |
| `benign_class` | One of four for a benign record; null for malicious. |
| `metadata` | Title, author, alt-text. Same shape as the service's `ScanRequest`. |
| `payload_location` | `body`, `metadata`, or null. |
| `payload_visibility` | `plain`, `hidden_markup`, `zero_width`, or null. |
| `scaffold` | Which document scaffold the record is built on. |
| `word_count` | Words in the body. |
| `token_count` | Tokens in the serialised document. See verification below. |
| `split` | `train`, `val`, `test`, or `challenge`. |
| `source` | `synthetic`. |
| `schema_version` | Schema revision. |

## Length and token verification

Documents are capped at {word_cap} words. That cap is a *proxy* for the Stage 2
encoder's {token_ceiling}-token ceiling and an unreliable one for this corpus
specifically: Base64 blobs, character-spaced text and Unicode Tag characters
tokenize far less efficiently than English prose.

**Token counts verified with the real tokenizer: {verified}.**

{verification_note}

Word counts: min {word_min}, max {word_max}, mean {word_mean}.

## Intended use

Training and evaluating an indirect-prompt-injection classifier at the document
ingestion boundary. Not suitable for evaluating direct chat-turn injection
defences, which are a different attack surface and deliberately absent.

## Limitations

1. **Synthetic construction.** This measures detection of these techniques as
   represented here, not of attacks in the wild. Results should be read as an
   upper bound on a narrow, well-specified distribution.
2. **Colour-matched text does not survive extraction.** Colour is a rendering
   property; by the time a document reaches the service it is plain text. The
   corpus models the case where an extractor leaked the style markup. Where an
   extractor strips styling entirely, the technique is invisible to this system
   by construction.
3. **Even class balance is a training convenience, not a deployment reality.**
   Injected documents are rare in a real corpus, and at a low base rate false
   positives dominate. Precision measured here is optimistic relative to
   deployment, and should be reported alongside precision at a realistic base
   rate.
4. **Per-technique test counts are near ten.** Report per-technique results as
   fractions rather than percentages, which would imply a precision the sample
   size does not support.
5. **Conditioned and delayed instructions are detected as static phrasing
   only.** Whether an instruction would ever be acted upon is agentic behaviour,
   outside what a document-boundary classifier can observe.
6. **The serialisation delimiter is forgeable.** Metadata is joined to the body
   with `[TITLE]`-style markers. The Stage 2 tokenizer is uncased, so those
   markers cannot be made unforgeable by an attacker willing to type them. This
   is the project's own subject in miniature and is documented in
   `app/serialization.py`.

## Files

| File | Records | SHA-256 |
|---|---|---|
{file_table}
"""


def render_dataset_card(manifest: dict) -> str:
    def table(rows: dict[str, int], header: str) -> str:
        lines = [f"| {header} | Count |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in rows.items()]
        return "\n".join(lines)

    verified = manifest["token_counts_verified"]
    if verified:
        note = (
            f"Every record was measured with `{manifest['tokenizer']}` and the build "
            f"rejects any record exceeding the ceiling. The Colab pre-flight diagnostic "
            f"should therefore report zero over-length examples; if it reports any, the "
            f"tokenizer versions differ between environments, which is worth knowing "
            f"before a training run rather than after."
        )
    else:
        note = (
            "**The real tokenizer was not available when this corpus was built**, so "
            "`token_count` holds a deliberately pessimistic estimate rather than a "
            "measurement. Run `python scripts/check_tokens.py` in an environment with "
            "access to the model hub before fine-tuning. Until that is done, the token "
            "ceiling is assumed, not established."
        )

    files = "\n".join(
        f"| `{name}` | {info['records']} | `{info['sha256'][:16]}…` |"
        for name, info in manifest["files"].items()
    )

    token_stats = manifest["token_count"] or {}

    return DATASET_CARD.format(
        version=manifest["dataset_version"],
        seed=manifest["generator_seed"],
        built_on=manifest["built_on"],
        main_total=manifest["counts"]["main_corpus"],
        malicious=manifest["counts"]["malicious"],
        benign=manifest["counts"]["benign"],
        challenge=manifest["counts"]["challenge"],
        total=manifest["counts"]["total"],
        split_line=", ".join(f"{k} {v}" for k, v in manifest["splits"].items()),
        technique_table=table(manifest["techniques"], "Technique"),
        benign_table=table(manifest["benign_classes"], "Class"),
        word_cap=manifest["word_cap"],
        token_ceiling=manifest["token_ceiling"],
        verified="yes" if verified else "**no**",
        verification_note=note,
        word_min=manifest["word_count"]["min"],
        word_max=manifest["word_count"]["max"],
        word_mean=manifest["word_count"]["mean"],
        token_max=token_stats.get("max", "n/a"),
        file_table=files,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / DATASET_VERSION)
    args = parser.parse_args()

    print(f"Building corpus at seed {args.seed} ...")
    main_records, challenge_records = generate_corpus(args.seed)
    assign_splits(main_records, args.seed)

    # Word cap: a hard failure, not a warning. A record over the cap cannot be
    # classified reliably and should never reach the corpus.
    over_cap = [r for r in main_records + challenge_records if r.word_count > HARD_WORD_CAP]
    if over_cap:
        print(f"FAIL: {len(over_cap)} record(s) exceed the {HARD_WORD_CAP}-word cap.")
        for record in over_cap[:5]:
            print(f"  {record.id}: {record.word_count} words")
        return 1

    tokenizer = load_tokenizer()
    if tokenizer is None:
        print(
            "WARNING: the Stage 2 tokenizer could not be loaded (transformers missing, "
            "or no access to the model hub).\n"
            "         Falling back to a pessimistic estimate. The manifest will record "
            "token_counts_verified = false.\n"
            "         Run scripts/check_tokens.py where the hub is reachable before "
            "fine-tuning."
        )
    else:
        print(f"Measuring tokens with {MODEL_NAME} ...")

    for record in main_records + challenge_records:
        serialized = serialize_document(record.text, record.metadata)
        record.token_count = (
            count_tokens(tokenizer, serialized) if tokenizer else estimate_tokens(serialized)
        )

    # The ceiling is enforced either way. With the real tokenizer this is a
    # measurement; with the fallback it is a pessimistic estimate, which still
    # makes a useful gate -- the estimate reads high, so a record that breaches
    # it is genuinely at risk rather than merely borderline.
    over_tokens = [r for r in main_records + challenge_records if r.token_count > MAX_TOKENS]
    if over_tokens:
        basis = "measured" if tokenizer else "estimated"
        print(f"FAIL: {len(over_tokens)} record(s) exceed the {MAX_TOKENS}-token ceiling ({basis}).")
        for record in over_tokens[:5]:
            print(f"  {record.id} ({record.technique or record.benign_class}): "
                  f"{record.token_count} tokens, {record.word_count} words")
        return 1

    data_dir: Path = args.out
    write_jsonl(data_dir / "corpus.jsonl", main_records)
    write_jsonl(data_dir / "challenge.jsonl", challenge_records)
    for split in ("train", "val", "test"):
        write_jsonl(
            data_dir / "splits" / f"{split}.jsonl",
            [r for r in main_records if r.split == split],
        )

    manifest = build_manifest(main_records, challenge_records, args.seed, tokenizer is not None, data_dir)
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (data_dir / "DATASET_CARD.md").write_text(render_dataset_card(manifest), encoding="utf-8")

    print(f"\nWrote {manifest['counts']['total']} records to {data_dir}")
    print(f"  main corpus : {manifest['counts']['main_corpus']} "
          f"({manifest['counts']['malicious']} malicious / {manifest['counts']['benign']} benign)")
    print(f"  splits      : {', '.join(f'{k} {v}' for k, v in manifest['splits'].items())}")
    print(f"  scaffolds   : {len(MAIN_SCAFFOLDS)} main, {len(HELD_OUT_SCAFFOLDS)} held out")
    print(f"  words       : min {manifest['word_count']['min']}, "
          f"max {manifest['word_count']['max']}, mean {manifest['word_count']['mean']}")
    if manifest["token_count"]:
        label = "measured" if manifest["token_counts_verified"] else "ESTIMATED"
        print(f"  tokens      : max {manifest['token_count']['max']} ({label}, ceiling {MAX_TOKENS})")
    print(f"  verified    : token_counts_verified = {manifest['token_counts_verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
