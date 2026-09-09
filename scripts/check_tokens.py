#!/usr/bin/env python3
"""Verify the corpus against the Stage 2 token ceiling.

    python scripts/check_tokens.py            # report and gate
    python scripts/check_tokens.py --update   # also write verified counts back

The 350-word cap is a proxy for the encoder's 512-token ceiling, and an
unreliable one for this corpus specifically: Base64 blobs, character-spaced
text and Unicode Tag characters tokenize far less efficiently than English
prose. A document comfortably inside the word cap can still exceed the token
ceiling and be silently truncated during fine-tuning, which is a defect that
produces no error and no obvious symptom -- the model simply never sees the
payload.

This script requires the real tokenizer and **fails rather than falling back**.
The estimator in ``datagen.tokens`` exists so that a build without network
access is not entirely blind; it is not a substitute for the measurement, and a
corpus that has only ever been estimated is recorded as unverified in the
manifest.

Run this once, in an environment with access to the model hub, before Phase 4.
The Colab pre-flight diagnostic should then report zero over-length examples.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.serialization import serialize_document  # noqa: E402
from datagen import DATASET_VERSION  # noqa: E402
from datagen.tokens import MAX_TOKENS, MODEL_NAME, count_tokens, load_tokenizer  # noqa: E402

CORPUS_FILES = ("corpus.jsonl", "challenge.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / DATASET_VERSION)
    parser.add_argument(
        "--update",
        action="store_true",
        help="write measured counts back into the corpus and flip the manifest's "
             "token_counts_verified flag to true",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    if tokenizer is None:
        print(
            f"FAIL: could not load the {MODEL_NAME} tokenizer.\n\n"
            "  This script measures rather than estimates, so it will not continue "
            "without it.\n"
            "  Install transformers (pip install transformers) and run this where "
            "huggingface.co is reachable.\n"
            "  The corpus remains marked token_counts_verified = false until it does."
        )
        return 2

    print(f"Measuring with {MODEL_NAME} (ceiling {MAX_TOKENS} tokens)\n")

    by_category: dict[str, list[int]] = defaultdict(list)
    over: list[tuple[str, str, int, int]] = []
    loaded: dict[str, list[dict]] = {}

    for filename in CORPUS_FILES:
        path = args.data / filename
        if not path.exists():
            print(f"No corpus at {path}. Run scripts/build_dataset.py first.")
            return 1
        records = read_jsonl(path)
        loaded[filename] = records

        for record in records:
            serialized = serialize_document(record["text"], record.get("metadata", {}))
            tokens = count_tokens(tokenizer, serialized)
            record["token_count"] = tokens

            category = record["technique"] or record["benign_class"]
            by_category[category].append(tokens)
            if tokens > MAX_TOKENS:
                over.append((record["id"], category, tokens, record["word_count"]))

    # The categories the project reference singles out as tokenizing badly are
    # listed first, because they are the ones this check exists for.
    risky = {"encoded_payload", "hidden_text"}
    ordering = sorted(by_category, key=lambda c: (c not in risky, -max(by_category[c])))

    print(f"{'category':<24}{'n':>5}{'mean':>8}{'max':>7}{'headroom':>10}")
    print("-" * 54)
    for category in ordering:
        counts = by_category[category]
        peak = max(counts)
        marker = "  <- watch" if category in risky else ""
        print(
            f"{category:<24}{len(counts):>5}{statistics.mean(counts):>8.0f}"
            f"{peak:>7}{MAX_TOKENS - peak:>10}{marker}"
        )

    every = [t for counts in by_category.values() for t in counts]
    print("-" * 54)
    print(f"{'ALL':<24}{len(every):>5}{statistics.mean(every):>8.0f}{max(every):>7}"
          f"{MAX_TOKENS - max(every):>10}")

    if over:
        print(f"\nFAIL: {len(over)} record(s) exceed the {MAX_TOKENS}-token ceiling:")
        for record_id, category, tokens, words in over[:20]:
            print(f"  {record_id:<14}{category:<24}{tokens} tokens from {words} words")
        print(
            "\nThese would be silently truncated during fine-tuning. Shorten the prose "
            "budget for the affected category in datagen/generator.py and rebuild."
        )
        return 1

    print(f"\nPASS: every record is within the {MAX_TOKENS}-token ceiling.")

    if args.update:
        for filename, records in loaded.items():
            write_jsonl(args.data / filename, records)

        manifest_path = args.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["token_counts_verified"] = True
        manifest["tokenizer"] = MODEL_NAME
        manifest["token_count"] = {
            "min": min(every),
            "max": max(every),
            "mean": round(statistics.mean(every), 1),
            "over_ceiling": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        print("Updated the corpus with measured counts and set "
              "token_counts_verified = true.")
        print("Re-run scripts/build_dataset.py only if you change the generator; it "
              "would otherwise overwrite these measurements with estimates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
