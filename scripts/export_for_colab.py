#!/usr/bin/env python3
"""Produce the flat files the Colab fine-tuning script expects.

    python scripts/export_for_colab.py

The fine-tuning brief specifies ``dataset_train.json`` and ``dataset_val.json``,
each holding objects with exactly the keys ``text`` and ``label``. This script
generates precisely that from the canonical corpus, so the fine-tuning script
never has to be edited and the two views can never drift apart by hand.

Two things this script does deliberately:

**It exports the serialised document, not the raw body.** ``text`` is the
document with its metadata joined on by ``app.serialization``. That is the exact
string Phase 5 will hand the classifier at inference time. Exporting the raw
body instead would train the model on a format it never sees in production, and
metadata-borne payloads would be invisible to it.

**It does not export the test split.** Selecting the best checkpoint on
validation F1 makes the validation set a model-selection set, so its metrics are
optimistically biased. The test split stays in this repository, untouched by
fine-tuning, until the Phase 6 evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.serialization import serialize_document  # noqa: E402
from datagen import DATASET_VERSION  # noqa: E402

#: Split name to the filename the fine-tuning script loads. These names are
#: fixed by that script and must not be changed here.
EXPORTS = {
    "train": "dataset_train.json",
    "val": "dataset_val.json",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / DATASET_VERSION)
    args = parser.parse_args()

    splits_dir: Path = args.data / "splits"
    export_dir: Path = args.data / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    if not splits_dir.exists():
        print(f"No splits found at {splits_dir}. Run scripts/build_dataset.py first.")
        return 1

    for split, filename in EXPORTS.items():
        records = read_jsonl(splits_dir / f"{split}.jsonl")
        flat = [
            {
                "text": serialize_document(record["text"], record.get("metadata", {})),
                "label": record["label"],
            }
            for record in records
        ]
        target = export_dir / filename
        target.write_text(json.dumps(flat, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        positives = sum(1 for row in flat if row["label"] == 1)
        print(
            f"{filename:<22} {len(flat):>4} records "
            f"({positives} injected / {len(flat) - positives} clean)"
        )

    print(f"\nWritten to {export_dir}")
    print("Upload both files to the Colab session working directory before training.")
    print("The test split is deliberately NOT exported: it is held back for Phase 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
