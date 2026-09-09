"""Dataset construction for the AI Prompt Firewall.

Build-time only. Nothing in the running service imports this package, and it is
deliberately excluded from the installed distribution in ``pyproject.toml``:
the corpus is an input to fine-tuning, not a runtime dependency of the
classifier.

Layout:

    slots.py       entity pools filling template placeholders
    scaffolds.py   the ordinary documents that carry the corpus
    payloads.py    injection payloads, by technique
    benign.py      benign inserts, by class
    generator.py   deterministic composition and split assignment
    tokens.py      Stage 2 token counting, with an offline fallback
"""

DATASET_VERSION = "v1"
