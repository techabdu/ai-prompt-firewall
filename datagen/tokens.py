"""Token counting against the Stage 2 tokenizer, with an offline fallback.

The 350-word cap is a proxy for the Stage 2 encoder's 512-token ceiling, and an
unreliable one for exactly the content this corpus contains. Base64 blobs,
character-spaced text and Unicode Tag characters tokenize far less efficiently
than English prose, so a document comfortably inside the word cap can still
exceed the token ceiling and be silently truncated during fine-tuning.

The authoritative check uses the real ``distilbert-base-uncased`` tokenizer.
Where that tokenizer cannot be loaded -- no ``transformers`` installed, or no
network access to fetch the vocabulary -- ``estimate_tokens`` provides a
deliberately pessimistic fallback so the build is not simply blind. The
fallback is *not* a substitute: it is calibrated, not exact, and any build that
relies on it is recorded as unverified in the dataset manifest so the gap
cannot be forgotten.
"""

import re

MODEL_NAME = "distilbert-base-uncased"

#: The Stage 2 encoder's sequence ceiling.
MAX_TOKENS = 512

# --- Fallback estimator rates ----------------------------------------------
# Deliberately above the real rates, so the estimate errs towards rejecting a
# borderline document rather than letting an over-length one through.
#
#   Ordinary English under an uncased WordPiece vocabulary runs about 1.3
#   tokens per word. 1.6 is used here.
#   Dense alphanumeric runs (base64) fragment into subwords at roughly one
#   token per 2.5 characters. One per 2 is used here.
#   Characters outside the vocabulary -- Unicode Tags among them -- become one
#   token each.

_TOKENS_PER_WORD = 1.6
_CHARS_PER_TOKEN_DENSE = 2.0
_SPECIAL_TOKEN_ALLOWANCE = 2  # [CLS] and [SEP]

# A "dense" chunk: long, unbroken, and mostly alphanumeric. Catches base64 and
# similar blobs without catching ordinary long words.
_DENSE_CHUNK = re.compile(r"^[A-Za-z0-9+/=_-]{20,}$")


def _is_out_of_vocabulary(char: str) -> bool:
    """Whether a character is one WordPiece will almost certainly not know.

    Covers the Unicode Tags block used for ASCII smuggling and the zero-width
    formatting characters, which are the two forms this corpus uses to hide
    text.
    """
    codepoint = ord(char)
    return 0xE0000 <= codepoint <= 0xE007F or codepoint in (0x200B, 0x200C, 0x200D, 0xFEFF)


def estimate_tokens(text: str) -> int:
    """Estimate the token count without the real tokenizer.

    Pessimistic by design -- see the module docstring. Use only when
    ``load_tokenizer`` returns ``None``.
    """
    exotic = sum(1 for char in text if _is_out_of_vocabulary(char))
    remaining = "".join(char for char in text if not _is_out_of_vocabulary(char))

    total = float(exotic)
    for chunk in remaining.split():
        if _DENSE_CHUNK.match(chunk):
            total += len(chunk) / _CHARS_PER_TOKEN_DENSE
        else:
            total += _TOKENS_PER_WORD

    return int(total) + _SPECIAL_TOKEN_ALLOWANCE


def load_tokenizer():
    """Load the Stage 2 tokenizer, or return ``None`` if it is unavailable.

    Returns ``None`` rather than raising, so that a caller can fall back to the
    estimator and record that it did. Loading needs neither torch nor a GPU:
    this is a dataset build-time step and nothing in the running service
    imports it.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        # Almost always no network access to the model hub. The caller decides
        # what to do about it; silently continuing is not one of the options.
        return None


def count_tokens(tokenizer, text: str) -> int:
    """Exact token count for ``text``, including the special tokens."""
    return len(tokenizer(text, truncation=False)["input_ids"])
