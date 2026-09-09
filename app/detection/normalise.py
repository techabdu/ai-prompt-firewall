"""Undoing obfuscation before the phrase rules run.

Three of the six attack techniques work by making a payload fail to *match*
rather than by hiding its intent. A Base64 blob, a character-spaced sentence and
a word carrying zero-width spaces all say exactly what an unobfuscated payload
says; a regular expression run against the raw text simply never sees it.

So the detector derives normalised variants of a document and runs the phrase
rules against each. A variant is produced only when its transformation actually
changes something, which keeps ordinary documents to a single pass and means a
hit on a derived variant is itself evidence that someone obfuscated something.

Nothing here imports anything outside the standard library. Stage 1 has to be
cheap enough that running it on every document is obviously worth it.
"""

import base64
import binascii
import re
from dataclasses import dataclass

#: Zero-width and formatting characters that survive text extraction.
ZERO_WIDTH_CHARS = frozenset({0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060})

#: The Unicode Tags block. Renders as nothing in essentially every viewer, which
#: is what makes it the documented vehicle for "ASCII smuggling".
TAG_BLOCK_START = 0xE0000
TAG_BLOCK_END = 0xE007F

# A run of single characters separated by whitespace: "I g n o r e". Eight
# elements minimum, so ordinary prose and initialisms do not qualify.
_SPACED_RUN = re.compile(r"(?<!\S)\S(?:[ \t]+\S){7,}(?!\S)")

# A standalone Base64-shaped run. Twenty-four characters minimum: shorter runs
# are far too common in identifiers to be worth decoding.
_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")


@dataclass(frozen=True)
class Variant:
    """One view of a document for the phrase rules to examine.

    ``derived`` marks a variant produced by undoing an obfuscation. A phrase
    rule firing on a derived variant means the payload was both present and
    concealed, which is worth more than the same phrase in plain text -- see
    the ``obfuscated_instruction`` rule.
    """

    name: str
    text: str
    derived: bool


def count_tag_characters(text: str) -> int:
    return sum(1 for char in text if TAG_BLOCK_START <= ord(char) <= TAG_BLOCK_END)


def count_zero_width(text: str) -> int:
    return sum(1 for char in text if ord(char) in ZERO_WIDTH_CHARS)


def decode_tag_characters(text: str) -> str:
    """Map Unicode Tag characters back to the ASCII they encode.

    Decoded in place rather than extracted, so that the recovered instruction
    sits in the surrounding context the phrase rules expect.
    """
    return "".join(
        chr(ord(char) - TAG_BLOCK_START)
        if TAG_BLOCK_START <= ord(char) <= TAG_BLOCK_END
        else char
        for char in text
    )


def strip_zero_width(text: str) -> str:
    """Remove zero-width characters, rejoining any word they were splitting."""
    return "".join(char for char in text if ord(char) not in ZERO_WIDTH_CHARS)


def collapse_spaced_runs(text: str) -> str:
    """Rejoin character-spaced text into ordinary words.

    Within a spaced run, a single space separates characters of one word and a
    wider gap separates words, so the two are undone differently. A run written
    without the wider gaps collapses into one long word, which still matches the
    phrase rules on the substring.
    """

    def collapse(match: re.Match) -> str:
        run = match.group(0)
        # Mark word gaps before removing the intra-word spaces, so the word
        # boundaries survive the collapse.
        marked = re.sub(r"[ \t]{2,}", "\x00", run)
        return marked.replace(" ", "").replace("\t", "").replace("\x00", " ")

    return _SPACED_RUN.sub(collapse, text)


def find_base64_runs(text: str) -> list[tuple[str, str]]:
    """Return ``(blob, decoded)`` for every Base64 run that decodes to text.

    A run that decodes to bytes rather than readable text is almost certainly a
    checksum or an identifier and is dropped here. That distinction is what lets
    the bare-blob rule carry a low weight: an opaque blob is weak evidence,
    while a blob that decodes into English is strong evidence.
    """
    decoded_runs: list[tuple[str, str]] = []

    for match in _BASE64_RUN.finditer(text):
        blob = match.group(0)
        padded = blob + "=" * (-len(blob) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue

        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        if _looks_like_prose(decoded):
            decoded_runs.append((blob, decoded))

    return decoded_runs


def _looks_like_prose(text: str) -> bool:
    """Whether decoded bytes read as English rather than as binary noise."""
    if len(text) < 12:
        return False
    printable = sum(1 for char in text if char.isprintable())
    letters_and_spaces = sum(1 for char in text if char.isalpha() or char.isspace())
    return printable == len(text) and letters_and_spaces / len(text) > 0.75


def build_variants(text: str) -> list[Variant]:
    """Produce the raw document plus any derived variant that differs from it.

    Variants are only emitted when their transformation changed something. An
    ordinary document therefore costs exactly one pass of the phrase rules, and
    a derived variant existing at all is a signal in its own right.
    """
    variants = [Variant("raw", text, derived=False)]

    deobfuscated = strip_zero_width(decode_tag_characters(text))
    if deobfuscated != text:
        variants.append(Variant("invisible_removed", deobfuscated, derived=True))

    despaced = collapse_spaced_runs(text)
    if despaced != text:
        variants.append(Variant("despaced", despaced, derived=True))

    decoded_runs = find_base64_runs(text)
    if decoded_runs:
        # The decoded text is examined on its own rather than spliced back into
        # the document: a decoded instruction is a complete statement, and the
        # surrounding prose adds nothing to matching it.
        variants.append(
            Variant("decoded", " ".join(decoded for _, decoded in decoded_runs), derived=True)
        )

    return variants
