"""Joining a document's metadata and body into the single string Stage 2 reads.

The Stage 2 classifier consumes one string, but a document arrives as a body
plus a metadata map (see ``ScanRequest``). Something has to join them, and
whatever does the joining has to be *identical* at fine-tuning time and at
inference time. If the training corpus is built with one layout and the
deployed service uses another, the model sees a format it never trained on and
metadata-borne payloads quietly stop being detected -- a failure that produces
no error and no obvious symptom.

That is why this module lives in ``app/`` rather than alongside the dataset
scripts: Phase 5 imports it, rather than reimplementing it.

Format
------
Fields appear in a fixed order, one per line, each introduced by a marker::

    [TITLE] Cluster Runbook
    [AUTHOR] platform-team
    [ALT] Diagram of the drain sequence
    [BODY] Quarterly maintenance notes for the ingestion cluster ...

Absent or empty fields are omitted entirely rather than emitted as empty
markers, so that a document with no metadata serialises to just its body with
one marker, not to three empty lines of noise.

A note on the markers, worth stating plainly
--------------------------------------------
These markers are a *layout convention*, not a security boundary. They are
ordinary text, and any text an attacker can write into a document they can also
write into a title. ``strip_markers`` removes the literal forms as defence in
depth, but it cannot make the boundary unforgeable: the Stage 2 tokenizer is
uncased, so ``[TITLE]`` and ``[title]`` reduce to the same tokens, as does
``[ title ]``. A determined attacker can still fabricate something the model
reads as a field boundary.

This is worth naming rather than hiding, because it is the project's own
subject in miniature: a channel with no structural separation between
instruction and data, where the only thing distinguishing the two is a
convention both sides agreed to honour. The classifier mitigates it; the
serialisation cannot solve it.
"""

import re
from collections.abc import Mapping

TITLE_MARKER = "[TITLE]"
AUTHOR_MARKER = "[AUTHOR]"
ALT_MARKER = "[ALT]"
BODY_MARKER = "[BODY]"

#: Metadata keys read by the serialiser, in the order they are emitted.
#: Any other key in the metadata map is ignored -- the layout is fixed so that
#: the same document always produces the same string, which an open-ended
#: key set could not guarantee.
METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", TITLE_MARKER),
    ("author", AUTHOR_MARKER),
    ("alt_text", ALT_MARKER),
)

# Matches any field marker, case-insensitively and tolerating internal spacing,
# because the uncased Stage 2 tokenizer collapses those variants together.
_MARKER_PATTERN = re.compile(r"\[\s*(?:TITLE|AUTHOR|ALT|BODY)\s*\]", re.IGNORECASE)


def strip_markers(value: str) -> str:
    """Remove anything that could pass for a field marker from a value.

    Applied to every field before joining, so that a payload containing the
    literal text ``[BODY]`` cannot fabricate a field boundary in the serialised
    string. See the module docstring for why this reduces the problem without
    eliminating it.
    """
    return _MARKER_PATTERN.sub("", value)


def _clean_metadata_value(value: str) -> str:
    """Normalise a metadata value to a single line.

    Metadata fields are single-valued, so embedded newlines are collapsed:
    a title spanning two lines would otherwise look like the start of an
    unmarked field.
    """
    return " ".join(strip_markers(value).split())


def serialize_document(text: str, metadata: Mapping[str, str] | None = None) -> str:
    """Join a document body and its metadata into one string for Stage 2.

    Args:
        text: The document body, as a text extractor delivered it.
        metadata: Optional metadata map. Only the keys in ``METADATA_FIELDS``
            are read; anything else is ignored.

    Returns:
        The serialised document. Always ends with the body, so that truncation
        at the token ceiling removes body text rather than silently discarding
        the metadata fields -- which are small, and are themselves a payload
        carrier worth keeping.
    """
    metadata = metadata or {}
    lines: list[str] = []

    for key, marker in METADATA_FIELDS:
        value = _clean_metadata_value(str(metadata.get(key, "")))
        if value:
            lines.append(f"{marker} {value}")

    body = strip_markers(text).strip()
    lines.append(f"{BODY_MARKER} {body}")

    return "\n".join(lines)


__all__ = [
    "serialize_document",
    "strip_markers",
    "METADATA_FIELDS",
    "TITLE_MARKER",
    "AUTHOR_MARKER",
    "ALT_MARKER",
    "BODY_MARKER",
]
