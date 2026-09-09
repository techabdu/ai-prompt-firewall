"""Request models for the scanning API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScanRequest(BaseModel):
    """A single document submitted for injection screening.

    The caller supplies text that has *already been extracted* from whatever
    format the document arrived in. This service deliberately does no document
    parsing of its own: keeping PDF and DOCX extraction on the caller's side
    keeps parser dependencies, and their failure modes, out of the detector.

    Metadata is carried as its own field rather than being flattened into
    ``text``. Payloads hidden in a document's title, author field or alt-text
    are one of the attack techniques this classifier is built to catch, and
    flattening them would make it impossible for Stage 1 to attribute a
    detection to a metadata field -- which in turn would remove per-technique
    reporting from the evaluation chapter.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_id": "wiki-4471",
                    "text": (
                        "Quarterly maintenance notes for the ingestion "
                        "cluster. Nodes are drained in rolling order and "
                        "rejoined once health checks pass."
                    ),
                    "metadata": {
                        "title": "Cluster Runbook",
                        "author": "platform-team",
                    },
                }
            ]
        }
    )

    doc_id: str | None = Field(
        default=None,
        description=(
            "Caller's identifier for this document. Echoed back unchanged so "
            "a pipeline can correlate a verdict with the document it scanned."
        ),
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Extracted document body. English-language text only.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Document metadata: title, author, alt-text and similar fields. "
            "Left deliberately free-form so that a new payload-bearing field "
            "discovered while building the dataset needs no model change."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject documents that are whitespace only.

        ``min_length`` alone would let a string of spaces through. A document
        with no characters in it is not something the detector can form an
        opinion about, so it is a client error rather than a clean verdict.
        """
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value
