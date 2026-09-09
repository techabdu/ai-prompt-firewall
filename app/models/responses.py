"""Response models for the scanning and health APIs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageResult(BaseModel):
    """Per-stage detection detail.

    Reported separately for each stage rather than collapsed into one number,
    for two reasons. First, the evaluation phase benchmarks latency, and the
    whole argument for a two-stage design is that the cheap stage runs on every
    document while the expensive one does not -- that claim is only measurable
    if the two timings are reported apart. Second, it makes a disagreement
    between the stages visible instead of hidden behind a single score.

    Populated by Stage 1 in Phase 3 and by Stage 2 in Phase 5. Until then both
    stage fields on a response are ``None``.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="This stage's confidence that the document carries a payload.",
    )
    triggered: bool = Field(
        ...,
        description="Whether this stage's score crossed its own decision threshold.",
    )
    latency_ms: float = Field(
        ...,
        description="Time this stage alone spent on the document, in milliseconds.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Stage-specific evidence. For Stage 1 this is which rules fired "
            "and where; for Stage 2, the raw class probabilities."
        ),
    )


class ScanResponse(BaseModel):
    """The verdict on a single document.

    The decision is binary. That matches the 0/1 labels in the dataset and the
    two-label Stage 2 model, and it is what a precision and recall comparison
    against Rebuff requires. A three-way verdict with a "suspicious" middle
    class would leave every metric table needing an argument for how that
    class is counted.

    ``score`` is kept alongside the binary decision so the threshold stays
    tunable: the false-positive-rate analysis needs to sweep it rather than
    being locked to whatever value the code happens to use.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_id": "wiki-4471",
                    "is_injection": False,
                    "score": 0.0,
                    "stage_1": None,
                    "stage_2": None,
                    "word_count": 118,
                    "truncated": False,
                    "latency_ms": 0.41,
                    "detector_version": "phase-1-stub",
                }
            ]
        }
    )

    doc_id: str | None = Field(
        default=None, description="Echoed unchanged from the request."
    )
    is_injection: bool = Field(
        ..., description="The binary decision. Always false while the detector is a stub."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence that the document carries an injection "
            "payload. Always 0.0 while the detector is a stub."
        ),
    )
    stage_1: StageResult | None = Field(
        default=None, description="Heuristic pre-filter detail. Null until Phase 3."
    )
    stage_2: StageResult | None = Field(
        default=None, description="Transformer classifier detail. Null until Phase 5."
    )
    word_count: int = Field(
        ..., description="Words counted in the submitted document body."
    )
    truncated: bool = Field(
        ...,
        description=(
            "True when the document exceeds the word cap, so that truncation "
            "is never silent. A payload sitting past the cut-off would "
            "otherwise be a false negative invisible in the metrics."
        ),
    )
    latency_ms: float = Field(
        ..., description="Total server-side processing time, in milliseconds."
    )
    detector_version: str = Field(
        ...,
        description=(
            "Which detector produced this verdict. 'phase-1-stub' means no "
            "detection ran and the result carries no information."
        ),
    )


class HealthResponse(BaseModel):
    """Liveness, plus an honest statement of which stages are actually loaded.

    A bare "ok" would be misleading for a service whose entire job is
    detection: it can be serving requests perfectly while carrying no detector
    at all. The two readiness flags make that distinction visible to whatever
    is probing the service.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "version": "0.1.0",
                    "stage_1_ready": False,
                    "stage_2_ready": False,
                }
            ]
        }
    )

    status: str = Field(..., description="'ok' when the service is serving requests.")
    version: str = Field(..., description="Service version.")
    stage_1_ready: bool = Field(
        ..., description="Whether the heuristic pre-filter is loaded."
    )
    stage_2_ready: bool = Field(
        ..., description="Whether the transformer classifier is loaded."
    )
