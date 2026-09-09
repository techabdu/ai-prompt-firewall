"""Document scanning endpoint.

Phase 1 provides the route, the validation and the response shape. No
detection runs: this is deliberately a stub, and it says so in every response
it returns.
"""

import time

from fastapi import APIRouter

from app import config
from app.models.requests import ScanRequest
from app.models.responses import ScanResponse

router = APIRouter(prefix=config.API_V1_PREFIX, tags=["scan"])


def count_words(text: str) -> int:
    """Count whitespace-separated words in a document body.

    Only the body is counted, not the metadata fields. The word cap exists to
    approximate the Stage 2 encoder's sequence limit, and how metadata is
    joined to the body before tokenization is a Phase 5 decision -- so
    including it in the count now would be guessing at a number that is not
    settled yet.
    """
    return len(text.split())


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Screen a document for indirect prompt-injection payloads",
)
def scan(request: ScanRequest) -> ScanResponse:
    """Screen one document and return a verdict.

    In Phase 1 the verdict is a stub. The endpoint validates the request,
    measures the document against the word cap, and returns a response whose
    ``detector_version`` marks it as carrying no detection result. Stage 1
    is wired in behind this same route in Phase 3, Stage 2 in Phase 5; the
    route, the request shape and the response shape do not change when they
    arrive.
    """
    started = time.perf_counter()

    word_count = count_words(request.text)

    # Nothing is actually truncated yet -- there is no Stage 2 to truncate for.
    # The flag reports that this document *would* be cut short once the
    # classifier exists, which is the point: an over-length document is
    # accepted rather than rejected, but never silently.
    truncated = word_count > config.MAX_WORDS

    latency_ms = (time.perf_counter() - started) * 1000

    return ScanResponse(
        doc_id=request.doc_id,
        is_injection=False,
        score=0.0,
        stage_1=None,  # Phase 3
        stage_2=None,  # Phase 5
        word_count=word_count,
        truncated=truncated,
        latency_ms=latency_ms,
        detector_version=config.DETECTOR_VERSION,
    )
