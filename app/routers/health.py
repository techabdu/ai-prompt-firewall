"""Health-check endpoint."""

from fastapi import APIRouter

from app import __version__, detection
from app.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and detector readiness",
)
def health() -> HealthResponse:
    """Report that the service is serving, and which stages are loaded.

    The readiness flags are read from the detection package rather than
    hardcoded here, so that wiring up a stage in a later phase does not
    require remembering to update the health check as well.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        stage_1_ready=detection.STAGE_1_READY,
        stage_2_ready=detection.STAGE_2_READY,
    )
