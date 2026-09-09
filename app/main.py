"""Application entry point.

Run locally with:

    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app import __version__, config
from app.routers import health, scan

DESCRIPTION = """
Middleware that screens documents for indirect prompt-injection payloads
before they enter a RAG pipeline's context window.

The detector is two-stage: a heuristic pre-filter followed by a CPU-only
transformer classifier, both behind the single `/v1/scan` endpoint.

**Phase 1 skeleton.** Neither stage is built yet. `/v1/scan` validates input
and returns a stub result tagged `phase-1-stub`; it carries no detection
result and must not be read as one.
"""


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    A factory rather than a module-level constant, so that the test suite can
    build an isolated instance and Phase 5 has a single place to load the
    model artifact once at startup instead of per request.
    """
    application = FastAPI(
        title="AI Prompt Firewall",
        description=DESCRIPTION,
        version=__version__,
        contact={"name": "Kamaludeen Abdulkadir — CST/22/CBS/00753"},
    )

    # Health sits outside the versioned prefix: liveness probing should not
    # need to know the API version.
    application.include_router(health.router)
    application.include_router(scan.router)

    return application


app = create_app()


__all__ = ["app", "create_app", "config"]
