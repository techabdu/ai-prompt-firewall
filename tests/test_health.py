"""Tests for the health-check endpoint."""

from fastapi.testclient import TestClient

from app import __version__


def test_health_returns_ok(client: TestClient) -> None:
    """The service reports itself as serving."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_version(client: TestClient) -> None:
    """The reported version comes from the package, not a duplicated literal."""
    response = client.get("/health")

    assert response.json()["version"] == __version__


def test_health_reports_no_stages_loaded(client: TestClient) -> None:
    """Neither detection stage is wired in Phase 1, and the probe says so.

    This is the test that would fail loudly if a later phase wired a stage in
    without updating the readiness flags -- which is exactly when a misleading
    "ok" would be most dangerous.
    """
    body = client.get("/health").json()

    assert body["stage_1_ready"] is False
    assert body["stage_2_ready"] is False
