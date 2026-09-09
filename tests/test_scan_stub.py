"""Tests for the Phase 1 scan stub.

These cover the API contract and the input-validation rules only. There is no
detection logic to test yet; the tests that measure detection quality against
the labelled dataset arrive with Stage 1 in Phase 3.
"""

from fastapi.testclient import TestClient

from app import config

BENIGN_TEXT = (
    "Quarterly maintenance notes for the ingestion cluster. Nodes are drained "
    "in rolling order and rejoined once health checks pass. Operators should "
    "record the start and end time of each drain in the shared log."
)


def test_scan_accepts_a_valid_document(client: TestClient) -> None:
    """A well-formed request returns a verdict-shaped response."""
    response = client.post("/v1/scan", json={"text": BENIGN_TEXT})

    assert response.status_code == 200
    body = response.json()
    assert body["is_injection"] is False
    assert body["score"] == 0.0
    assert body["stage_1"] is None
    assert body["stage_2"] is None


def test_scan_marks_itself_as_a_stub(client: TestClient) -> None:
    """Stub output is labelled, so it cannot be mistaken for a real verdict.

    Phase 3 changes this string. If this test is still passing after Stage 1
    is wired in, the detector version was never updated and every logged
    result is mislabelled.
    """
    body = client.post("/v1/scan", json={"text": BENIGN_TEXT}).json()

    assert body["detector_version"] == "phase-1-stub"


def test_scan_counts_words(client: TestClient) -> None:
    """The reported word count matches the submitted body."""
    body = client.post("/v1/scan", json={"text": BENIGN_TEXT}).json()

    assert body["word_count"] == len(BENIGN_TEXT.split())


def test_scan_echoes_doc_id(client: TestClient) -> None:
    """The caller's identifier comes back unchanged, for correlation."""
    body = client.post(
        "/v1/scan", json={"doc_id": "wiki-4471", "text": BENIGN_TEXT}
    ).json()

    assert body["doc_id"] == "wiki-4471"


def test_scan_accepts_metadata(client: TestClient) -> None:
    """Metadata is carried as its own field and does not break the request.

    Metadata payloads are a Phase 2 attack category, so the field has to exist
    at the boundary from the start even though nothing reads it yet.
    """
    response = client.post(
        "/v1/scan",
        json={
            "text": BENIGN_TEXT,
            "metadata": {"title": "Cluster Runbook", "author": "platform-team"},
        },
    )

    assert response.status_code == 200


def test_scan_accepts_a_request_without_metadata(client: TestClient) -> None:
    """Metadata is optional."""
    response = client.post("/v1/scan", json={"text": BENIGN_TEXT})

    assert response.status_code == 200


def test_scan_rejects_a_missing_text_field(client: TestClient) -> None:
    """A request with no document is a client error."""
    response = client.post("/v1/scan", json={"doc_id": "wiki-4471"})

    assert response.status_code == 422


def test_scan_rejects_empty_text(client: TestClient) -> None:
    """An empty string is not a document."""
    response = client.post("/v1/scan", json={"text": ""})

    assert response.status_code == 422


def test_scan_rejects_whitespace_only_text(client: TestClient) -> None:
    """Nor is a string of whitespace.

    This is the case a bare min_length constraint would let through, which is
    why the model carries an explicit validator.
    """
    response = client.post("/v1/scan", json={"text": "   \n\t  "})

    assert response.status_code == 422


def test_scan_flags_an_over_cap_document(client: TestClient) -> None:
    """A document past the word cap is accepted, and flagged rather than cut.

    Rejecting it would produce hard failures in a real pipeline; truncating it
    silently would hide a payload sitting past the cut-off. So it is accepted,
    and the response says plainly that it is over length.
    """
    long_text = " ".join(["word"] * (config.MAX_WORDS + 50))

    response = client.post("/v1/scan", json={"text": long_text})

    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is True
    assert body["word_count"] == config.MAX_WORDS + 50


def test_scan_does_not_flag_a_document_at_the_cap(client: TestClient) -> None:
    """Exactly at the cap is within limits, not over it."""
    at_cap_text = " ".join(["word"] * config.MAX_WORDS)

    body = client.post("/v1/scan", json={"text": at_cap_text}).json()

    assert body["truncated"] is False
    assert body["word_count"] == config.MAX_WORDS


def test_scan_reports_latency(client: TestClient) -> None:
    """A latency figure is reported, for the Phase 6 benchmark."""
    body = client.post("/v1/scan", json={"text": BENIGN_TEXT}).json()

    assert body["latency_ms"] >= 0.0
