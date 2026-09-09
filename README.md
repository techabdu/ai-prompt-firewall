# AI Prompt Firewall

Middleware that screens documents for **indirect prompt-injection payloads** before they
enter a RAG pipeline's context window.

Retrieval-augmented generation pulls external documents into an LLM's context with no
structural boundary between an instruction and retrieved data. An attacker who can
influence a document likely to be retrieved — editing a wiki page, planting text in a PDF,
hiding instructions in file metadata — can embed commands the model treats as
authoritative, without compromising any credentials. This service sits at that ingestion
boundary and flags documents carrying such payloads.

Final-year BSc Cybersecurity thesis, Bayero University Kano.
Kamaludeen Abdulkadir — CST/22/CBS/00753.

---

## Status: Phase 1 of 7 — skeleton only

**No detection logic is built yet.** `/v1/scan` accepts documents, validates them and
returns a stub result tagged `phase-1-stub`. It carries no detection result and must not
be read as one.

| Phase | What it adds | State |
|---|---|---|
| 1 | FastAPI skeleton, API contract | **done** |
| 2 | Labelled synthetic dataset | not started |
| 3 | Stage 1 heuristic pre-filter + tests | not started |
| 4 | Stage 2 classifier, fine-tuned on Colab | not started |
| 5 | Both stages wired behind one endpoint | not started |
| 6 | Evaluation harness, benchmarked against Rebuff | not started |
| 7 | Upload-and-scan frontend (stretch) | not started |

---

## Architecture

A two-stage detector behind a single endpoint:

- **Stage 1 — heuristic pre-filter.** Regex, pattern matching, structural scoring. Cheap
  and fast, so it runs on every document.
- **Stage 2 — transformer classifier.** A CPU-efficient encoder catching the semantic and
  disguised cases heuristics miss. Fine-tuned off-repo on Google Colab; only the resulting
  model artifact is deployed, and it runs CPU-only.

Two stages rather than one because heuristics alone are evaded by anything that doesn't
match a known pattern, while a transformer on every document is expensive on laptop-grade
CPU. Running the cheap filter first means the classifier only sees what gets past it.

### Constraints

- **CPU-only at inference.** No GPU, no runtime dependency on a cloud inference API.
- **~350 words per document**, a proxy for the Stage 2 encoder's 512-token ceiling.
- **English-language documents only.**

---

## Running it locally

Requires Python 3.10 or newer. No GPU, no model download.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

The service starts on `http://127.0.0.1:8000`. Interactive API docs are at
`http://127.0.0.1:8000/docs`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "stage_1_ready": false,
  "stage_2_ready": false
}
```

The two readiness flags report which detection stages are actually loaded. A service that
answered a bare `"ok"` while carrying no detector would be misleading, given that detection
is its entire job.

### Scanning a document

```bash
curl -X POST http://127.0.0.1:8000/v1/scan \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_id": "wiki-4471",
    "text": "Quarterly maintenance notes for the ingestion cluster.",
    "metadata": {"title": "Cluster Runbook", "author": "platform-team"}
  }'
```

```json
{
  "doc_id": "wiki-4471",
  "is_injection": false,
  "score": 0.0,
  "stage_1": null,
  "stage_2": null,
  "word_count": 7,
  "truncated": false,
  "latency_ms": 0.04,
  "detector_version": "phase-1-stub"
}
```

### Running the tests

```bash
pytest
```

---

## The API contract

### `POST /v1/scan`

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `doc_id` | string | no | Echoed back, so a pipeline can correlate a verdict with its document. |
| `text` | string | **yes** | Extracted document body. Must contain a non-whitespace character. |
| `metadata` | object (string → string) | no | Title, author, alt-text and similar fields. |

The caller supplies text **already extracted** from whatever format the document arrived
in. This service does no document parsing of its own, which keeps PDF and DOCX parser
dependencies and their failure modes outside the detector.

Metadata is carried as its own field rather than flattened into `text`, because payloads
hidden in a title, author field or alt-text are one of the attack techniques being
classified — flattening them would make it impossible to attribute a detection to a
metadata field.

**Response**

| Field | Type | Notes |
|---|---|---|
| `doc_id` | string \| null | Echoed from the request. |
| `is_injection` | boolean | The binary decision. |
| `score` | float 0.0–1.0 | Confidence, so the decision threshold stays tunable. |
| `stage_1` | object \| null | Heuristic detail. Null until Phase 3. |
| `stage_2` | object \| null | Classifier detail. Null until Phase 5. |
| `word_count` | integer | Words in the submitted body. |
| `truncated` | boolean | True when the document exceeds the word cap. |
| `latency_ms` | float | Server-side processing time. |
| `detector_version` | string | `phase-1-stub` while no detection runs. |

The decision is binary rather than a three-way verdict: that matches the 0/1 dataset labels
and the two-label Stage 2 model, and is what a precision/recall comparison against Rebuff
requires. The `score` alongside it keeps the threshold sweepable for false-positive-rate
analysis.

**Over-length documents are accepted, flagged, and not rejected.** `truncated: true` means
the document is past the word cap and will be cut short once Stage 2 exists. Rejecting
outright would cause hard failures in a real pipeline; truncating silently would hide a
payload sitting past the cut-off, producing a false negative invisible in the metrics.

### `GET /health`

Returns `status`, `version`, and the `stage_1_ready` / `stage_2_ready` flags described
above.

---

## Layout

```
app/
  main.py            application factory, router mounting
  config.py          named constants: word cap, token ceiling, detector version
  models/
    requests.py      ScanRequest
    responses.py     ScanResponse, StageResult, HealthResponse
  routers/
    health.py        GET /health
    scan.py          POST /v1/scan
  detection/         stage1.py arrives in Phase 3, stage2.py in Phase 5
tests/
docs/                Phase specifications
```

`app/config.py` holds every tunable number in one place — `MAX_WORDS`,
`STAGE2_MAX_TOKENS`, `DETECTOR_VERSION` — because each has to be quoted and justified in
the thesis, and hunting them down across modules is how they drift apart.

---

## Scope

Deliberately **not** built, at any phase:

- Dual-LLM verification, or any second "judge" model.
- Agentic orchestration, sub-agent routing, tool-invocation-hijack defences.
- Memory or identity-file poisoning defences.
- Multimodal (image/audio) injection handling.
- RAG "jamming" / blocker-document denial-of-service defences — a different threat class.
- Direct chat-turn injection as a build target; relevant only as a baseline comparison.

Each exclusion is a decision with a stated reason, not an oversight. See
`PROJECT_REFERENCE.docx` section 4.

## Project documents

- `CLAUDE.md` — locked architecture and hard constraints.
- `PROJECT_REFERENCE.docx` — full build context, scope rationale, dataset design guidance,
  seven-phase roadmap.
- `docs/Phase1_Specification.docx` — this phase's specification, design decisions and
  constraint traceability.
