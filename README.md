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
| 2 | Labelled synthetic dataset | **done** |
| 3 | Stage 1 heuristic pre-filter + tests | **done** |
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

---

## The dataset

860 labelled documents: 800 in the main corpus, evenly split between benign and
indirect-injection-bearing, plus a 60-document challenge set. English only, every
document inside the word cap.

```bash
python scripts/build_dataset.py      # rebuild the corpus (deterministic)
python scripts/check_tokens.py       # verify against the 512-token ceiling
python scripts/export_for_colab.py   # write the flat files Colab expects
```

`data/v1/DATASET_CARD.md` documents composition, construction and limitations.
`data/v1/manifest.json` carries counts, distributions, checksums and the seed.

### Canonical corpus, derived exports

`data/v1/corpus.jsonl` is the source of truth: one record per line, full schema —
technique, benign class, metadata, payload location and visibility, scaffold,
split. JSONL because adding an example is a one-line diff, where a JSON array
reflows on every insert and makes a dataset change unreviewable.

`data/v1/export/dataset_train.json` and `dataset_val.json` are generated from it,
flattened to exactly the two keys (`text`, `label`) the Colab fine-tuning script
expects. They are regenerable with one command and never edited by hand, so the
fine-tuning script never has to change and the two views cannot drift apart.

**The test split is not exported.** Selecting the best checkpoint on validation F1
makes validation a model-selection set, so its metrics are optimistically biased.
The test split stays here, untouched by fine-tuning, until Phase 6.

### What the corpus is built to resist

Malicious documents cover six techniques: `imperative_override`,
`disguised_legitimate`, `hidden_text`, `encoded_payload`, `metadata_payload`,
`conditioned_delayed`.

The benign half is where a dataset is won or lost. Alongside `ordinary`
documents it contains `discusses_injection` (security training material quoting
attack strings as examples), `lexical_decoy` ("ignore the third column", "follow
the instructions on page four") and `imperative_benign` — genuine runbooks full
of privileged commands, structurally near-identical to the `disguised_legitimate`
attack. A corpus without these lets a keyword matcher score well while detecting
nothing.

Every scaffold carries both labels in roughly equal numbers, so the document
template cannot serve as a proxy for the label. Four scaffolds are held out of
training entirely and appear only in the challenge set, which asks whether a
detector generalises to document types it has never seen.

---

## Stage 1: the heuristic pre-filter

Fourteen rules — regular expressions, character detection, structural patterns.
No model, no learned parameters, standard library only. Mean latency **1.08 ms**
per document (p95 1.52 ms), which is what justifies running it on everything
before the transformer sees anything.

**Mechanism weights sit below the decision threshold on purpose.** A
Base64-shaped run, character spacing and a couple of zero-width characters are
all produced by entirely ordinary documents — checksums, tables flattened by a
text extractor, text copied out of a web page. What carries weight is the
*content* recovered from the mechanism: an instruction that appears only once
the obfuscation is undone, or a blob that decodes into English rather than into
the noise a checksum decodes to. The mechanism alone is a coincidence.

```bash
python scripts/evaluate_stage1.py           # full report
python scripts/evaluate_stage1.py --sweep   # threshold sweep (training split only)
pytest tests/test_stage1_evaluation.py -s   # the same metrics, as tests
```

### Results at threshold 0.30

| Split | n | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| train | 560 | 1.000 | 0.986 | 0.993 | 0.000 |
| val | 120 | 1.000 | 1.000 | 1.000 | 0.000 |
| test | 120 | 1.000 | 1.000 | 1.000 | 0.000 |
| challenge | 60 | 1.000 | 0.967 | 0.983 | 0.000 |
| **novel phrasings** | 24 | 1.000 | **0.333** | 0.500 | — |

These figures survived a post-review repair of eleven detection defects — six
false-positive classes and two outright evasions, none of which the corpus could
see, because every one needs a document feature the generator never produces: a
table, a checksum, an uppercase heading, a stray quote, a multi-line payload.
`tests/test_stage1_rules.py` pins each as a regression.

**Read the last row first.** Perfect scores on the corpus are not what they
look like. The rules were written by someone who could read `datagen/payloads.py`,
so those figures partly measure how well the rules fit the generator. The
challenge set does not expose this — it varies the document scaffold while
drawing payloads from the same pool.

`data/v1/novel_phrasings.jsonl` does expose it: 24 payloads saying the same
things in words that appear nowhere in the training pools. Recall drops from
1.000 to 0.333, and it drops in a very specific way:

| Technique | test | challenge | novel phrasing |
|---|---|---|---|
| hidden_text | 10/10 | 5/5 | **4/4** |
| encoded_payload | 10/10 | 5/5 | **4/4** |
| imperative_override | 10/10 | 5/5 | **0/4** |
| disguised_legitimate | 10/10 | 5/5 | **0/4** |
| conditioned_delayed | 10/10 | 5/5 | **0/4** |
| metadata_payload | 10/10 | 4/5 | **0/4** |

Rules that detect a **mechanism** — an invisible character, a style attribute, an
encoding — generalise perfectly, because detecting a Unicode Tag character does
not depend on what it spells. Rules that detect a **phrase** do not generalise at
all, because an attacker has no reason to use the wording a rule author happened
to anticipate.

That split is the case for Stage 2, stated as a measurement rather than an
assumption. It is asserted in `tests/test_stage1_evaluation.py` so a later change
cannot quietly erase it.

### Fast-reject: not built, and the evidence says not to

The project reference leaves open whether Stage 1 may block a document outright
on a high-confidence match. The evidence needed is a score band where precision
is perfect and a lower band where it is not. On the training split precision is
1.000 in *every* band, down to 0.50–0.70.

Uniform perfection is not evidence that fast-rejecting is safe; it is evidence
that the corpus cannot distinguish a safe threshold from an unsafe one. Combined
with 0.333 recall on rephrased attacks — the rules are far more fragile than the
corpus suggests — there is no basis for letting Stage 1 block anything on its
own. No fast-reject path is built, and none should appear in an architecture
diagram.

---

## Layout

```
app/
  main.py            application factory, router mounting
  config.py          named constants: word cap, token ceiling, detector version
  serialization.py   the metadata-and-body join Stage 2 reads
  models/
    requests.py      ScanRequest
    responses.py     ScanResponse, StageResult, HealthResponse
  routers/
    health.py        GET /health
    scan.py          POST /v1/scan
  detection/
    stage1.py        the heuristic pre-filter: scoring and entry point
    rules.py         the thirteen rules and their weights
    normalise.py     de-obfuscation, run before the phrase rules
                     stage2.py arrives in Phase 5
datagen/             corpus construction (build-time only, not installed)
  scaffolds.py       the ordinary documents that carry the corpus
  payloads.py        injection payloads, by technique
  benign.py          benign inserts, by class
  novel_phrasings.py the generalisation probe
  generator.py       deterministic composition and split assignment
  tokens.py          Stage 2 token counting, with an offline fallback
scripts/             build_dataset, check_tokens, export_for_colab, evaluate_stage1
data/v1/             the corpus, splits, exports, manifest, dataset card
tests/
docs/                Phase specifications
```

Stage 1 is **not wired into `/v1/scan`** — the roadmap places that in Phase 5, and
`stage_1_ready` on `/health` stays `false` until the endpoint actually calls the
detector. The flag means loaded into the request path, not present in the
repository, and it would be lying if it said otherwise.

`app/serialization.py` sits in `app/` rather than in `datagen/` on purpose: it
defines the exact string the classifier is trained on, and Phase 5 imports it so
that inference and fine-tuning cannot use different layouts. A mismatch there
would silently stop metadata-borne payloads being detected.

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
