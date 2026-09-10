"""Central configuration for the AI Prompt Firewall service.

Every tunable number the detector depends on lives here rather than being
scattered through the code. Each of these values has to be quoted and
justified in the thesis's implementation chapter, so there should only ever
be one place to look them up.
"""

# --- Service identity -------------------------------------------------------

SERVICE_NAME = "ai-prompt-firewall"

# Path prefix for the versioned detection API. Health checks sit outside it,
# because an orchestrator probing liveness should not have to know the API
# version.
API_V1_PREFIX = "/v1"

# Identifies which detector produced a given response. This string is what stops
# stub output being mistaken for a real verdict in a log or a screenshot.
#
# Still "phase-1-stub" after Phase 3, and correctly so: Stage 1 exists as a
# module but /v1/scan does not call it yet. The roadmap wires both stages behind
# the endpoint in Phase 5, and that is when this changes -- along with
# STAGE_1_READY in app/detection/__init__.py.
DETECTOR_VERSION = "phase-1-stub"


# --- Stage 1: heuristic pre-filter -------------------------------------------

# Score at or above which Stage 1 calls a document injected.
#
# Selected by sweeping thresholds on the TRAINING split alone -- never on
# validation, which Stage 2 uses for checkpoint selection, and never on test,
# which nothing may touch before the Phase 6 evaluation. Choosing a threshold on
# test data borrows information from the very set the reported number is meant
# to measure, and it is the more tempting error here because the sweep takes a
# second and the test split is sitting right there.
#
# Reproduce the sweep that produced this value with:
#     python scripts/evaluate_stage1.py --sweep
STAGE1_THRESHOLD = 0.30

# Longest body Stage 1 will scan, in characters. Roughly nine times the 350-word
# cap, so no legitimate document comes near it.
#
# It exists because nothing in ScanRequest enforces an upper bound -- over-length
# documents are flagged, not rejected -- and Stage 1's cost grows faster than
# linearly with input size. Without this, one oversized body took over a second
# and produced a multi-megabyte verdict. The cheapness of the pre-filter is the
# entire justification for running it before Stage 2, so it is enforced rather
# than assumed.
STAGE1_MAX_SCAN_CHARS = 20_000


# --- Document limits --------------------------------------------------------

# Per-document word ceiling. A hard constraint of the project: documents longer
# than this cannot be reliably classified by the Stage 2 encoder.
MAX_WORDS = 350

# Sequence ceiling of the Stage 2 encoder (a DistilBERT-class model, 512
# tokens).
#
# MAX_WORDS is a *proxy* for this limit, and an unreliable one for several of
# the Phase 2 attack categories. Base64 blobs, ASCII-art text and invisible
# Unicode characters tokenize far less efficiently than ordinary English prose,
# so a 350-word document built from one of those techniques can still exceed
# 512 tokens and be silently truncated during fine-tuning. Actual token counts
# must be checked with the target tokenizer for those categories specifically,
# rather than inferred from word counts.
#
# See PROJECT_REFERENCE.docx, section 6.
STAGE2_MAX_TOKENS = 512
