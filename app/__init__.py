"""AI Prompt Firewall.

A FastAPI middleware service that screens documents for indirect
prompt-injection payloads before they enter a RAG pipeline's context window.

The detector is two-stage: a lightweight heuristic pre-filter (Stage 1),
followed by a CPU-efficient transformer classifier (Stage 2). Neither stage
exists yet -- this package currently provides the service skeleton and the
API contract the later phases are built against.
"""

__version__ = "0.1.0"
