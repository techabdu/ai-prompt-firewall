"""Detection stages.

    normalise.py  -- de-obfuscation, run before the phrase rules. Owns the
                     character classes and shape patterns both halves share.
    rules.py      -- the thirteen rules and their weights.
    stage1.py     -- heuristic pre-filter: scoring and the public entry point.
                     Built in Phase 3.
    stage2.py     -- CPU-only transformer classifier, loading the model artifact
                     fine-tuned off-repo on Colab. Arrives in Phase 5.

The readiness flags below are the single place the rest of the service asks
whether a stage is wired into the request path.

Both are still False after Phase 3, and that is deliberate rather than an
oversight. Stage 1 exists as a module, but the roadmap places the wiring of both
stages behind /v1/scan in Phase 5, and these flags mean "loaded into the request
path", not "present in the repository". Reporting stage_1_ready as true while
the endpoint still returns a stub would make the health check lie. Phase 5 flips
the first two, alongside DETECTOR_VERSION in app/config.py.
"""

STAGE_1_READY = False
STAGE_2_READY = False

__all__ = ["STAGE_1_READY", "STAGE_2_READY"]
