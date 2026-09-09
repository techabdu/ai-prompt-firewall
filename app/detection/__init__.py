"""Detection stages.

Deliberately empty of logic in Phase 1. The two modules that will live here
are:

    stage1.py  -- heuristic pre-filter: regex, pattern matching, structural
                  scoring. Built in Phase 3.
    stage2.py  -- CPU-only transformer classifier, loading the model artifact
                  fine-tuned off-repo on Colab. Built in Phase 5.

The readiness flags below are the single place the rest of the service asks
whether a stage exists. Phase 3 flips the first, Phase 5 the second; nothing
else needs to change for the health check to start telling the truth.
"""

STAGE_1_READY = False
STAGE_2_READY = False

__all__ = ["STAGE_1_READY", "STAGE_2_READY"]
