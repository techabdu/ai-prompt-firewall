"""Stage 1: the heuristic pre-filter.

Runs before the transformer classifier and on every document, so it has to be
cheap. The whole argument for a two-stage design rests on that: if Stage 1 costs
a meaningful fraction of Stage 2, the architecture has no justification. It uses
the standard library alone and makes no network call.

What it is not
--------------
Heuristics catch what matches a known pattern. An attacker phrasing an
instruction in a way no rule anticipates passes straight through, and no amount
of rule-writing changes that in principle. Stage 1 is not a defence on its own.
Phase 3 measures how far it gets precisely so that the gap Stage 2 has to close
is a measurement rather than an assumption.

Scoring
-------
Hits combine by noisy-OR: ``score = 1 - Π(1 - w)``. Each hit is an independent
piece of evidence, several weak hits accumulate into something meaningful, a
single strong hit stands on its own, and the result cannot leave [0, 1].

Three adjustments apply before combination -- a per-rule hit cap, a weighting-up
of hits found in metadata, and a discount for hits inside quotation marks. Each
is documented at its constant below.
"""

import time
from dataclasses import dataclass, field

from app import config
from app.detection import normalise
from app.detection.rules import (
    ALL_RULES,
    CHARACTER_RULES,
    PHRASE_RULES,
    SYNTHETIC_RULES,
)

#: At most this many hits from any one rule contribute to the score. Without a
#: cap, one lenient rule firing repeatedly saturates the score by itself and the
#: other twelve rules stop affecting the outcome.
MAX_HITS_PER_RULE = 3

#: Hits inside a metadata field are weighted up. A title is a short descriptive
#: label, so an instruction occupying part of one is far more anomalous than the
#: same sentence somewhere in two hundred words of body text.
METADATA_WEIGHT_MULTIPLIER = 1.3

#: Hits inside quotation marks are weighted down. The benign corpus contains
#: security training material that quotes attack strings in order to warn about
#: them, and that is the largest single expected source of false positives. An
#: attacker has little reason to quote their own payload; a document explaining
#: the attack has every reason to quote it.
QUOTATION_WEIGHT_MULTIPLIER = 0.4

#: No single hit may reach certainty. Leaves room for the score to keep rising
#: as further evidence arrives, and keeps one rule from ending the calculation.
MAX_SINGLE_WEIGHT = 0.95

#: A metadata field longer than this is anomalous in itself -- titles and author
#: fields are labels, not paragraphs.
METADATA_LONG_FIELD_WORDS = 15

#: Zero-width characters below this count are treated as incidental: a couple can
#: survive an innocent copy and paste out of a web page.
ZERO_WIDTH_INCIDENTAL_LIMIT = 2
ZERO_WIDTH_INCIDENTAL_WEIGHT = 0.25

_QUOTE_OPEN = "“"
_QUOTE_CLOSE = "”"


@dataclass(frozen=True)
class RuleHit:
    """One piece of evidence, with enough context to diagnose it later.

    A false positive that cannot be explained is only disappointing; one that
    names the rule, the location and the matched text is actionable, and this is
    what populates the ``stage_1`` object in the scan response.
    """

    rule_id: str
    weight: float
    location: str
    variant: str
    excerpt: str
    quoted: bool = False

    def as_dict(self) -> dict:
        return {
            "rule": self.rule_id,
            "weight": round(self.weight, 4),
            "location": self.location,
            "variant": self.variant,
            "excerpt": self.excerpt,
            "quoted": self.quoted,
        }


@dataclass(frozen=True)
class Stage1Result:
    """The pre-filter's verdict on one document."""

    score: float
    triggered: bool
    hits: tuple[RuleHit, ...] = field(default_factory=tuple)
    latency_ms: float = 0.0

    @property
    def rules_fired(self) -> tuple[str, ...]:
        """Rule identifiers that contributed, in descending order of weight."""
        seen: dict[str, float] = {}
        for hit in self.hits:
            seen[hit.rule_id] = max(seen.get(hit.rule_id, 0.0), hit.weight)
        return tuple(sorted(seen, key=lambda rule: -seen[rule]))

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "triggered": self.triggered,
            "latency_ms": round(self.latency_ms, 4),
            "rules_fired": list(self.rules_fired),
            "hits": [hit.as_dict() for hit in self.hits],
        }


# --- helpers ----------------------------------------------------------------


def _excerpt(text: str, start: int, end: int, padding: int = 24) -> str:
    """A short window around a match, for the evidence trail."""
    fragment = text[max(0, start - padding): min(len(text), end + padding)]
    return " ".join(fragment.split())


def _inside_quotes(text: str, position: int) -> bool:
    """Whether a position falls inside a quotation.

    Straight quotes are counted for parity; curly quotes are checked by which of
    the pair appeared most recently. Neither is exact -- an unbalanced quote
    elsewhere in the document throws the count off -- but the discount it feeds
    is a weighting, not a decision, so an occasional misread costs little.
    """
    prefix = text[:position]
    if prefix.count('"') % 2 == 1:
        return True

    last_open = prefix.rfind(_QUOTE_OPEN)
    last_close = prefix.rfind(_QUOTE_CLOSE)
    return last_open > last_close


def _effective_weight(base: float, *, in_metadata: bool, quoted: bool) -> float:
    weight = base
    if in_metadata:
        weight *= METADATA_WEIGHT_MULTIPLIER
    if quoted:
        weight *= QUOTATION_WEIGHT_MULTIPLIER
    return min(weight, MAX_SINGLE_WEIGHT)


# --- rule application -------------------------------------------------------


def _apply_phrase_rules(text: str, location: str, variant: str) -> list[RuleHit]:
    """Run every phrase rule over one view of the document."""
    in_metadata = location != "body"
    hits: list[RuleHit] = []

    for rule in PHRASE_RULES:
        for match in rule.pattern.finditer(text):
            quoted = _inside_quotes(text, match.start())
            hits.append(
                RuleHit(
                    rule_id=rule.id,
                    weight=_effective_weight(rule.weight, in_metadata=in_metadata, quoted=quoted),
                    location=location,
                    variant=variant,
                    excerpt=_excerpt(text, match.start(), match.end()),
                    quoted=quoted,
                )
            )

    return hits


def _apply_character_rules(text: str) -> list[RuleHit]:
    """Run the obfuscation detectors over the raw document.

    Raw only: these rules detect the concealment itself, and normalising first
    would erase the very thing they look for.
    """
    hits: list[RuleHit] = []

    for rule_id, rule in CHARACTER_RULES.items():
        matches = list(rule.pattern.finditer(text))
        if not matches:
            continue

        weight = rule.weight
        if rule_id == "zero_width_chars" and len(matches) <= ZERO_WIDTH_INCIDENTAL_LIMIT:
            # A stray zero-width character is far more likely to be a copy-paste
            # artefact than an attack, so a handful is weak evidence rather than
            # strong.
            weight = ZERO_WIDTH_INCIDENTAL_WEIGHT

        for match in matches[:MAX_HITS_PER_RULE]:
            hits.append(
                RuleHit(
                    rule_id=rule_id,
                    weight=weight,
                    location="body",
                    variant="raw",
                    excerpt=_excerpt(text, match.start(), match.end()),
                )
            )

    return hits


def _apply_metadata_rules(metadata: dict[str, str]) -> list[RuleHit]:
    """Examine metadata fields individually rather than as part of the body."""
    hits: list[RuleHit] = []

    for key, value in metadata.items():
        if not isinstance(value, str) or not value.strip():
            continue

        for variant in normalise.build_variants(value):
            hits.extend(_apply_phrase_rules(variant.text, location=key, variant=variant.name))

        if len(value.split()) > METADATA_LONG_FIELD_WORDS:
            rule = SYNTHETIC_RULES["metadata_anomaly"]
            hits.append(
                RuleHit(
                    rule_id=rule.id,
                    weight=rule.weight,
                    location=key,
                    variant="raw",
                    excerpt=" ".join(value.split())[:80],
                )
            )

    return hits


def _score(hits: list[RuleHit]) -> float:
    """Combine hits by noisy-OR, capping the contribution of any single rule."""
    counted: dict[str, int] = {}
    product = 1.0

    # Strongest hits first, so that the per-rule cap keeps the best evidence
    # rather than whichever happened to be found first.
    for hit in sorted(hits, key=lambda h: -h.weight):
        used = counted.get(hit.rule_id, 0)
        if used >= MAX_HITS_PER_RULE:
            continue
        counted[hit.rule_id] = used + 1
        product *= 1.0 - hit.weight

    return 1.0 - product


# --- public entry point -----------------------------------------------------


def scan(
    text: str,
    metadata: dict[str, str] | None = None,
    threshold: float | None = None,
) -> Stage1Result:
    """Screen one document with the heuristic pre-filter.

    Args:
        text: The document body, as a text extractor delivered it.
        metadata: Optional metadata map, examined field by field.
        threshold: Decision threshold. Defaults to the value in ``app.config``,
            which was selected on the training split alone.

    Returns:
        A score in [0, 1], the binary decision at the threshold, and every hit
        that contributed.
    """
    started = time.perf_counter()
    threshold = config.STAGE1_THRESHOLD if threshold is None else threshold
    metadata = metadata or {}

    hits = _apply_character_rules(text)

    obfuscation_seen = False
    for variant in normalise.build_variants(text):
        variant_hits = _apply_phrase_rules(variant.text, location="body", variant=variant.name)
        hits.extend(variant_hits)
        if variant.derived and variant_hits:
            obfuscation_seen = True

    hits.extend(_apply_metadata_rules(metadata))

    if obfuscation_seen:
        # A phrase matched only once an obfuscation was undone. Concealment and
        # content are two separate facts, so both contribute.
        rule = SYNTHETIC_RULES["obfuscated_instruction"]
        hits.append(
            RuleHit(
                rule_id=rule.id,
                weight=rule.weight,
                location="body",
                variant="derived",
                excerpt="instruction recovered after de-obfuscation",
            )
        )

    score = _score(hits)
    latency_ms = (time.perf_counter() - started) * 1000

    return Stage1Result(
        score=score,
        triggered=score >= threshold,
        hits=tuple(sorted(hits, key=lambda h: -h.weight)),
        latency_ms=latency_ms,
    )


__all__ = ["scan", "Stage1Result", "RuleHit", "ALL_RULES"]
