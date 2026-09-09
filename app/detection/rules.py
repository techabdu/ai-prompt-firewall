"""The Stage 1 rule set.

Thirteen rules. Each is a named pattern with a hand-assigned weight, and the
weights are the detector's only tunable parameters besides the decision
threshold.

They are assigned by judgement, not fitted to the corpus. A weight expresses how
much a match constrains the space of legitimate documents: a Unicode Tag
character carries 0.90 because no ordinary business document contains one, while
a Base64-shaped run carries 0.30 because checksums and identifiers look the
same. Fitting the weights to the data would make Stage 1 a model trained without
admitting it, which is both outside what a heuristic pre-filter is meant to be
and impossible to justify in writing afterwards.

Every rule records what it targets and what it must *not* match, because for
most of these rules the near-miss is the interesting case: the benign corpus was
built to contain exactly the phrasings that a careless version of each rule
would flag.
"""

import re
from dataclasses import dataclass

# --- shared fragments -------------------------------------------------------

# Objects that make an "ignore" instruction an attack rather than ordinary
# English. "Ignore the third column" and "ignore the previous version" are both
# in the benign corpus; what separates the attack is that its object is the
# model's own instructions.
_INSTRUCTION_OBJECT = (
    r"(?:instruction|instructions|directive|directives|prompt|prompts|context|"
    r"guidance|rule|rules|persona|role|restrictions?)"
)

# Terms naming an automated reader. Business documents describe assistants;
# they do not issue orders to them.
_MODEL_TERM = (
    r"(?:AI\s+assistants?|automated\s+(?:readers?|assistants?|systems?|"
    r"summaris\w+|summariz\w+|processing)|language\s+models?|LLMs?|assistants?|"
    r"chatbots?|indexing\s+systems?)"
)

# Obligation language. The pairing with a model term is what carries the signal:
# either half alone is unremarkable.
_OBLIGATION = (
    r"(?:must|should|shall|is\s+required\s+to|are\s+required\s+to|"
    r"is\s+instructed\s+to|are\s+instructed\s+to|needs?\s+to|ha(?:s|ve)\s+to)"
)

# Terms naming the model's own context -- the thing an exfiltration payload is
# after.
# Note on "configuration": the bare word is far too common in technical writing
# to carry any signal -- "no configuration changes are required" is a changelog
# line, not an attack. It is admitted only when qualified as configuration
# values, or when followed by a pronoun ("the configuration you were started
# with"), which is what turns it from a noun about the system into a reference to
# the model's own state.
_CONTEXT_TERM = (
    r"(?:system\s+prompt|instruction\s+block|operating\s+instructions|"
    r"context\s+window|configuration\s+values?|configuration(?=\s+you\b)|"
    r"initial\s+instruction\w*|prior\s+instructions|redaction\s+step|"
    r"configured\s+constraints|"
    r"your\s+(?:instructions|directives|constraints|configuration))"
)

_DISCLOSE_VERB = (
    r"(?:output|reveal|disclos\w+|echo|reproduc\w+|append|includ\w+|print|"
    r"list|provid\w+|return|skip|bypass|record\s+and\s+disclose)"
)

# Up to N intervening words.
def _gap(words: int) -> str:
    return rf"(?:\W+\w+){{0,{words}}}?\W+"


@dataclass(frozen=True)
class Rule:
    """One pattern rule.

    ``targets`` and ``must_not_match`` are documentation rather than behaviour,
    but they are the reason each pattern is shaped the way it is and they are
    what the unit tests are written against.
    """

    id: str
    weight: float
    pattern: re.Pattern
    targets: str
    description: str
    must_not_match: str = ""


FLAGS = re.IGNORECASE | re.VERBOSE | re.DOTALL


# --- phrase rules -----------------------------------------------------------
# Run against every normalisation variant of the document, and separately
# against each metadata field.

PHRASE_RULES: tuple[Rule, ...] = (
    Rule(
        id="override_phrase",
        weight=0.60,
        targets="imperative_override",
        description=(
            "A disregard verb whose object is the model's own instructions, or an "
            "assertion that prior instructions no longer apply."
        ),
        must_not_match=(
            "'ignore the third column', 'please ignore the previous version of this "
            "document', 'disregard any alert raised during the window' -- all ordinary "
            "English, all present in the benign corpus. The object requirement is what "
            "excludes them."
        ),
        pattern=re.compile(
            rf"""
              \b(?:ignore|ignoring|disregard|disregarding|forget|discard|cancel|void)\b
              {_gap(4)}
              \b{_INSTRUCTION_OBJECT}\b
            | \b(?:instructions?|directives?|prompts?|system\s+message|context)\b
              {_gap(5)}
              \b(?:are|is|were|must\s+be)\s+
              (?:void|obsolete|cancelled|canceled|superseded|ignored|disregarded|
                 no\s+longer\s+(?:in\s+effect|appl\w+|valid))\b
            # "replaces" is deliberately absent from the verbs below. A policy memo
            # that replaces last year's guidance is the most ordinary sentence in
            # the corpus, and admitting the verb flagged a run of clean documents.
            # The attack claims precedence over the model's instructions, not over
            # another document, so the object list is narrow here.
            | \b(?:supersed\w+|overrid\w+)\b
              {_gap(4)}
              \b(?:instruction|instructions|directive|directives|prompt|context|
                   system\s+message|system\s+prompt|access\s+policy)\b
            | \bsystem\s+override\b
            | \boverride\s+in\s+effect\b
            """,
            FLAGS,
        ),
    ),
    Rule(
        id="role_reassign",
        weight=0.50,
        targets="imperative_override",
        description="Reassignment of the model's role or persona.",
        must_not_match=(
            "'You must rotate the service credential' -- genuine runbooks address the "
            "reader in the second person constantly, so the modal alone signals nothing. "
            "The rule requires an explicit change of role."
        ),
        pattern=re.compile(
            r"""
              \byou\s+(?:are|must)\s+now\b
            | \byou\s+are\s+no\s+longer\b
            | \badopt\s+the\s+persona\b
            | \bact\s+as\s+(?:an?\s+)?(?:unrestricted|autonomous|different|new)\b
            | \bfrom\s+(?:this\s+point|now)\s+(?:forward|on)\b[^.]{0,60}\brespond\b
            | \bnew\s+directive\b
            """,
            FLAGS,
        ),
    ),
    Rule(
        id="addresses_model",
        weight=0.60,
        targets="disguised_legitimate",
        description=(
            "A term for an automated reader within a short window of an obligation. "
            "The single most discriminating rule in the set: legitimate documents "
            "describe assistants, they do not instruct them."
        ),
        must_not_match=(
            "'an assistant reading it treats the text as a command' and 'when assessing "
            "a supplier's assistant, ask how retrieved content is separated' -- both "
            "name an assistant with no obligation attached, and both are benign."
        ),
        pattern=re.compile(
            rf"\b{_MODEL_TERM}\b{_gap(6)}\b{_OBLIGATION}\b",
            FLAGS,
        ),
    ),
    Rule(
        id="prompt_exfiltration",
        weight=0.60,
        targets="disguised_legitimate",
        description=(
            "A disclosure verb paired with a term for the model's own context, in "
            "either order."
        ),
        must_not_match=(
            "'we use a system prompt is not an answer to that question' -- security "
            "training material names these terms without asking for them. The verb "
            "pairing is what makes it a request."
        ),
        pattern=re.compile(
            rf"""
              \b{_DISCLOSE_VERB}\b{_gap(10)}\b{_CONTEXT_TERM}\b
            | \b{_CONTEXT_TERM}\b{_gap(6)}\b{_DISCLOSE_VERB}\b
            """,
            FLAGS,
        ),
    ),
    Rule(
        id="conditioned_trigger",
        weight=0.50,
        targets="conditioned_delayed",
        description=(
            "A trigger clause arming a future response, followed by a directive verb."
        ),
        must_not_match=(
            "'if your question is not answered here, raise a ticket' and 'escalate if "
            "the queue has not drained' -- conditionals are everywhere in operational "
            "prose. The trigger has to be a future request, not a future state."
        ),
        pattern=re.compile(
            r"""
            (?:
              \bwhen(?:ever)?\s+(?:the\s+|a\s+)?
                (?:user|reader|anyone|someone|question|conversation|summaris\w+|summariz\w+|
                   this\s+(?:page|document|article))
            | \bif\s+(?:the\s+|a\s+)?
                (?:user|reader|anyone|someone|summary|question|asked|this\s+document)
            | \bshould\s+(?:anyone|the\s+conversation|a\s+user|the\s+reader)
            | \bthe\s+next\s+time\s+you\s+are\s+asked
            | \bon\s+the\s+next\s+request
            )
            [^.]{0,90}?
            \b(?:respond|repl(?:y|ies)|answer|state|say|omit|confirm|provid\w+|
                 append|recommend|output|report|includ\w+|name)\b
            """,
            FLAGS,
        ),
    ),
    Rule(
        id="caps_directive",
        weight=0.25,
        targets="structural",
        description=(
            "An all-capitals directive header. Weak on its own -- legitimate documents "
            "shout occasionally -- but it rarely appears alone in an injected document."
        ),
        must_not_match="An ordinary acronym, which is why a trailing colon or dash is required.",
        pattern=re.compile(
            r"""
              \b(?:IMPORTANT|ATTENTION|NOTICE|WARNING|OVERRIDE|SYSTEM|URGENT|ALERT)\b
              (?:\s+[A-Z]{2,})*\s*[:\-—]
            | \b[A-Z]{3,}\s+[A-Z]{3,}\b\s*[:\-—]
            """,
            re.VERBOSE,
        ),
    ),
)


# --- character and structural rules -----------------------------------------
# Run against the raw document only: these detect the obfuscation itself, and
# undoing it first would erase the evidence.

CHARACTER_RULES: dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule(
            id="unicode_tag_chars",
            weight=0.90,
            targets="hidden_text",
            description=(
                "Any character from the Unicode Tags block. Close to conclusive: these "
                "render as nothing and have no legitimate use in a business document."
            ),
            pattern=re.compile(r"[\U000E0000-\U000E007F]"),
        ),
        Rule(
            id="zero_width_chars",
            weight=0.65,
            targets="encoded_payload",
            description=(
                "Zero-width characters. Weighted by count, because one or two can "
                "survive an innocent copy and paste while a dozen cannot."
            ),
            pattern=re.compile(r"[​‌‍⁠﻿]"),
        ),
        Rule(
            id="invisible_styling",
            weight=0.85,
            targets="hidden_text",
            description=(
                "An inline style attribute that would render text invisible. Colour "
                "does not survive text extraction, so this is what a colour-matched "
                "payload looks like once an extractor has leaked the attribute."
            ),
            pattern=re.compile(
                r"""
                style\s*=\s*["'][^"']*?
                (?: color\s*:\s*(?:\#(?:fff|ffffff)\b
                                 |rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)
                                 |transparent|white)
                  | display\s*:\s*none
                  | font-size\s*:\s*[01](?:\.\d+)?\s*px
                )
                """,
                FLAGS,
            ),
        ),
        Rule(
            id="base64_blob",
            weight=0.30,
            targets="encoded_payload",
            description=(
                "A standalone Base64-shaped run. Deliberately low: checksums and "
                "identifiers look identical. The weight that matters is carried by "
                "obfuscated_instruction, which fires only if the blob decodes to an "
                "instruction."
            ),
            pattern=re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])"),
        ),
        Rule(
            id="char_spaced_run",
            weight=0.60,
            targets="encoded_payload",
            description=(
                "Eight or more consecutive single-character tokens. Effectively absent "
                "from business prose."
            ),
            pattern=re.compile(r"(?<!\S)\S(?:[ \t]+\S){7,}(?!\S)"),
        ),
    )
}


# --- rules with no pattern of their own -------------------------------------
# Emitted by the detector rather than matched directly.

SYNTHETIC_RULES: dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule(
            id="obfuscated_instruction",
            weight=0.85,
            targets="encoded_payload, hidden_text",
            description=(
                "A phrase rule matched only after an obfuscation was undone. Strong "
                "evidence: the payload was present and someone took trouble to hide it. "
                "Fires alongside the phrase rule, which is intended rather than double "
                "counting -- concealment and content are two separate facts."
            ),
            pattern=re.compile(r"(?!)"),  # never matches directly
        ),
        Rule(
            id="metadata_anomaly",
            weight=0.55,
            targets="metadata_payload",
            description=(
                "A metadata field carrying an instruction, or one far longer than such "
                "fields normally run. A title is a short descriptive label; an "
                "imperative occupying part of one is anomalous in a way the same "
                "sentence buried in body text is not."
            ),
            pattern=re.compile(r"(?!)"),
        ),
    )
}


ALL_RULES: dict[str, Rule] = {
    **{rule.id: rule for rule in PHRASE_RULES},
    **CHARACTER_RULES,
    **SYNTHETIC_RULES,
}
