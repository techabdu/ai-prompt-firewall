"""Composition of the labelled corpus from the authored pools.

Deterministic: the same seed produces a byte-identical corpus. That is the
reason this is a seeded generator rather than a set of model-generated
documents. The methodology chapter can state that the dataset is rebuildable
from the repository, and a reader can verify it by running the build. A model
call cannot offer that, and the cost -- diversity has to come from the size of
the authored pools rather than from the generator -- is a fair trade at this
scale.
"""

import base64
import hashlib
import random
import re
from dataclasses import asdict, dataclass, field

from datagen import benign as benign_pools
from datagen import payloads as payload_pools
from datagen.scaffolds import HELD_OUT_SCAFFOLDS, MAIN_SCAFFOLDS, Scaffold
from datagen.slots import SLOT_POOLS

SCHEMA_VERSION = 1

#: Target band for a document body, in words. The hard cap is 350; documents
#: aim well below it because the encoded and hidden categories add tokens out of
#: all proportion to the words they add, and the budget that matters is the
#: 512-token one.
#:
#: Each document draws its own target from this band, so lengths vary rather
#: than every document stopping at the same threshold. Length that correlates
#: with the label would be a feature the classifier could learn instead of the
#: attack, which is why the band is shared by both classes.
TARGET_MIN_WORDS = 150
TARGET_MAX_WORDS = 250
MAX_WORDS_TARGET = 270
HARD_WORD_CAP = 350

#: Prose budget for documents carrying a concealed payload, in words.
#:
#: Deliberately well below the general budget. Unicode Tag characters are
#: outside the WordPiece vocabulary, so each one costs a token: a 60-character
#: hidden instruction spends 60 tokens while adding a single "word" to the
#: count. This is precisely the mismatch the project reference warns about, and
#: without a separate budget these documents sit closest to the ceiling of any
#: category in the corpus.
CONCEALED_PROSE_BUDGET = 190

#: Documents per technique. Sums to 400. Two techniques carry the remainder
#: from dividing 400 by six; they are the two with the widest phrasing
#: variation, so the surplus buys the most coverage there.
TECHNIQUE_COUNTS: dict[str, int] = {
    "imperative_override": 68,
    "disguised_legitimate": 68,
    "hidden_text": 66,
    "encoded_payload": 66,
    "metadata_payload": 66,
    "conditioned_delayed": 66,
}

#: Documents per benign class. Sums to 400.
BENIGN_COUNTS: dict[str, int] = {
    "ordinary": 100,
    "discusses_injection": 100,
    "lexical_decoy": 100,
    "imperative_benign": 100,
}

#: Challenge-set counts, built only from held-out scaffolds.
CHALLENGE_TECHNIQUE_COUNT = 5  # per technique, 6 techniques -> 30
CHALLENGE_BENIGN_COUNTS: dict[str, int] = {
    "ordinary": 8,
    "discusses_injection": 7,
    "lexical_decoy": 8,
    "imperative_benign": 7,
}

INSERT_POSITIONS = ("opening", "middle", "closing")

_SLOT_RE = re.compile(r"\{(\w+)\}")


@dataclass
class Record:
    """One labelled document. Field order here is the field order on disk."""

    id: str
    text: str
    label: int
    technique: str | None
    technique_note: str | None
    benign_class: str | None
    metadata: dict[str, str] = field(default_factory=dict)
    payload_location: str | None = None
    payload_visibility: str | None = None
    scaffold: str = ""
    word_count: int = 0
    token_count: int | None = None
    split: str = ""
    source: str = "synthetic"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


# --- slot filling -----------------------------------------------------------


def fill_slots(template: str, rng: random.Random) -> str:
    """Fill ``{slot}`` placeholders, drawing each occurrence independently.

    Independently matters: a template reading "Present: {person}, {person}" must
    not name the same person twice. Repeated slots therefore avoid values
    already used for that slot within the same template.

    A placeholder with no matching pool is left in place rather than raising.
    ``{blob}`` in an encoded payload's framing is exactly that case: it is
    filled later with the encoded instruction, not from a slot pool.
    """
    used: dict[str, set[str]] = {}

    def pick(match: re.Match) -> str:
        name = match.group(1)
        pool = SLOT_POOLS.get(name)
        if pool is None:
            return match.group(0)
        seen = used.setdefault(name, set())
        for _ in range(8):
            value = rng.choice(pool)
            if value not in seen:
                seen.add(value)
                return value
        return rng.choice(pool)

    return _SLOT_RE.sub(pick, template)


# --- concealment and encoding ----------------------------------------------


def encode_unicode_tags(text: str) -> str:
    """Map ASCII to the Unicode Tags block (U+E0000-U+E007F).

    The documented "ASCII smuggling" technique. These characters survive text
    extraction and are invisible in essentially every renderer, so a reader sees
    nothing and the model sees the instruction.
    """
    return "".join(chr(0xE0000 + ord(char)) if ord(char) < 0x80 else char for char in text)


def encode_zero_width(text: str) -> str:
    """Insert a zero-width space inside each word, leaving spacing intact.

    This is an *obfuscation*, not a concealment, which is why it belongs with
    the encoded payloads rather than the hidden ones. A zero-width space has no
    width, so the text renders exactly as written and a human reads it normally
    -- but "ignore" is no longer the string "ignore" to anything matching on
    whole words.

    An earlier version replaced the spaces between words instead. That was
    wrong: with the real spaces gone the payload renders as one run-together
    blob, which is conspicuous rather than hidden, and it defeated the purpose
    of the technique twice over.
    """
    return " ".join(
        word[0] + "​" + word[1:] if len(word) > 1 else word for word in text.split()
    )


_MARKUP_TEMPLATES = (
    '<span style="color:#ffffff;background-color:#ffffff">{text}</span>',
    '<div style="color:rgb(255,255,255);font-size:1px">{text}</div>',
    '<p style="color:#fff;display:none">{text}</p>',
    '<span style="color:transparent">{text}</span>',
)


def encode_hidden_markup(text: str, rng: random.Random) -> str:
    """Wrap text in inline style markup that would render it invisible.

    Colour is a rendering property and does not survive text extraction: by the
    time a document reaches this service, white-on-white text is simply text.
    What this models is the common case where an extractor preserves the style
    attribute, leaving the concealment visible in the extracted string. Where an
    extractor strips styling entirely, the technique is undetectable by this
    system -- a limitation recorded in the dataset card rather than papered over.
    """
    return rng.choice(_MARKUP_TEMPLATES).format(text=text)


def encode_spaced(text: str) -> str:
    """Separate every character with a space, defeating whole-word matching.

    Word boundaries are kept as a wider gap rather than collapsed away. That is
    both the documented form of the technique and the harder case for a
    detector: collapsing the spaces would leave one unbroken run that any
    length heuristic would flag, whereas preserving the gaps leaves something
    that still reads as spaced-out text to a human.
    """
    return "   ".join(" ".join(word) for word in text.split())


def encode_base64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


#: Obfuscation scheme name to its encoder. Keyed by the ``scheme`` field on an
#: ``EncodedPayload``.
_ENCODERS = {
    "base64": encode_base64,
    "spaced": encode_spaced,
    "zwsp": encode_zero_width,
}


# --- body composition -------------------------------------------------------


def _compose_paragraphs(
    scaffold: Scaffold, rng: random.Random, word_budget: int
) -> list[str]:
    """Pick and fill scaffold paragraphs until the body reaches its own target.

    Each document draws an individual target from the band rather than all
    stopping at one threshold, so document length varies the way real documents
    do. The target is capped by ``word_budget``, which is what remains after the
    payload or benign insert has taken its share.
    """
    target = min(word_budget, rng.randint(TARGET_MIN_WORDS, TARGET_MAX_WORDS))

    order = list(range(len(scaffold.paragraphs)))
    rng.shuffle(order)

    chosen: list[str] = []
    words = 0
    for index in order:
        paragraph = fill_slots(scaffold.paragraphs[index], rng)
        length = len(paragraph.split())
        if chosen and words + length > word_budget:
            continue
        chosen.append(paragraph)
        words += length
        if words >= target and len(chosen) >= 3:
            break

    return chosen


def _insert_at(paragraphs: list[str], insert: str, position: str) -> list[str]:
    """Place an inserted paragraph at the requested position."""
    result = list(paragraphs)
    if position == "opening":
        result.insert(0, insert)
    elif position == "closing":
        result.append(insert)
    else:
        result.insert(max(1, len(result) // 2), insert)
    return result


def _build_metadata(scaffold: Scaffold, rng: random.Random) -> dict[str, str]:
    metadata = {
        "title": fill_slots(rng.choice(scaffold.titles), rng),
        "author": fill_slots(rng.choice(scaffold.authors), rng),
    }
    alt = fill_slots(rng.choice(scaffold.alt_texts), rng) if scaffold.alt_texts else ""
    if alt:
        metadata["alt_text"] = alt
    return metadata


# --- record builders --------------------------------------------------------


def _build_malicious(
    record_id: str, technique: str, scaffold: Scaffold, rng: random.Random
) -> Record:
    """Build one injected document using the named technique."""
    metadata = _build_metadata(scaffold, rng)
    position = rng.choice(INSERT_POSITIONS)
    visibility = "plain"
    location = "body"

    if technique == "metadata_payload":
        payload = rng.choice(payload_pools.METADATA_PAYLOADS)
        # The body stays entirely clean: that is the whole point of the
        # technique, and a body carrying its own payload would make the record
        # a hybrid rather than a clean example of this class.
        paragraphs = _compose_paragraphs(scaffold, rng, MAX_WORDS_TARGET)
        metadata[payload.field] = fill_slots(payload.value, rng)
        location = "metadata"
        note = payload.note

    elif technique == "hidden_text":
        hidden = rng.choice(payload_pools.HIDDEN_TEXT)
        # Two concealment forms, and only two, because only these two actually
        # hide anything from a reader. Unicode Tag characters render as nothing
        # at all, and leaked style markup carries a colour that would have made
        # the text invisible before extraction flattened it.
        visibility = rng.choice(("zero_width", "hidden_markup"))
        concealed = (
            encode_hidden_markup(hidden.instruction, rng)
            if visibility == "hidden_markup"
            else encode_unicode_tags(hidden.instruction)
        )
        # Hidden text hides *within* visible prose rather than standing alone as
        # its own paragraph, so it is appended inline to a real paragraph.
        paragraphs = _compose_paragraphs(scaffold, rng, CONCEALED_PROSE_BUDGET)
        target = 0 if position == "opening" else (-1 if position == "closing" else len(paragraphs) // 2)
        paragraphs[target] = f"{paragraphs[target]} {concealed}"
        note = hidden.note

    elif technique == "encoded_payload":
        encoded = rng.choice(payload_pools.ENCODED_PAYLOADS)
        blob = _ENCODERS[encoded.scheme](encoded.instruction)
        insert = fill_slots(encoded.framing, rng).replace("{blob}", blob)
        budget = MAX_WORDS_TARGET - len(insert.split())
        paragraphs = _insert_at(_compose_paragraphs(scaffold, rng, budget), insert, position)
        note = encoded.note

    else:
        payload = rng.choice(payload_pools.TECHNIQUES[technique])
        insert = fill_slots(payload.text, rng)
        budget = MAX_WORDS_TARGET - len(insert.split())
        paragraphs = _insert_at(_compose_paragraphs(scaffold, rng, budget), insert, position)
        note = payload.note

    text = "\n\n".join(paragraphs)
    return Record(
        id=record_id,
        text=text,
        label=1,
        technique=technique,
        technique_note=note,
        benign_class=None,
        metadata=metadata,
        payload_location=location,
        payload_visibility=visibility,
        scaffold=scaffold.key,
        word_count=len(text.split()),
    )


def _build_benign(
    record_id: str, benign_class: str, scaffold: Scaffold, rng: random.Random
) -> Record:
    """Build one clean document of the named benign class."""
    metadata = _build_metadata(scaffold, rng)
    pool = benign_pools.BENIGN_CLASSES[benign_class]

    if pool:
        insert = fill_slots(rng.choice(pool).text, rng)
        budget = MAX_WORDS_TARGET - len(insert.split())
        paragraphs = _insert_at(
            _compose_paragraphs(scaffold, rng, budget), insert, rng.choice(INSERT_POSITIONS)
        )
    else:
        paragraphs = _compose_paragraphs(scaffold, rng, MAX_WORDS_TARGET)

    text = "\n\n".join(paragraphs)
    return Record(
        id=record_id,
        text=text,
        label=0,
        technique=None,
        technique_note=None,
        benign_class=benign_class,
        metadata=metadata,
        payload_location=None,
        payload_visibility=None,
        scaffold=scaffold.key,
        word_count=len(text.split()),
    )


# --- corpus assembly --------------------------------------------------------


def _generate_unique(builder, seen: set[str], rng: random.Random, attempts: int = 40) -> Record:
    """Build a record, retrying until its body is one not already in the corpus.

    Duplicate bodies would inflate the corpus without adding information, and
    would also defeat the cross-split leakage test by putting genuinely
    identical text in two splits. Retries are deterministic because the RNG is
    seeded and consumed in order.
    """
    for _ in range(attempts):
        record = builder(rng)
        if record.text not in seen:
            seen.add(record.text)
            return record
    raise RuntimeError(
        "Exhausted attempts to generate a unique document. The authored pools are "
        "too small for the requested corpus size."
    )


def generate_corpus(seed: int) -> tuple[list[Record], list[Record]]:
    """Generate the main corpus and the challenge set.

    Returns:
        ``(main, challenge)``. Splits are not yet assigned; see ``assign_splits``.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    main: list[Record] = []

    # Malicious. Scaffolds are cycled so that every scaffold carries roughly the
    # same number of injected documents, and -- with the benign pass below --
    # appears under both labels. Without that parity a classifier could learn
    # the template instead of the attack.
    index = 0
    for technique, count in TECHNIQUE_COUNTS.items():
        for _ in range(count):
            scaffold = MAIN_SCAFFOLDS[index % len(MAIN_SCAFFOLDS)]
            record_id = f"mal-{index + 1:04d}"
            main.append(
                _generate_unique(
                    lambda r, t=technique, s=scaffold, i=record_id: _build_malicious(i, t, s, r),
                    seen,
                    rng,
                )
            )
            index += 1

    # Benign, over the same scaffold cycle.
    index = 0
    for benign_class, count in BENIGN_COUNTS.items():
        for _ in range(count):
            scaffold = MAIN_SCAFFOLDS[index % len(MAIN_SCAFFOLDS)]
            record_id = f"ben-{index + 1:04d}"
            main.append(
                _generate_unique(
                    lambda r, b=benign_class, s=scaffold, i=record_id: _build_benign(i, b, s, r),
                    seen,
                    rng,
                )
            )
            index += 1

    # Challenge set: held-out scaffolds only, so it measures generalisation to
    # document types the model has never seen rather than recall on familiar
    # ones.
    challenge: list[Record] = []
    index = 0
    for technique in TECHNIQUE_COUNTS:
        for _ in range(CHALLENGE_TECHNIQUE_COUNT):
            scaffold = HELD_OUT_SCAFFOLDS[index % len(HELD_OUT_SCAFFOLDS)]
            record_id = f"cmal-{index + 1:04d}"
            record = _generate_unique(
                lambda r, t=technique, s=scaffold, i=record_id: _build_malicious(i, t, s, r),
                seen,
                rng,
            )
            record.split = "challenge"
            challenge.append(record)
            index += 1

    index = 0
    for benign_class, count in CHALLENGE_BENIGN_COUNTS.items():
        for _ in range(count):
            scaffold = HELD_OUT_SCAFFOLDS[index % len(HELD_OUT_SCAFFOLDS)]
            record_id = f"cben-{index + 1:04d}"
            record = _generate_unique(
                lambda r, b=benign_class, s=scaffold, i=record_id: _build_benign(i, b, s, r),
                seen,
                rng,
            )
            record.split = "challenge"
            challenge.append(record)
            index += 1

    return main, challenge


def assign_splits(records: list[Record], seed: int) -> None:
    """Assign train, validation and test splits in place.

    Stratified by label and by technique or benign class, so no split is missing
    a category and each holds the same 70/15/15 proportion of every one. Order
    within a stratum comes from a hash of the record identifier and the seed,
    which makes the assignment reproducible without depending on the order the
    records happened to be generated in.
    """
    strata: dict[tuple, list[Record]] = {}
    for record in records:
        key = (record.label, record.technique or record.benign_class)
        strata.setdefault(key, []).append(record)

    for key in sorted(strata, key=lambda k: (k[0], str(k[1]))):
        group = sorted(
            strata[key],
            key=lambda r: hashlib.sha256(f"{seed}:{r.id}".encode()).hexdigest(),
        )
        total = len(group)
        n_train = round(total * 0.70)
        n_val = round(total * 0.15)
        for position, record in enumerate(group):
            if position < n_train:
                record.split = "train"
            elif position < n_train + n_val:
                record.split = "val"
            else:
                record.split = "test"
