"""A catalog of well-known chord progressions — Pop, '50s, jazz ii-V-I, the
12-bar blues, the Andalusian cadence, and more — expressed generically as a
tonic-relative interval + quality per chord so any of them can be realized
in any key, plus a way to check whether a real chord sequence matches one.

Each progression is defined by hand from real music theory (not derived
from a single scale's diatonic chords), since several well-known
progressions genuinely mix scale contexts — e.g. the Andalusian cadence's
closing V is a borrowed/altered dominant, not natural minor's own
(diatonic, minor-quality) v.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .chords import ChordSegment, _merge_adjacent, parse_chord_label
from .theory import (
    NOTE_NAME_TO_PITCH_CLASS,
    _CHORD_QUALITY_NUMERAL_DECORATION,
    _roman_numeral_for_interval,
    _root_name_for_interval,
    analyze_chord_function,
    build_chord,
)


@dataclass
class ProgressionStep:
    interval: int  # semitones above the progression's tonic
    quality: str  # a CHORD_INTERVALS key, e.g. "major", "minor7"


@dataclass
class NamedProgression:
    name: str
    description: str
    mode: str  # "major" or "minor" — context for roman-numeral spelling
    steps: list[ProgressionStep] = field(default_factory=list)


def _step(interval: int, quality: str) -> ProgressionStep:
    return ProgressionStep(interval=interval, quality=quality)


NAMED_PROGRESSIONS: dict[str, NamedProgression] = {
    "pop": NamedProgression(
        name="Pop progression",
        description="I-V-vi-IV — one of the most common progressions in popular music.",
        mode="major",
        steps=[_step(0, "major"), _step(7, "major"), _step(9, "minor"), _step(5, "major")],
    ),
    "fifties": NamedProgression(
        name="'50s progression",
        description="I-vi-IV-V — the doo-wop progression.",
        mode="major",
        steps=[_step(0, "major"), _step(9, "minor"), _step(5, "major"), _step(7, "major")],
    ),
    "three_chord": NamedProgression(
        name="Three-chord (I-IV-V)",
        description="I-IV-V — the basic folk/rock/country progression.",
        mode="major",
        steps=[_step(0, "major"), _step(5, "major"), _step(7, "major")],
    ),
    "jazz_ii_v_i": NamedProgression(
        name="ii-V-I turnaround",
        description="ii7-V7-Imaj7 — the fundamental jazz cadence.",
        mode="major",
        steps=[_step(2, "minor7"), _step(7, "dominant7"), _step(0, "major7")],
    ),
    "pachelbels_canon": NamedProgression(
        name="Pachelbel's Canon",
        description="I-V-vi-iii-IV-I-IV-V — the progression from Canon in D.",
        mode="major",
        steps=[
            _step(0, "major"), _step(7, "major"), _step(9, "minor"), _step(4, "minor"),
            _step(5, "major"), _step(0, "major"), _step(5, "major"), _step(7, "major"),
        ],
    ),
    "authentic_cadence": NamedProgression(
        name="Authentic cadence",
        description="V-I — the strongest, most conclusive cadence in tonal music.",
        mode="major",
        steps=[_step(7, "major"), _step(0, "major")],
    ),
    "plagal_cadence": NamedProgression(
        name="Plagal cadence",
        description='IV-I — the "Amen" cadence.',
        mode="major",
        steps=[_step(5, "major"), _step(0, "major")],
    ),
    "deceptive_cadence": NamedProgression(
        name="Deceptive cadence",
        description="V-vi — a cadence that resolves the dominant somewhere other than the tonic.",
        mode="major",
        steps=[_step(7, "major"), _step(9, "minor")],
    ),
    "twelve_bar_blues": NamedProgression(
        name="12-bar blues",
        description="I7-IV7-V7 over 12 bars — every chord a dominant 7th, including the tonic.",
        mode="major",
        steps=[
            _step(0, "dominant7"), _step(0, "dominant7"), _step(0, "dominant7"), _step(0, "dominant7"),
            _step(5, "dominant7"), _step(5, "dominant7"), _step(0, "dominant7"), _step(0, "dominant7"),
            _step(7, "dominant7"), _step(5, "dominant7"), _step(0, "dominant7"), _step(7, "dominant7"),
        ],
    ),
    "minor_three_chord": NamedProgression(
        name="Minor three-chord (i-iv-v)",
        description="i-iv-v — the natural-minor three-chord progression (a minor-quality v, unlike a minor blues turnaround).",
        mode="minor",
        steps=[_step(0, "minor"), _step(5, "minor"), _step(7, "minor")],
    ),
    "minor_pop": NamedProgression(
        name="Minor-key pop progression",
        description="i-VI-III-VII — a common minor-key rock/pop progression.",
        mode="minor",
        steps=[_step(0, "minor"), _step(8, "major"), _step(3, "major"), _step(10, "major")],
    ),
    "andalusian_cadence": NamedProgression(
        name="Andalusian cadence",
        description="i-VII-VI-V — VII and VI are natural minor's own; the closing V is a borrowed/altered major dominant, not natural minor's own minor v.",
        mode="minor",
        steps=[_step(0, "minor"), _step(10, "major"), _step(8, "major"), _step(7, "major")],
    ),
}


@dataclass
class ProgressionChord:
    roman_numeral: str  # e.g. "I", "vi", "bVII", "ii7"
    root: str  # note name, e.g. "G"
    quality: str
    notes: list[str]
    pitch_classes: list[int]


def realize_progression(progression: NamedProgression, tonic: str) -> list[ProgressionChord]:
    """Turn a `NamedProgression` into the real chords for a specific tonic
    — e.g. `realize_progression(NAMED_PROGRESSIONS["pop"], "G")` gives
    G-D-Em-C, the same I-V-vi-IV shape realized in G major.
    """
    chords = []
    for step in progression.steps:
        root_name = _root_name_for_interval(tonic, progression.mode, step.interval)
        base_numeral = _roman_numeral_for_interval(step.interval, progression.mode)
        numeral = _CHORD_QUALITY_NUMERAL_DECORATION[step.quality](base_numeral)
        chord = build_chord(root_name, step.quality)
        chords.append(
            ProgressionChord(
                roman_numeral=numeral,
                root=chord.root,
                quality=chord.quality,
                notes=chord.notes,
                pitch_classes=chord.pitch_classes,
            )
        )
    return chords


def identify_progression(chords: list[tuple[str, str]], key_tonic: str, key_mode: str) -> list[str]:
    """Given a real chord sequence (root, quality) pairs — e.g. straight
    from `detect_chords` — and the key it's in, return the names (keys of
    `NAMED_PROGRESSIONS`) of every catalog progression whose roman-numeral
    sequence matches it exactly.
    """
    numerals = [analyze_chord_function(root, quality, key_tonic, key_mode).roman_numeral for root, quality in chords]

    matches = []
    for name, progression in NAMED_PROGRESSIONS.items():
        expected = [
            _CHORD_QUALITY_NUMERAL_DECORATION[step.quality](_roman_numeral_for_interval(step.interval, progression.mode))
            for step in progression.steps
        ]
        if numerals == expected:
            matches.append(name)
    return matches


# Only major/minor triad qualities can ever come out of detect_chords (see
# chords.py — real audio chord detection there is triad-only), so a
# progression step's own quality (which can be a seventh chord, e.g.
# NAMED_PROGRESSIONS["jazz_ii_v_i"]'s "minor7"/"dominant7"/"major7") has to
# collapse down to whichever triad it's built on before it can ever become
# a corrected ChordSegment's label.
_TRIAD_QUALITY_FOR_STEP = {
    "major": "major",
    "minor": "minor",
    "major7": "major",
    "minor7": "minor",
    "dominant7": "major",
}

_MIN_PROGRESSION_MATCH_RATIO = 0.5  # below this, nothing in the catalog explains the sequence well enough to trust
DEFAULT_CONFIDENCE_THRESHOLD = 0.05  # same "genuine ambiguity" cutoff as ChordSegment.confidence's own convention


def _triad_label(root: str, quality: str) -> str:
    triad_quality = _TRIAD_QUALITY_FOR_STEP.get(quality, "major")
    return root if triad_quality == "major" else f"{root}m"


def _tiled_expected_steps(progression: NamedProgression, phase: int, length: int) -> list[ProgressionStep]:
    steps = progression.steps
    return [steps[(phase + i) % len(steps)] for i in range(length)]


@dataclass
class ProgressionMatch:
    name: str  # a NAMED_PROGRESSIONS key
    phase: int  # which step of the progression's own cycle the sequence starts on
    match_ratio: float  # fraction of positions where the tiled progression already agrees with the real sequence


def best_progression_match(chords: list[tuple[str, str]], key_tonic: str, key_mode: str) -> ProgressionMatch | None:
    """Find the catalog progression that best explains a real, possibly
    noisy chord sequence — a fuzzier cousin of `identify_progression`'s
    exact-match check.

    A real song usually loops a short progression many times over its
    full length (and might start recording mid-loop), so each candidate
    is tiled cyclically to the sequence's length at every possible
    starting phase, and scored by what fraction of positions agree — the
    best (progression, phase) pair overall wins, even when a few
    positions disagree (real misreads, or a genuine one-off deviation
    like a bridge or a passing chord) rather than the loop itself
    changing.

    Matching compares each position's (root interval above the tonic,
    triad quality) rather than a fully-decorated roman numeral string —
    `detect_chords` only ever produces plain major/minor triads (see
    chords.py), so a catalog progression built from seventh chords (e.g.
    "twelve_bar_blues") would otherwise never match anything: "V7" and
    "V" are different strings even though a real dominant-seventh chord
    and its detected triad share the same root and are both "major"
    underneath (see _TRIAD_QUALITY_FOR_STEP). `key_mode` only limits the
    search to progressions written for that mode (e.g. a minor-key song
    is never explained by a major-mode catalog entry, even if the raw
    interval/quality pairs happen to line up) — it doesn't otherwise
    affect scoring, since interval and triad quality are mode-independent
    facts about each chord.

    Returns None if `chords` is empty, or if even the best match agrees
    with fewer than half the sequence — too weak a fit to be worth using
    to correct anything.
    """
    if not chords:
        return None

    key_tonic_pc = NOTE_NAME_TO_PITCH_CLASS[key_tonic]
    actual = [((NOTE_NAME_TO_PITCH_CLASS[root] - key_tonic_pc) % 12, quality) for root, quality in chords]

    best: ProgressionMatch | None = None
    for name, progression in NAMED_PROGRESSIONS.items():
        if progression.mode != key_mode:
            continue
        for phase in range(len(progression.steps)):
            tiled_steps = _tiled_expected_steps(progression, phase, len(actual))
            match_ratio = sum(
                1
                for (interval, quality), step in zip(actual, tiled_steps)
                if interval == step.interval and quality == _TRIAD_QUALITY_FOR_STEP.get(step.quality)
            ) / len(actual)
            if best is None or match_ratio > best.match_ratio:
                best = ProgressionMatch(name=name, phase=phase, match_ratio=match_ratio)

    if best is not None and best.match_ratio < _MIN_PROGRESSION_MATCH_RATIO:
        return None
    return best


DEFAULT_WINDOW_SIZE = 16  # chord segments per local-matching chunk — see clean_chords_with_progression


@dataclass
class ProgressionCleanupResult:
    chords: list[ChordSegment]
    matches: list[ProgressionMatch]  # one per chunk that actually got a correction applied from it; empty if nothing did


def clean_chords_with_progression(
    chords: list[ChordSegment],
    key_tonic: str,
    key_mode: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> ProgressionCleanupResult:
    """Correct low-confidence chord guesses using the closest-matching
    catalog progression, instead of displaying every shaky raw reading at
    face value.

    A real song is sectional — a verse and a chorus often loop different
    progressions — so this doesn't fit one progression to the entire
    sequence. Instead it walks the sequence in fixed-size, non-overlapping
    chunks of `window_size` segments, finds each chunk's own best match
    independently (see `best_progression_match`), and — within that chunk
    only — overwrites the segments whose confidence is at or below
    `confidence_threshold` *and* whose chord disagrees with what the
    chunk's match expects at that position. A segment the detector was
    already confident about is left alone even if it doesn't fit the
    pattern — real songs do genuinely deviate from a textbook progression
    sometimes (a bridge, a passing chord, a key change), and a confident
    real reading shouldn't be overruled by a generic template just
    because it's unusual. A trailing partial chunk shorter than 3
    segments is left as detected — too little evidence to fit anything
    meaningfully.

    A corrected segment keeps its original `correlation`/`confidence` —
    those still honestly describe how ambiguous the audio itself was —
    with only `chord` changed and `progression_corrected` set, so a
    caller can tell it apart from a directly-detected one. Adjacent
    segments that end up sharing a chord after correction (within a
    chunk, or across a chunk boundary) are merged back together, same as
    `detect_chords` does for its own raw output.
    """
    if len(chords) < 3:
        return ProgressionCleanupResult(chords=chords, matches=[])

    cleaned = list(chords)
    matches: list[ProgressionMatch] = []

    for chunk_start in range(0, len(chords), window_size):
        chunk = chords[chunk_start : chunk_start + window_size]
        if len(chunk) < 3:
            continue

        root_quality_pairs = [parse_chord_label(seg.chord) for seg in chunk]
        match = best_progression_match(root_quality_pairs, key_tonic, key_mode)
        if match is None:
            continue

        progression = NAMED_PROGRESSIONS[match.name]
        realized = realize_progression(progression, key_tonic)
        expected_labels = [
            _triad_label(realized[(match.phase + i) % len(realized)].root, realized[(match.phase + i) % len(realized)].quality)
            for i in range(len(chunk))
        ]

        chunk_corrected = False
        for offset, (seg, expected_label) in enumerate(zip(chunk, expected_labels)):
            if seg.confidence <= confidence_threshold and seg.chord != expected_label:
                cleaned[chunk_start + offset] = ChordSegment(
                    start=seg.start,
                    end=seg.end,
                    chord=expected_label,
                    correlation=seg.correlation,
                    confidence=seg.confidence,
                    progression_corrected=True,
                )
                chunk_corrected = True

        if chunk_corrected:
            matches.append(match)

    return ProgressionCleanupResult(chords=_merge_adjacent(cleaned), matches=matches)


def _diatonic_or_majority_quality(root: str, qualities: list[str], key_tonic: str, key_mode: str) -> str:
    for quality in ("major", "minor"):
        if analyze_chord_function(root, quality, key_tonic, key_mode).is_diatonic:
            return quality
    # Neither reading is the key's own diatonic triad on this root (a
    # genuinely chromatic root) — nothing to prefer on theory grounds, so
    # fall back to whichever quality actually showed up more often.
    return Counter(qualities).most_common(1)[0][0]


def resolve_quality_oscillation(
    chords: list[ChordSegment],
    key_tonic: str,
    key_mode: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[ChordSegment]:
    """Collapse a root that keeps flipping between major and minor back to
    a single quality, using the detected key's own diatonic triad on that
    root as the tiebreaker — e.g. "Fm, F, Fm, F, ..." collapses to a
    single "F" if the song is in a key where F is diatonically major (an
    A minor song's bVI), or to a single "Am" instead of "A" for the tonic
    of that same key, since A minor's own i chord is minor. This isn't a
    blanket "prefer major" rule: whichever reading actually belongs to
    the key wins, in either direction.

    Why this exists: a power/distorted-guitar chord (root + fifth, no
    third) genuinely carries no major/minor information in its chroma, so
    detect_chords' template matching is close to a coin flip between the
    two — the same chord repeating in the same riff commonly shows up as
    alternating labels on the same root even though only one chord is
    actually being played, a different failure mode than a wrong root
    entirely (see clean_chords_with_progression) or an isolated single-
    window misread (see chords.py's own smoothing pass).

    Within each maximal run of consecutive segments sharing a root, if
    more than one quality appears, every segment in the run whose
    confidence is at or below `confidence_threshold` is reassigned to the
    winning quality; a segment the detector was already confident about
    is left alone even if it's the "wrong" side of the flip, same
    conservative rule as clean_chords_with_progression. Adjacent segments
    that end up sharing a chord afterward are merged, same as
    detect_chords does for its own raw output.
    """
    if len(chords) < 2:
        return chords

    roots = [parse_chord_label(seg.chord)[0] for seg in chords]
    qualities = [parse_chord_label(seg.chord)[1] for seg in chords]

    cleaned = list(chords)
    run_start = 0
    for i in range(1, len(chords) + 1):
        if i < len(chords) and roots[i] == roots[run_start]:
            continue

        run_end = i  # exclusive
        run_qualities = qualities[run_start:run_end]
        if run_end - run_start >= 2 and len(set(run_qualities)) > 1:
            root = roots[run_start]
            winning_quality = _diatonic_or_majority_quality(root, run_qualities, key_tonic, key_mode)
            winning_label = root if winning_quality == "major" else f"{root}m"

            for j in range(run_start, run_end):
                seg = chords[j]
                if seg.confidence <= confidence_threshold and seg.chord != winning_label:
                    cleaned[j] = ChordSegment(
                        start=seg.start,
                        end=seg.end,
                        chord=winning_label,
                        correlation=seg.correlation,
                        confidence=seg.confidence,
                        quality_resolved=True,
                    )

        run_start = i

    return _merge_adjacent(cleaned)
