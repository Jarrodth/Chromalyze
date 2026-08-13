"""Interval theory — naming and quality classification for the distance
between two notes. This is the piece the rest of music theory is built on
top of (a "major third" is what makes a triad major, not just "4 semitones"),
and it's also useful standalone for teaching: showing someone that F-to-B is
an augmented 4th while C-to-G is a perfect 5th.

Two entry points, for two different situations:
- `interval_between(lower, upper)`: takes real spelled note names (e.g. "F",
  "B") and gives the theoretically correct, letter-aware quality. This is
  the one to use whenever real note spelling is available, since letter
  distance is what actually distinguishes intervals real notation treats as
  different even though they sound identical — e.g. F-to-B (augmented 4th)
  vs its enharmonic twin B-to-F (diminished 5th): same 6 semitones, 2
  different, both real, interval names.
- `interval_from_semitones(semitones)`: takes a bare semitone count (0-11)
  with no note-name context, and returns music's single most conventional
  name for that distance. Useful when only pitch classes are available —
  e.g. straight out of key.py/chords.py, which work in raw chroma bins with
  no note-name spelling to disambiguate from.
"""

from __future__ import annotations

from dataclasses import dataclass

from .theory import LETTER_ORDER, NOTE_NAME_TO_PITCH_CLASS

DEGREE_NAMES = {
    1: "Unison",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth",
    6: "Sixth",
    7: "Seventh",
}

# How many semitones each degree spans in a plain major scale — the
# reference point every quality below is measured as a deviation from.
_DEGREE_REFERENCE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

# Unison, 4th, and 5th are "perfect" intervals: their qualities run
# diminished/perfect/augmented, with no separate major/minor distinction.
# Every other degree is "imperfect": diminished/minor/major/augmented.
_PERFECT_DEGREES = {1, 4, 5}

_PERFECT_QUALITY_BY_OFFSET = {
    -2: "doubly diminished",
    -1: "diminished",
    0: "perfect",
    1: "augmented",
    2: "doubly augmented",
}

_IMPERFECT_QUALITY_BY_OFFSET = {
    -3: "doubly diminished",
    -2: "diminished",
    -1: "minor",
    0: "major",
    1: "augmented",
    2: "doubly augmented",
}

_QUALITY_ABBREVIATIONS = {
    "doubly diminished": "dd",
    "diminished": "d",
    "minor": "m",
    "perfect": "P",
    "major": "M",
    "augmented": "A",
    "doubly augmented": "AA",
}


@dataclass
class Interval:
    degree: int  # 1-7 (unison through seventh)
    quality: str  # "diminished" / "minor" / "major" / "perfect" / "augmented" (or doubly- variants)
    semitones: int  # 0-11
    name: str  # e.g. "Major Third", "Perfect Fifth", "Augmented Fourth"
    short_name: str  # e.g. "M3", "P5", "A4"


def _normalize_offset(offset: int) -> int:
    """Bring a raw semitone deviation into the shortest signed distance mod
    12 — needed because letter distance and semitone distance are computed
    independently, so a degree can come out "wrapped" (e.g. C up to B# is a
    letter-7th, but B# sits 0 semitones above C, which is 11 semitones past
    a plain major 7th's reference point — normalizing that -11 down to +1
    correctly identifies it as an augmented 7th, enharmonic to an octave).
    """
    if offset > 6:
        offset -= 12
    elif offset < -6:
        offset += 12
    return offset


def _build_interval(degree: int, semitones: int) -> Interval:
    reference = _DEGREE_REFERENCE_SEMITONES[degree]
    offset = _normalize_offset(semitones - reference)
    table = _PERFECT_QUALITY_BY_OFFSET if degree in _PERFECT_DEGREES else _IMPERFECT_QUALITY_BY_OFFSET
    if offset not in table:
        raise ValueError(f"no standard interval quality for a degree-{degree} interval {offset:+d} semitones from perfect/major")
    quality = table[offset]
    name = f"{quality.title()} {DEGREE_NAMES[degree]}"
    short_name = f"{_QUALITY_ABBREVIATIONS[quality]}{degree}"
    return Interval(degree=degree, quality=quality, semitones=semitones, name=name, short_name=short_name)


def interval_between(lower: str, upper: str) -> Interval:
    """The interval spanning upward from `lower` to `upper` (real note
    names, e.g. "F", "B", "Eb"), within one octave.
    """
    lower_pc = NOTE_NAME_TO_PITCH_CLASS[lower]
    upper_pc = NOTE_NAME_TO_PITCH_CLASS[upper]
    semitones = (upper_pc - lower_pc) % 12

    letter_steps = (LETTER_ORDER.index(upper[0]) - LETTER_ORDER.index(lower[0])) % 7
    degree = letter_steps + 1

    return _build_interval(degree, semitones)


# The single conventional degree for each bare semitone distance (0-11),
# used when no note-name spelling is available to pick between enharmonic
# alternatives — e.g. 6 semitones is reported as an augmented 4th (the more
# common real-world spelling of a tritone) rather than its diminished-5th
# twin, and 3 semitones as a minor 3rd rather than an augmented 2nd.
_DEFAULT_SEMITONE_DEGREE = {
    0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 6, 10: 7, 11: 7,
}


def interval_from_semitones(semitones: int) -> Interval:
    """The conventional interval for a bare semitone distance (0-11), with
    no note-name spelling available. Always resolves to one canonical
    spelling per distance (see `_DEFAULT_SEMITONE_DEGREE`), not the full
    ambiguity real notation allows — use `interval_between` instead
    whenever real note names are on hand.
    """
    semitones = semitones % 12
    degree = _DEFAULT_SEMITONE_DEGREE[semitones]
    return _build_interval(degree, semitones)


def common_interval_reference() -> list[Interval]:
    """All 12 conventional intervals within an octave, unison through major
    7th, in semitone order — a ready-made reference table for teaching.
    """
    return [interval_from_semitones(semitones) for semitones in range(12)]
