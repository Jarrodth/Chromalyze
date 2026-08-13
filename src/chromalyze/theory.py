"""Pure music theory — scales, diatonic chords, and roman numeral analysis.

Entirely instrument-agnostic: everything here works with pitch classes and
note names only. Mapping a scale onto a specific instrument's fretboard is
instruments.py's job, built on top of what's defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

# All 7 diatonic modes, as semitone intervals above the tonic. Ionian and
# Aeolian are the familiar major and natural minor scales; the rest are
# their own well-defined modes, included since "all possible music theory"
# means not just picking the two most common ones.
MODE_INTERVALS = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

# key.py's KeyResult reports "major"/"minor" — map those onto the modes
# that actually are major/natural-minor scales.
MODE_ALIASES = {"major": "ionian", "minor": "aeolian"}

NOTE_NAME_TO_PITCH_CLASS = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "E#": 5, "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

NATURAL_LETTER_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTER_ORDER = ["C", "D", "E", "F", "G", "A", "B"]

ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]


def _resolve_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def _accidental_suffix(diff: int) -> str:
    if diff == 0:
        return ""
    symbol = "#" if diff > 0 else "b"
    return symbol * abs(diff)


def spell_scale(tonic: str, mode: str) -> list[str]:
    """Spell a scale using each of the 7 natural letter names exactly once
    (starting from the tonic's own letter), with the correct accidental on
    each — not a fixed chromatic lookup, which could otherwise reuse a
    letter or skip one entirely (e.g. G major must be spelled G A B C D E
    F#, never G A B C D E Gb — that reuses "G" and never uses "F").
    """
    mode = _resolve_mode(mode)
    intervals = MODE_INTERVALS[mode]
    tonic_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    start_letter_index = LETTER_ORDER.index(tonic[0])

    notes = []
    for i, interval in enumerate(intervals):
        target_pc = (tonic_pc + interval) % 12
        letter = LETTER_ORDER[(start_letter_index + i) % 7]
        natural_pc = NATURAL_LETTER_PITCH_CLASS[letter]
        diff = (target_pc - natural_pc) % 12
        if diff > 6:
            diff -= 12
        notes.append(letter + _accidental_suffix(diff))
    return notes


@dataclass
class Scale:
    tonic: str
    mode: str
    notes: list[str]  # properly spelled, one per letter, in scale-degree order
    pitch_classes: list[int]  # same order as notes


def build_scale(tonic: str, mode: str) -> Scale:
    """Build a Scale for any tonic and any of the 7 modes."""
    resolved_mode = _resolve_mode(mode)
    tonic_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    intervals = MODE_INTERVALS[resolved_mode]
    pitch_classes = [(tonic_pc + i) % 12 for i in intervals]
    notes = spell_scale(tonic, mode)
    return Scale(tonic=tonic, mode=resolved_mode, notes=notes, pitch_classes=pitch_classes)


def _classify_triad(root_to_third: int, root_to_fifth: int) -> str:
    if root_to_third == 4 and root_to_fifth == 7:
        return "major"
    if root_to_third == 3 and root_to_fifth == 7:
        return "minor"
    if root_to_third == 3 and root_to_fifth == 6:
        return "diminished"
    if root_to_third == 4 and root_to_fifth == 8:
        return "augmented"
    return "other"


_QUALITY_NUMERAL_DECORATION = {
    "major": lambda numeral: numeral,
    "minor": lambda numeral: numeral.lower(),
    "diminished": lambda numeral: numeral.lower() + "°",
    "augmented": lambda numeral: numeral + "+",
    "other": lambda numeral: numeral + "?",
}


@dataclass
class DiatonicChord:
    degree: int  # 1-7
    root: str  # note name, e.g. "D"
    quality: str  # "major" / "minor" / "diminished" / "augmented"
    roman_numeral: str  # e.g. "I", "ii", "vii°"


def diatonic_triads(scale: Scale) -> list[DiatonicChord]:
    """The 7 triads built by stacking thirds within `scale` — the standard
    diatonic chord set for that key/mode, derived programmatically (by
    actually stacking scale thirds and measuring the resulting intervals)
    rather than a hardcoded per-mode chord table, so it's correct for any
    of the 7 modes, not just major/minor.
    """
    n = len(scale.pitch_classes)
    triads = []
    for degree in range(n):
        root = scale.pitch_classes[degree]
        third = scale.pitch_classes[(degree + 2) % n]
        fifth = scale.pitch_classes[(degree + 4) % n]
        root_to_third = (third - root) % 12
        root_to_fifth = (fifth - root) % 12
        quality = _classify_triad(root_to_third, root_to_fifth)
        numeral = _QUALITY_NUMERAL_DECORATION[quality](ROMAN_NUMERALS[degree])
        triads.append(
            DiatonicChord(degree=degree + 1, root=scale.notes[degree], quality=quality, roman_numeral=numeral)
        )
    return triads


def relative_key(tonic: str, mode: str) -> tuple[str, str]:
    """The relative major (from a minor key) or relative minor (from a
    major key) — same key signature, different tonic. Returns (tonic, mode).
    """
    resolved_mode = _resolve_mode(mode)
    tonic_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    if resolved_mode == "ionian":
        relative_pc = (tonic_pc - 3) % 12
        relative_mode = "aeolian"
    elif resolved_mode == "aeolian":
        relative_pc = (tonic_pc + 3) % 12
        relative_mode = "ionian"
    else:
        raise ValueError("relative_key only applies to major (ionian) / minor (aeolian) keys")

    from .key import MAJOR_TONIC_NAMES, MINOR_TONIC_NAMES

    names = MAJOR_TONIC_NAMES if relative_mode == "ionian" else MINOR_TONIC_NAMES
    return names[relative_pc], relative_mode


def parallel_key(tonic: str, mode: str) -> tuple[str, str]:
    """Same tonic, opposite mode (e.g. C major -> C minor)."""
    resolved_mode = _resolve_mode(mode)
    if resolved_mode not in ("ionian", "aeolian"):
        raise ValueError("parallel_key only applies to major (ionian) / minor (aeolian) keys")
    parallel_mode = "aeolian" if resolved_mode == "ionian" else "ionian"
    return tonic, parallel_mode


def _roman_numeral_for_interval(interval: int, mode: str) -> str:
    """Base (uppercase, undecorated) roman numeral for a semitone interval
    above the tonic, relative to `mode`'s own scale degrees — e.g. in major,
    interval 4 is the natural (major) 3rd -> "III"; interval 3 is a
    semitone flat of that -> "bIII". Handles any interval, not just ones
    that land exactly on a scale degree, so chromatic/borrowed chords get a
    sensible numeral too.
    """
    resolved_mode = _resolve_mode(mode)
    scale_intervals = MODE_INTERVALS[resolved_mode]

    best_degree = 0
    best_diff = None
    for degree, degree_interval in enumerate(scale_intervals):
        diff = interval - degree_interval
        if diff > 6:
            diff -= 12
        if diff < -6:
            diff += 12
        # On a tie, prefer the flat (negative) spelling — bVII/bIII/bVI
        # (borrowed chords a whole step below a scale degree) are far more
        # common in real music than the enharmonically-equivalent sharp
        # spelling of the degree below.
        if best_diff is None or abs(diff) < abs(best_diff) or (abs(diff) == abs(best_diff) and diff < best_diff):
            best_diff = diff
            best_degree = degree

    return _accidental_suffix(best_diff) + ROMAN_NUMERALS[best_degree]


@dataclass
class ChordFunction:
    roman_numeral: str  # e.g. "I", "vi", "bVII"
    is_diatonic: bool  # True if this exact root+quality is one of the key's own diatonic triads


def analyze_chord_function(chord_root: str, chord_quality: str, key_tonic: str, key_mode: str) -> ChordFunction:
    """Label a chord (e.g. from detect_chords) with its roman-numeral
    function within a given key (e.g. from detect_key) — the piece that
    actually connects chord recognition and key detection into real
    harmonic analysis.
    """
    key_mode_resolved = _resolve_mode(key_mode)
    key_tonic_pc = NOTE_NAME_TO_PITCH_CLASS[key_tonic]
    chord_root_pc = NOTE_NAME_TO_PITCH_CLASS[chord_root]
    interval = (chord_root_pc - key_tonic_pc) % 12

    base_numeral = _roman_numeral_for_interval(interval, key_mode_resolved)
    numeral = _QUALITY_NUMERAL_DECORATION.get(chord_quality, lambda n: n + "?")(base_numeral)

    scale = build_scale(key_tonic, key_mode_resolved)
    triads = diatonic_triads(scale)
    is_diatonic = any(
        NOTE_NAME_TO_PITCH_CLASS[t.root] == chord_root_pc and t.quality == chord_quality for t in triads
    )

    return ChordFunction(roman_numeral=numeral, is_diatonic=is_diatonic)
