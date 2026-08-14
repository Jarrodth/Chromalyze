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


def _spell_intervals(tonic: str, intervals: list[int], letter_offsets: list[int]) -> list[str]:
    """Spell each of `intervals` (semitones above `tonic`) using real letter
    names with the correct accidental, rather than a fixed chromatic lookup
    that could reuse a letter or skip one entirely. `letter_offsets[i]` is
    how many letters past the tonic's own letter note `i` sits: consecutive
    integers (0,1,2,...) for a 7-note scale that uses every letter once;
    every-other integers (0,2,4,...) for a tertian chord, whose letters skip
    by a third each time (e.g. C-E-G-B skips D/F/A); or an arbitrary,
    possibly-repeating sequence for a scale that skips or reuses letters,
    like a pentatonic or blues scale.
    """
    tonic_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    start_letter_index = LETTER_ORDER.index(tonic[0])

    notes = []
    for interval, letter_offset in zip(intervals, letter_offsets):
        target_pc = (tonic_pc + interval) % 12
        letter = LETTER_ORDER[(start_letter_index + letter_offset) % 7]
        natural_pc = NATURAL_LETTER_PITCH_CLASS[letter]
        diff = (target_pc - natural_pc) % 12
        if diff > 6:
            diff -= 12
        notes.append(letter + _accidental_suffix(diff))
    return notes


def spell_scale(tonic: str, mode: str) -> list[str]:
    """Spell a scale using each of the 7 natural letter names exactly once
    (starting from the tonic's own letter), with the correct accidental on
    each — not a fixed chromatic lookup, which could otherwise reuse a
    letter or skip one entirely (e.g. G major must be spelled G A B C D E
    F#, never G A B C D E Gb — that reuses "G" and never uses "F").
    """
    mode = _resolve_mode(mode)
    intervals = MODE_INTERVALS[mode]
    return _spell_intervals(tonic, intervals, letter_offsets=list(range(len(intervals))))


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


_TRIAD_QUALITY_NUMERAL_DECORATION = {
    "major": lambda numeral: numeral,
    "minor": lambda numeral: numeral.lower(),
    "diminished": lambda numeral: numeral.lower() + "°",
    "augmented": lambda numeral: numeral + "+",
    "other": lambda numeral: numeral + "?",
}

# Semitone intervals above the root for every triad and seventh-chord
# quality this module knows how to name. Shared by `build_chord` (spelling
# an arbitrary chord from a root + quality) and the diatonic-chord builders
# below (deriving quality from a scale, then looking the chord back up
# here to get real spelled note names).
CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "major7": [0, 4, 7, 11],
    "dominant7": [0, 4, 7, 10],
    "minor7": [0, 3, 7, 10],
    "minor-major7": [0, 3, 7, 11],
    "half-diminished7": [0, 3, 6, 10],
    "diminished7": [0, 3, 6, 9],
    "augmented-major7": [0, 4, 8, 11],
    "augmented7": [0, 4, 8, 10],
}


@dataclass
class Chord:
    root: str
    quality: str
    notes: list[str]  # properly spelled, root through the highest chord tone
    pitch_classes: list[int]  # same order as notes


def build_chord(root: str, quality: str) -> Chord:
    """Build any triad or seventh chord from a root note and a quality
    (one of `CHORD_INTERVALS`'s keys) — independent of any scale or key,
    e.g. `build_chord("D", "minor7")` for a plain Dm7.
    """
    intervals = CHORD_INTERVALS[quality]
    root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
    pitch_classes = [(root_pc + i) % 12 for i in intervals]
    notes = _spell_intervals(root, intervals, letter_offsets=[i * 2 for i in range(len(intervals))])
    return Chord(root=root, quality=quality, notes=notes, pitch_classes=pitch_classes)


def build_power_chord(root: str) -> Chord:
    """A power chord — just the root and its perfect fifth, no third, so
    it's neither major nor minor (the whole reason guitarists reach for
    it: it works interchangeably over either quality underneath).

    Deliberately not just another CHORD_INTERVALS entry fed through
    build_chord: that function's letter-spelling assumes tertian stacking
    (each successive chord tone a third — 2 letters — above the last, so
    it can build the whole notes list from `[i * 2 for i in range(...)]`).
    A fifth is 4 letters above the root (C-D-E-F-G), not 2, so reusing
    that logic here would misspell it (e.g. C's fifth as some form of E,
    not G). letter_offsets=[0, 4] spells it correctly instead.
    """
    intervals = [0, 7]
    root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
    pitch_classes = [(root_pc + i) % 12 for i in intervals]
    notes = _spell_intervals(root, intervals, letter_offsets=[0, 4])
    return Chord(root=root, quality="power", notes=notes, pitch_classes=pitch_classes)


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
        numeral = _TRIAD_QUALITY_NUMERAL_DECORATION[quality](ROMAN_NUMERALS[degree])
        triads.append(
            DiatonicChord(degree=degree + 1, root=scale.notes[degree], quality=quality, roman_numeral=numeral)
        )
    return triads


def _classify_seventh(root_to_third: int, root_to_fifth: int, root_to_seventh: int) -> str:
    triad_quality = _classify_triad(root_to_third, root_to_fifth)
    if triad_quality == "major":
        if root_to_seventh == 11:
            return "major7"
        if root_to_seventh == 10:
            return "dominant7"
    elif triad_quality == "minor":
        if root_to_seventh == 10:
            return "minor7"
        if root_to_seventh == 11:
            return "minor-major7"
    elif triad_quality == "diminished":
        if root_to_seventh == 10:
            return "half-diminished7"
        if root_to_seventh == 9:
            return "diminished7"
    elif triad_quality == "augmented":
        if root_to_seventh == 11:
            return "augmented-major7"
        if root_to_seventh == 10:
            return "augmented7"
    return "other7"


_SEVENTH_QUALITY_NUMERAL_DECORATION = {
    "major7": lambda numeral: numeral + "maj7",
    "dominant7": lambda numeral: numeral + "7",
    "minor7": lambda numeral: numeral.lower() + "7",
    "minor-major7": lambda numeral: numeral.lower() + "(maj7)",
    "half-diminished7": lambda numeral: numeral.lower() + "ø7",
    "diminished7": lambda numeral: numeral.lower() + "°7",
    "augmented-major7": lambda numeral: numeral + "+(maj7)",
    "augmented7": lambda numeral: numeral + "+7",
    "other7": lambda numeral: numeral + "7?",
}

# Used by analyze_chord_function, which may be handed either a triad or a
# seventh-chord quality — one combined lookup covers both.
_CHORD_QUALITY_NUMERAL_DECORATION = {**_TRIAD_QUALITY_NUMERAL_DECORATION, **_SEVENTH_QUALITY_NUMERAL_DECORATION}


@dataclass
class DiatonicSeventhChord:
    degree: int  # 1-7
    root: str  # note name, e.g. "D"
    quality: str  # a CHORD_INTERVALS key ending in "7", e.g. "minor7", "half-diminished7"
    roman_numeral: str  # e.g. "Imaj7", "ii7", "viiø7"


def diatonic_sevenths(scale: Scale) -> list[DiatonicSeventhChord]:
    """The 7 four-note chords built by stacking a further third onto each
    of `scale`'s diatonic triads (see `diatonic_triads`) — same
    stack-and-classify approach, one third taller, so it's correct for any
    of the 7 modes without a hardcoded per-mode table. In a major scale
    this is the familiar Imaj7 ii7 iii7 IVmaj7 V7 vi7 viiø7.
    """
    n = len(scale.pitch_classes)
    sevenths = []
    for degree in range(n):
        root = scale.pitch_classes[degree]
        third = scale.pitch_classes[(degree + 2) % n]
        fifth = scale.pitch_classes[(degree + 4) % n]
        seventh = scale.pitch_classes[(degree + 6) % n]
        root_to_third = (third - root) % 12
        root_to_fifth = (fifth - root) % 12
        root_to_seventh = (seventh - root) % 12
        quality = _classify_seventh(root_to_third, root_to_fifth, root_to_seventh)
        numeral = _SEVENTH_QUALITY_NUMERAL_DECORATION[quality](ROMAN_NUMERALS[degree])
        sevenths.append(
            DiatonicSeventhChord(degree=degree + 1, root=scale.notes[degree], quality=quality, roman_numeral=numeral)
        )
    return sevenths


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


def _nearest_scale_degree(interval: int, mode: str) -> tuple[int, int]:
    """The scale degree (0-6) whose plain reference position is closest to
    `interval` semitones above the tonic, and the signed semitone offset
    from that reference — the shared math behind both naming a chromatic
    interval's roman numeral and spelling its root note's letter, since
    both are really "which scale degree is this closest to, and by how
    much is it altered".
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

    return best_degree, best_diff


def _roman_numeral_for_interval(interval: int, mode: str) -> str:
    """Base (uppercase, undecorated) roman numeral for a semitone interval
    above the tonic, relative to `mode`'s own scale degrees — e.g. in major,
    interval 4 is the natural (major) 3rd -> "III"; interval 3 is a
    semitone flat of that -> "bIII". Handles any interval, not just ones
    that land exactly on a scale degree, so chromatic/borrowed chords get a
    sensible numeral too.
    """
    degree, diff = _nearest_scale_degree(interval, mode)
    return _accidental_suffix(diff) + ROMAN_NUMERALS[degree]


def _root_name_for_interval(tonic: str, mode: str, interval: int) -> str:
    """The correctly letter-spelled note name for a chord root sitting
    `interval` semitones above `tonic`, using the same nearest-scale-degree
    logic as `_roman_numeral_for_interval` — e.g. in C major, interval 10
    (a bVII root) is spelled "Bb", reusing the 7th degree's own letter "B"
    with a flat, not "A#".
    """
    degree, _ = _nearest_scale_degree(interval, mode)
    return _spell_intervals(tonic, [interval], [degree])[0]


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
    numeral = _CHORD_QUALITY_NUMERAL_DECORATION.get(chord_quality, lambda n: n + "?")(base_numeral)

    scale = build_scale(key_tonic, key_mode_resolved)
    triads = diatonic_triads(scale)
    is_diatonic = any(
        NOTE_NAME_TO_PITCH_CLASS[t.root] == chord_root_pc and t.quality == chord_quality for t in triads
    )

    return ChordFunction(roman_numeral=numeral, is_diatonic=is_diatonic)
