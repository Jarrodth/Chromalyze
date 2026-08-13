"""A catalog of named scales that don't fit the 7-diatonic-mode system in
theory.py: the two pentatonic scales, the blues scale, and the two extra
minor forms (harmonic minor, melodic minor) that give natural minor its
raised leading tone / raised 6th-and-7th.

Spelling reuses theory.py's `_spell_intervals`, since a pentatonic or blues
scale is still just "intervals above a tonic, each with a letter" — the
only difference from a 7-note diatonic scale is that some letters get
skipped (pentatonic) or reused with a different accidental (blues' blue
note), instead of using each of the 7 letters exactly once.
"""

from __future__ import annotations

from .theory import NOTE_NAME_TO_PITCH_CLASS, Scale, _spell_intervals

# (semitone intervals above the tonic, letter offsets from the tonic's own
# letter) for each named scale. Letter offsets are derived from the real
# scale each one comes from, not invented independently:
# - major/minor pentatonic are the major/natural-minor scale with 2 notes
#   removed, so they reuse that scale's own letters (skipping the omitted
#   degrees) rather than picking new ones.
# - blues is minor pentatonic plus one chromatic "blue note" (a flat 5th)
#   inserted between the 4th and 5th degrees — conventionally spelled by
#   reusing the 5th degree's own letter with a flat, e.g. C blues is
#   C Eb F Gb G Bb (both "G" and "Gb" appear; unlike a diatonic scale,
#   there's no rule against reusing a letter here, since the scale isn't
#   trying to use all 7 letters in the first place).
_SCALES: dict[str, tuple[list[int], list[int]]] = {
    "major_pentatonic": ([0, 2, 4, 7, 9], [0, 1, 2, 4, 5]),
    "minor_pentatonic": ([0, 3, 5, 7, 10], [0, 2, 3, 4, 6]),
    "blues": ([0, 3, 5, 6, 7, 10], [0, 2, 3, 4, 4, 6]),
    "harmonic_minor": ([0, 2, 3, 5, 7, 8, 11], [0, 1, 2, 3, 4, 5, 6]),
    "melodic_minor": ([0, 2, 3, 5, 7, 9, 11], [0, 1, 2, 3, 4, 5, 6]),
}

NAMED_SCALE_INTERVALS = {name: intervals for name, (intervals, _) in _SCALES.items()}


def build_named_scale(tonic: str, name: str) -> Scale:
    """Build one of the scales in `NAMED_SCALE_INTERVALS` (pentatonic,
    blues, harmonic minor, melodic minor) for any tonic.

    Harmonic minor and melodic minor are ordinary 7-note scales, so
    `diatonic_triads`/`diatonic_sevenths` from theory.py work on the result
    exactly as they do for any of the 7 modes — e.g. harmonic minor's
    raised 7th degree correctly produces an augmented bIII chord and a
    proper (major-quality) V, both hallmarks of real harmonic-minor harmony
    that natural minor doesn't have.
    """
    intervals, letter_offsets = _SCALES[name]
    tonic_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    pitch_classes = [(tonic_pc + interval) % 12 for interval in intervals]
    notes = _spell_intervals(tonic, intervals, letter_offsets)
    return Scale(tonic=tonic, mode=name, notes=notes, pitch_classes=pitch_classes)
