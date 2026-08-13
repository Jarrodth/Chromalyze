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

from dataclasses import dataclass, field

from .theory import (
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
