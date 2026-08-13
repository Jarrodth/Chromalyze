"""Instrument/tuning definitions and fretboard position mapping — maps
pure music theory (theory.py's scales/chords) onto real playable positions
for fretted string instruments.

Keyboard instruments deliberately have no equivalent here: a scale's pitch
classes map directly onto piano keys with no per-instrument variation to
account for, unlike a guitar or bass where the same scale looks completely
different depending on tuning and string count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .theory import NOTE_NAME_TO_PITCH_CLASS

DEFAULT_NUM_FRETS = 24


@dataclass
class Tuning:
    name: str
    open_string_pitch_classes: list[int]  # lowest-pitched string first
    open_string_notes: list[str] = field(default_factory=list)  # letter names, same order/length

    @classmethod
    def from_note_names(cls, name: str, notes: list[str]) -> "Tuning":
        return cls(
            name=name,
            open_string_pitch_classes=[NOTE_NAME_TO_PITCH_CLASS[n] for n in notes],
            open_string_notes=list(notes),
        )


# Guitar — standard + drop tunings + common alternate tunings + extended range.
GUITAR_STANDARD = Tuning.from_note_names("Guitar - Standard (E A D G B E)", ["E", "A", "D", "G", "B", "E"])
GUITAR_DROP_D = Tuning.from_note_names("Guitar - Drop D (D A D G B E)", ["D", "A", "D", "G", "B", "E"])
GUITAR_DROP_C = Tuning.from_note_names("Guitar - Drop C (C G C F A D)", ["C", "G", "C", "F", "A", "D"])
GUITAR_DROP_B = Tuning.from_note_names("Guitar - Drop B (B F# B E G# C#)", ["B", "F#", "B", "E", "G#", "C#"])
GUITAR_HALF_STEP_DOWN = Tuning.from_note_names(
    "Guitar - Half Step Down (Eb Ab Db Gb Bb Eb)", ["Eb", "Ab", "Db", "Gb", "Bb", "Eb"]
)
GUITAR_DADGAD = Tuning.from_note_names("Guitar - DADGAD", ["D", "A", "D", "G", "A", "D"])
GUITAR_OPEN_D = Tuning.from_note_names("Guitar - Open D (D A D F# A D)", ["D", "A", "D", "F#", "A", "D"])
GUITAR_OPEN_G = Tuning.from_note_names("Guitar - Open G (D G D G B D)", ["D", "G", "D", "G", "B", "D"])
GUITAR_7_STRING_STANDARD = Tuning.from_note_names(
    "7-String Guitar - Standard (B E A D G B E)", ["B", "E", "A", "D", "G", "B", "E"]
)
GUITAR_7_STRING_DROP_A = Tuning.from_note_names(
    "7-String Guitar - Drop A (A E A D G B E)", ["A", "E", "A", "D", "G", "B", "E"]
)
GUITAR_8_STRING_STANDARD = Tuning.from_note_names(
    "8-String Guitar - Standard (F# B E A D G B E)", ["F#", "B", "E", "A", "D", "G", "B", "E"]
)

# Bass — standard + drop + extended range.
BASS_STANDARD = Tuning.from_note_names("Bass - Standard (E A D G)", ["E", "A", "D", "G"])
BASS_DROP_D = Tuning.from_note_names("Bass - Drop D (D A D G)", ["D", "A", "D", "G"])
BASS_HALF_STEP_DOWN = Tuning.from_note_names("Bass - Half Step Down (Eb Ab Db Gb)", ["Eb", "Ab", "Db", "Gb"])
BASS_5_STRING_STANDARD = Tuning.from_note_names("5-String Bass - Standard (B E A D G)", ["B", "E", "A", "D", "G"])
BASS_5_STRING_HIGH_C = Tuning.from_note_names("5-String Bass - High C (E A D G C)", ["E", "A", "D", "G", "C"])
BASS_6_STRING_STANDARD = Tuning.from_note_names(
    "6-String Bass - Standard (B E A D G C)", ["B", "E", "A", "D", "G", "C"]
)

PRESET_TUNINGS: dict[str, Tuning] = {
    "guitar_standard": GUITAR_STANDARD,
    "guitar_drop_d": GUITAR_DROP_D,
    "guitar_drop_c": GUITAR_DROP_C,
    "guitar_drop_b": GUITAR_DROP_B,
    "guitar_half_step_down": GUITAR_HALF_STEP_DOWN,
    "guitar_dadgad": GUITAR_DADGAD,
    "guitar_open_d": GUITAR_OPEN_D,
    "guitar_open_g": GUITAR_OPEN_G,
    "guitar_7string_standard": GUITAR_7_STRING_STANDARD,
    "guitar_7string_drop_a": GUITAR_7_STRING_DROP_A,
    "guitar_8string_standard": GUITAR_8_STRING_STANDARD,
    "bass_standard": BASS_STANDARD,
    "bass_drop_d": BASS_DROP_D,
    "bass_half_step_down": BASS_HALF_STEP_DOWN,
    "bass_5string_standard": BASS_5_STRING_STANDARD,
    "bass_5string_high_c": BASS_5_STRING_HIGH_C,
    "bass_6string_standard": BASS_6_STRING_STANDARD,
}


@dataclass
class FretPosition:
    string_index: int  # 0 = lowest-pitched string
    fret: int  # 0 = open string
    pitch_class: int
    scale_degree: int | None  # 1-based position within the pitch_classes passed in, if found


def fretboard_positions(
    pitch_classes: list[int], tuning: Tuning, num_frets: int = DEFAULT_NUM_FRETS
) -> list[FretPosition]:
    """Every (string, fret) position on `tuning`'s fretboard, up to
    `num_frets` frets, whose note is one of `pitch_classes` (typically
    Scale.pitch_classes from theory.py, so a chord shape or full scale can
    both be mapped the same way — pass whichever pitch classes matter).
    """
    degree_by_pitch_class = {pc: i + 1 for i, pc in enumerate(pitch_classes)}

    positions = []
    for string_index, open_pc in enumerate(tuning.open_string_pitch_classes):
        for fret in range(num_frets + 1):
            pc = (open_pc + fret) % 12
            if pc in degree_by_pitch_class:
                positions.append(
                    FretPosition(
                        string_index=string_index,
                        fret=fret,
                        pitch_class=pc,
                        scale_degree=degree_by_pitch_class[pc],
                    )
                )
    return positions
