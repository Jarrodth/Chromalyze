"""The CAGED system — 5 moveable major-chord shapes, named for the open
chords they're based on (C, A, G, E, D), that together cover the entire
neck: each shape is literally that open chord's own fret pattern, barred
and slid up to a new root. The 5 shapes also anchor the 5 "box" scale
patterns guitarists use for improvisation.

Specific to standard 6-string guitar tuning (E A D G B E): CAGED's 5
shapes look different from each other specifically because of standard
tuning's string-interval pattern (perfect 4ths between every adjacent pair
except G-B, a major 3rd). That asymmetry is what CAGED is built on, so
this module intentionally does NOT generalize to drop tunings, 7/8-string,
or bass — the same 5-shapes idea doesn't hold once the tuning changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instruments import GUITAR_STANDARD, FretPosition, fretboard_positions
from .theory import NOTE_NAME_TO_PITCH_CLASS

# The fixed order the 5 shapes appear in going up the neck from any given
# root — real and standard regardless of which root you start on; only the
# actual fret numbers shift.
CAGED_ORDER = ["C", "A", "G", "E", "D"]

# The conventional numbering for the 5 scale "box" positions specifically —
# distinct from CAGED_ORDER above, which is the chord-shape sequence. This
# is CAGED_ORDER rotated to start at E: widely-taught pentatonic/scale box
# systems number "Position 1" as the E-shape-anchored box (e.g. A minor
# pentatonic's famous "box 1" starts at fret 5, the E shape's position for
# that root), then continue E-D-C-A-G. Fixing the shape identity to each
# position number (rather than sorting boxes by ascending fret, which was
# this module's original behavior) means "Position 1" always means the same
# shape in every key — verified against a real reference site's own
# position numbering for C major, which lands in this exact E-D-C-A-G
# order.
SCALE_POSITION_ORDER = ["E", "D", "C", "A", "G"]


@dataclass
class CagedShapeTemplate:
    root_string_index: int  # which string this shape is conventionally positioned/referenced by
    natural_root_pc: int  # the root's pitch class when this shape is in its real open, unbarred position
    fret_offsets: dict[int, int]  # string_index -> fret, in that natural position; a muted string is simply absent


# Each shape's fret_offsets is exactly its real open-chord fingering — the
# whole reason the system is named C-A-G-E-D is that each moveable shape
# literally IS one of these 5 open chords, barred and slid up the neck
# (e.g. the E shape is just an open E major chord moved up).
CAGED_SHAPES: dict[str, CagedShapeTemplate] = {
    "C": CagedShapeTemplate(root_string_index=1, natural_root_pc=0, fret_offsets={1: 3, 2: 2, 3: 0, 4: 1, 5: 0}),
    "A": CagedShapeTemplate(root_string_index=1, natural_root_pc=9, fret_offsets={1: 0, 2: 2, 3: 2, 4: 2, 5: 0}),
    "G": CagedShapeTemplate(
        root_string_index=0, natural_root_pc=7, fret_offsets={0: 3, 1: 2, 2: 0, 3: 0, 4: 0, 5: 3}
    ),
    "E": CagedShapeTemplate(root_string_index=0, natural_root_pc=4, fret_offsets={0: 0, 1: 2, 2: 2, 3: 1, 4: 0, 5: 0}),
    "D": CagedShapeTemplate(root_string_index=2, natural_root_pc=2, fret_offsets={2: 0, 3: 2, 4: 3, 5: 2}),
}

_ROLE_BY_INTERVAL = {0: "root", 4: "third", 7: "fifth"}


@dataclass
class CagedChordPosition:
    string_index: int
    fret: int
    pitch_class: int
    role: str  # "root", "third", or "fifth"


@dataclass
class CagedChordVoicing:
    shape: str  # "C", "A", "G", "E", or "D"
    root: str
    positions: list[CagedChordPosition]  # muted strings simply absent


def caged_chord_shapes(root: str) -> list[CagedChordVoicing]:
    """All 5 moveable major-chord shapes for `root`, each the real open
    C/A/G/E/D chord transposed up the neck to the nearest fret that
    produces this root — e.g. `caged_chord_shapes("C")` gives 5 different
    ways to play a C major chord, sorted by fret position up the neck (in
    this case landing in the canonical C-A-G-E-D order at frets 0, 3, 5,
    8, 10 — the same order and, for C specifically, the same frets every
    guitarist learns them in).
    """
    root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
    voicings = []
    for shape_name, template in CAGED_SHAPES.items():
        transpose = (root_pc - template.natural_root_pc) % 12
        positions = []
        for string_index, offset in sorted(template.fret_offsets.items()):
            fret = offset + transpose
            open_pc = GUITAR_STANDARD.open_string_pitch_classes[string_index]
            pc = (open_pc + fret) % 12
            role = _ROLE_BY_INTERVAL[(pc - root_pc) % 12]
            positions.append(CagedChordPosition(string_index=string_index, fret=fret, pitch_class=pc, role=role))
        voicings.append(CagedChordVoicing(shape=shape_name, root=root, positions=positions))
    voicings.sort(key=lambda v: min(p.fret for p in v.positions))
    return voicings


@dataclass
class CagedScaleBox:
    shape: str  # "C", "A", "G", "E", or "D" — which shape this box surrounds
    min_fret: int
    max_fret: int
    positions: list[FretPosition]


def caged_scale_boxes(tonic: str, pitch_classes: list[int], padding: int = 1) -> list[CagedScaleBox]:
    """The 5 CAGED-labeled scale "box" patterns covering the neck for a
    scale (or any set of `pitch_classes`, e.g. `Scale.pitch_classes` from
    theory.py) in `tonic` — one box per shape, each centered on where that
    shape's own chord tones fall, widened by `padding` frets on either side
    to cover the fuller pattern guitarists actually play around each shape.
    Always returned in SCALE_POSITION_ORDER (E-D-C-A-G), not sorted by
    fret, so "Position 1" names the same shape regardless of root — a
    box's fret range can therefore land lower than an earlier-numbered
    box's for some roots (e.g. for C, the A-shape and G-shape boxes sit
    below the E-shape one), which is expected: position identity comes
    from the shape, not from neck order.

    `padding=1` (a 4-5 fret span depending on the shape's own natural
    spread) verified to still cover all 6 strings for every scale type this
    app offers (see tests) — narrower than an earlier padding=2 default,
    matched to the ~5-fret, ~3-notes-per-string boxes real reference sites
    use rather than the wider, denser boxes padding=2 produced.
    """
    root_pc = NOTE_NAME_TO_PITCH_CLASS[tonic]
    boxes = []
    for shape_name in SCALE_POSITION_ORDER:
        template = CAGED_SHAPES[shape_name]
        transpose = (root_pc - template.natural_root_pc) % 12
        shape_frets = [offset + transpose for offset in template.fret_offsets.values()]
        min_fret = max(0, min(shape_frets) - padding)
        max_fret = max(shape_frets) + padding
        positions = [
            p
            for p in fretboard_positions(pitch_classes, GUITAR_STANDARD, num_frets=max_fret)
            if min_fret <= p.fret <= max_fret
        ]
        boxes.append(CagedScaleBox(shape=shape_name, min_fret=min_fret, max_fret=max_fret, positions=positions))
    return boxes
