from chromalyze.caged import CAGED_ORDER, CAGED_SHAPES, SCALE_POSITION_ORDER, caged_chord_shapes, caged_scale_boxes
from chromalyze.scales import build_named_scale
from chromalyze.theory import build_scale


def test_caged_shape_templates_match_real_open_chords():
    # Each shape's natural (untransposed) position must be exactly the
    # real, well-known open chord it's named after — verified against the
    # actual pitch classes those open chords produce.
    guitar_open_strings = [4, 9, 2, 7, 11, 4]  # E A D G B E

    def pitch_classes_for(template):
        return sorted({(guitar_open_strings[s] + f) % 12 for s, f in template.fret_offsets.items()})

    # C major = {0, 4, 7}; open C fingering is X-3-2-0-1-0
    assert pitch_classes_for(CAGED_SHAPES["C"]) == [0, 4, 7]
    # A major = {9, 1, 4}; open A fingering is X-0-2-2-2-0
    assert pitch_classes_for(CAGED_SHAPES["A"]) == sorted({9, 1, 4})
    # G major = {7, 11, 2}; open G fingering is 3-2-0-0-0-3
    assert pitch_classes_for(CAGED_SHAPES["G"]) == sorted({7, 11, 2})
    # E major = {4, 8, 11}; open E fingering is 0-2-2-1-0-0
    assert pitch_classes_for(CAGED_SHAPES["E"]) == sorted({4, 8, 11})
    # D major = {2, 6, 9}; open D fingering is X-X-0-2-3-2
    assert pitch_classes_for(CAGED_SHAPES["D"]) == sorted({2, 6, 9})


def test_caged_chord_shapes_c_major_lands_at_the_textbook_frets():
    # The famous fret positions every guitarist learns C major at, in
    # C-A-G-E-D order: open (C shape), 3rd fret (A shape), 5th fret
    # (G shape), 8th fret (E shape), 10th fret (D shape).
    voicings = caged_chord_shapes("C")
    assert [v.shape for v in voicings] == CAGED_ORDER
    assert [min(p.fret for p in v.positions) for v in voicings] == [0, 3, 5, 8, 10]


def test_caged_chord_shapes_are_always_a_real_major_triad():
    from chromalyze.theory import NOTE_NAME_TO_PITCH_CLASS

    for root in ["C", "D", "E", "F", "G", "A", "B", "F#", "Bb"]:
        root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
        expected_pitch_classes = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
        expected_role_by_interval = {0: "root", 4: "third", 7: "fifth"}

        for voicing in caged_chord_shapes(root):
            pitch_classes = {p.pitch_class for p in voicing.positions}
            assert pitch_classes.issubset(expected_pitch_classes)
            for position in voicing.positions:
                interval = (position.pitch_class - root_pc) % 12
                assert position.role == expected_role_by_interval[interval]


def test_caged_scale_boxes_cover_five_shapes_in_order():
    scale = build_named_scale("C", "major_pentatonic")
    boxes = caged_scale_boxes("C", scale.pitch_classes)
    # Positions are numbered by fixed shape identity (E-D-C-A-G), not
    # sorted by ascending fret — "Position 1" must always be the E shape,
    # regardless of root.
    assert [b.shape for b in boxes] == SCALE_POSITION_ORDER
    # Each box must actually contain playable positions.
    for box in boxes:
        assert box.positions
        assert all(box.min_fret <= p.fret <= box.max_fret for p in box.positions)


def test_caged_scale_boxes_position_numbering_is_fixed_across_roots():
    # The whole reason for fixing the shape order is that "Position 1"
    # should mean the same shape in every key — for C specifically, the
    # unshifted A-shape and G-shape boxes land at a lower fret than the
    # E-shape box that's numbered first, which is expected: position
    # identity comes from the shape, not neck order.
    for root in ["C", "G", "D", "A", "E"]:
        scale = build_scale(root, "major")
        boxes = caged_scale_boxes(root, scale.pitch_classes)
        assert [b.shape for b in boxes] == SCALE_POSITION_ORDER


def test_caged_scale_boxes_are_a_tight_few_fret_span():
    # Matched to the ~4-6 fret, few-notes-per-string boxes real reference
    # sites use (verified against guitarscale.org's own C Major "Shapes"
    # tab) rather than a much wider window — each shape's own natural
    # spread differs, so span isn't identical across all 5, but none
    # should balloon past a realistic hand position.
    scale = build_scale("C", "major")
    boxes = caged_scale_boxes("C", scale.pitch_classes)
    for box in boxes:
        span = box.max_fret - box.min_fret
        assert 3 <= span <= 5, f"{box.shape} span={span} outside expected range"


def test_caged_scale_boxes_generalize_beyond_major():
    # The whole point of taking pitch_classes as a parameter (rather than
    # being hardcoded to major, like the CAGED chord shapes are) is that
    # every scale type the app offers must produce 5 real, playable boxes —
    # not just the pentatonic case already covered above. A full 7-note
    # scale is the densest realistic case, so every one of the 6 strings
    # should have at least one note in every box (no gaps a guitarist would
    # have to awkwardly skip over).
    scales = [
        build_scale("C", "major"),
        build_scale("C", "minor"),
        build_named_scale("C", "harmonic_minor"),
        build_named_scale("C", "melodic_minor"),
    ]
    for scale in scales:
        boxes = caged_scale_boxes("C", scale.pitch_classes)
        assert [b.shape for b in boxes] == SCALE_POSITION_ORDER
        for box in boxes:
            strings_covered = {p.string_index for p in box.positions}
            assert strings_covered == set(range(6)), f"{scale.pitch_classes} {box.shape} box missing strings"


def test_caged_scale_boxes_never_exceed_three_notes_per_string():
    # Real CAGED scale-box charts cap at 3 notes per string — a plain
    # fret-range window doesn't guarantee that (some strings can pick up
    # an extra scale tone their neighbors don't), so this checks the cap
    # actually holds across every scale type the app offers, not just one.
    scales = [
        build_scale("C", "major"),
        build_scale("C", "minor"),
        build_named_scale("C", "harmonic_minor"),
        build_named_scale("C", "melodic_minor"),
        build_named_scale("C", "major_pentatonic"),
        build_named_scale("C", "minor_pentatonic"),
        build_named_scale("C", "blues"),
    ]
    for scale in scales:
        boxes = caged_scale_boxes("C", scale.pitch_classes)
        for box in boxes:
            by_string: dict[int, int] = {}
            for p in box.positions:
                by_string[p.string_index] = by_string.get(p.string_index, 0) + 1
            assert by_string, f"{scale.pitch_classes} {box.shape} box has no positions at all"
            assert max(by_string.values()) <= 3, f"{scale.pitch_classes} {box.shape} box has a string with >3 notes"
            assert set(by_string) == set(range(6)), f"{scale.pitch_classes} {box.shape} box missing strings after capping"
