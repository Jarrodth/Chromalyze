from chromalyze.caged import CAGED_ORDER, CAGED_SHAPES, caged_chord_shapes, caged_scale_boxes
from chromalyze.scales import build_named_scale


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
    assert [b.shape for b in boxes] == CAGED_ORDER
    # Each box must actually contain playable positions.
    for box in boxes:
        assert box.positions
        assert all(box.min_fret <= p.fret <= box.max_fret for p in box.positions)
    # Boxes must be in ascending order up the neck.
    assert [b.min_fret for b in boxes] == sorted(b.min_fret for b in boxes)
