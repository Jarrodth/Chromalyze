from chromalyze.instruments import (
    BASS_5_STRING_STANDARD,
    BASS_STANDARD,
    GUITAR_7_STRING_STANDARD,
    GUITAR_8_STRING_STANDARD,
    GUITAR_DROP_D,
    GUITAR_STANDARD,
    Tuning,
    fretboard_positions,
)
from chromalyze.theory import build_scale


def test_guitar_standard_tuning_open_strings():
    # E A D G B E, low to high, as pitch classes.
    assert GUITAR_STANDARD.open_string_pitch_classes == [4, 9, 2, 7, 11, 4]


def test_guitar_drop_d_only_changes_the_low_string():
    assert GUITAR_DROP_D.open_string_pitch_classes == [2, 9, 2, 7, 11, 4]
    assert GUITAR_DROP_D.open_string_pitch_classes[1:] == GUITAR_STANDARD.open_string_pitch_classes[1:]


def test_extended_range_guitars_have_the_right_string_count_and_open_notes():
    assert len(GUITAR_7_STRING_STANDARD.open_string_pitch_classes) == 7
    assert GUITAR_7_STRING_STANDARD.open_string_pitch_classes == [11, 4, 9, 2, 7, 11, 4]  # B E A D G B E

    assert len(GUITAR_8_STRING_STANDARD.open_string_pitch_classes) == 8
    assert GUITAR_8_STRING_STANDARD.open_string_pitch_classes == [6, 11, 4, 9, 2, 7, 11, 4]  # F# B E A D G B E


def test_bass_tunings():
    assert BASS_STANDARD.open_string_pitch_classes == [4, 9, 2, 7]  # E A D G
    assert BASS_5_STRING_STANDARD.open_string_pitch_classes == [11, 4, 9, 2, 7]  # B E A D G


def test_custom_tuning_from_note_names():
    custom = Tuning.from_note_names("My Tuning", ["D", "A", "D", "F#", "A", "D"])  # Open D
    assert custom.open_string_pitch_classes == [2, 9, 2, 6, 9, 2]


def test_fretboard_positions_low_e_string_c_major():
    scale = build_scale("C", "major")
    positions = fretboard_positions(scale.pitch_classes, GUITAR_STANDARD, num_frets=12)
    low_e_frets = sorted(p.fret for p in positions if p.string_index == 0)
    # Well-known real fretboard positions for C major on a standard-tuned
    # low E string.
    assert low_e_frets == [0, 1, 3, 5, 7, 8, 10, 12]


def test_fretboard_positions_degree_matches_scale_order():
    scale = build_scale("C", "major")
    positions = fretboard_positions(scale.pitch_classes, GUITAR_STANDARD, num_frets=12)
    # Open low E (pitch class 4) is the 3rd note of the C major scale (C D
    # [E] F G A B).
    open_low_e = next(p for p in positions if p.string_index == 0 and p.fret == 0)
    assert open_low_e.scale_degree == 3


def test_fretboard_positions_respects_num_frets():
    scale = build_scale("C", "major")
    positions = fretboard_positions(scale.pitch_classes, GUITAR_STANDARD, num_frets=5)
    assert all(p.fret <= 5 for p in positions)


def test_fretboard_positions_on_drop_d_shifts_only_the_low_string():
    scale = build_scale("D", "major")
    standard_positions = fretboard_positions(scale.pitch_classes, GUITAR_STANDARD, num_frets=12)
    drop_d_positions = fretboard_positions(scale.pitch_classes, GUITAR_DROP_D, num_frets=12)

    # Strings 1-5 (index 1+, i.e. everything but the lowest) are identical
    # between standard and drop D tuning — only the low string's open note
    # changed.
    standard_upper = sorted((p.string_index, p.fret) for p in standard_positions if p.string_index != 0)
    drop_d_upper = sorted((p.string_index, p.fret) for p in drop_d_positions if p.string_index != 0)
    assert standard_upper == drop_d_upper
