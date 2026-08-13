from chromalyze.intervals import common_interval_reference, interval_between, interval_from_semitones


def test_interval_between_covers_every_diatonic_interval_from_c():
    cases = [
        ("C", "C", 1, "perfect", 0, "P1"),
        ("C", "D", 2, "major", 2, "M2"),
        ("C", "Eb", 3, "minor", 3, "m3"),
        ("C", "E", 3, "major", 4, "M3"),
        ("C", "F", 4, "perfect", 5, "P4"),
        ("C", "G", 5, "perfect", 7, "P5"),
        ("C", "Ab", 6, "minor", 8, "m6"),
        ("C", "A", 6, "major", 9, "M6"),
        ("C", "Bb", 7, "minor", 10, "m7"),
        ("C", "B", 7, "major", 11, "M7"),
    ]
    for lower, upper, degree, quality, semitones, short_name in cases:
        result = interval_between(lower, upper)
        assert result.degree == degree
        assert result.quality == quality
        assert result.semitones == semitones
        assert result.short_name == short_name


def test_interval_between_distinguishes_enharmonic_twins_by_spelling():
    # F to B and B to F are both 6 semitones apart, but they are not the
    # same interval — letter distance (a 4th vs a 5th) is what actually
    # decides which one it is, exactly the distinction a fixed
    # semitones-only lookup could never make.
    f_to_b = interval_between("F", "B")
    assert (f_to_b.degree, f_to_b.quality, f_to_b.short_name) == (4, "augmented", "A4")

    b_to_f = interval_between("B", "F")
    assert (b_to_f.degree, b_to_f.quality, b_to_f.short_name) == (5, "diminished", "d5")

    assert f_to_b.semitones == b_to_f.semitones == 6


def test_interval_between_altered_unisons():
    assert interval_between("C", "C#").short_name == "A1"
    assert interval_between("C", "Cb").short_name == "d1"


def test_interval_between_handles_enharmonic_natural_letter_pairs():
    # E to Fb is enharmonically a unison, but letter distance makes it a
    # 2nd — spelled as a diminished 2nd, the standard real-notation answer.
    result = interval_between("E", "Fb")
    assert (result.degree, result.quality) == (2, "diminished")


def test_interval_between_wraps_augmented_seventh_correctly():
    # C to B# is enharmonically an octave (0 semitones as a pitch class),
    # but spelled as a letter-7th it's a (real, if rare) augmented 7th.
    result = interval_between("C", "B#")
    assert (result.degree, result.quality) == (7, "augmented")


def test_interval_from_semitones_covers_every_distance():
    expected_short_names = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]
    for semitones, expected in enumerate(expected_short_names):
        assert interval_from_semitones(semitones).short_name == expected


def test_interval_from_semitones_tritone_defaults_to_augmented_fourth():
    tritone = interval_from_semitones(6)
    assert tritone.name == "Augmented Fourth"
    assert tritone.degree == 4


def test_interval_from_semitones_wraps_beyond_an_octave():
    assert interval_from_semitones(12).short_name == interval_from_semitones(0).short_name == "P1"
    assert interval_from_semitones(13).short_name == interval_from_semitones(1).short_name == "m2"


def test_common_interval_reference_has_all_twelve_in_order():
    reference = common_interval_reference()
    assert [i.semitones for i in reference] == list(range(12))
    assert [i.short_name for i in reference] == [
        "P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7",
    ]
    assert [i.name for i in reference] == [
        "Perfect Unison", "Minor Second", "Major Second", "Minor Third",
        "Major Third", "Perfect Fourth", "Augmented Fourth", "Perfect Fifth",
        "Minor Sixth", "Major Sixth", "Minor Seventh", "Major Seventh",
    ]
