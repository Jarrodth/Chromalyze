from chromalyze.theory import (
    analyze_chord_function,
    build_chord,
    build_scale,
    diatonic_sevenths,
    diatonic_triads,
    parallel_key,
    relative_key,
    spell_scale,
)


def test_spell_scale_major_keys_use_correct_accidentals():
    # Each of these must use every letter name exactly once, with the
    # accidental that real key signatures use — not just "a note with the
    # right pitch class", which could reuse a letter (e.g. G major must be
    # spelled with F#, never Gb, since Gb would repeat the letter G).
    assert spell_scale("C", "major") == ["C", "D", "E", "F", "G", "A", "B"]
    assert spell_scale("G", "major") == ["G", "A", "B", "C", "D", "E", "F#"]
    assert spell_scale("F", "major") == ["F", "G", "A", "Bb", "C", "D", "E"]
    assert spell_scale("D", "major") == ["D", "E", "F#", "G", "A", "B", "C#"]
    assert spell_scale("Bb", "major") == ["Bb", "C", "D", "Eb", "F", "G", "A"]


def test_spell_scale_minor_keys_share_their_relative_majors_signature():
    # E minor shares G major's key signature (1 sharp, F#).
    assert spell_scale("E", "minor") == ["E", "F#", "G", "A", "B", "C", "D"]
    # C# minor shares E major's key signature (4 sharps).
    assert spell_scale("C#", "minor") == ["C#", "D#", "E", "F#", "G#", "A", "B"]


def test_build_scale_pitch_classes_match_notes():
    scale = build_scale("D", "major")
    assert len(scale.pitch_classes) == len(scale.notes) == 7
    # D major: D E F# G A B C#
    assert scale.pitch_classes == [2, 4, 6, 7, 9, 11, 1]


def test_diatonic_triads_c_major():
    scale = build_scale("C", "major")
    triads = diatonic_triads(scale)

    expected = [
        (1, "C", "major", "I"),
        (2, "D", "minor", "ii"),
        (3, "E", "minor", "iii"),
        (4, "F", "major", "IV"),
        (5, "G", "major", "V"),
        (6, "A", "minor", "vi"),
        (7, "B", "diminished", "vii°"),
    ]
    actual = [(t.degree, t.root, t.quality, t.roman_numeral) for t in triads]
    assert actual == expected


def test_diatonic_triads_a_minor():
    scale = build_scale("A", "minor")
    triads = diatonic_triads(scale)

    # Natural minor: i ii° III iv v VI VII
    qualities = [t.quality for t in triads]
    assert qualities == ["minor", "diminished", "major", "minor", "minor", "major", "major"]


def test_relative_key_round_trips():
    assert relative_key("C", "major") == ("A", "aeolian")
    assert relative_key("A", "minor") == ("C", "ionian")
    assert relative_key("G", "major") == ("E", "aeolian")


def test_parallel_key():
    assert parallel_key("C", "major") == ("C", "aeolian")
    assert parallel_key("C", "minor") == ("C", "ionian")


def test_analyze_chord_function_diatonic_chords_in_c_major():
    cases = [
        ("C", "major", "I"),
        ("D", "minor", "ii"),
        ("E", "minor", "iii"),
        ("F", "major", "IV"),
        ("G", "major", "V"),
        ("A", "minor", "vi"),
    ]
    for root, quality, expected_numeral in cases:
        fn = analyze_chord_function(root, quality, "C", "major")
        assert fn.roman_numeral == expected_numeral
        assert fn.is_diatonic is True


def test_analyze_chord_function_borrowed_chord_in_c_major():
    # Bb major is a semitone below the diatonic vii/B — a very common
    # "modal mixture" borrowed chord (bVII), not part of C major's own
    # diatonic set.
    fn = analyze_chord_function("Bb", "major", "C", "major")
    assert fn.roman_numeral == "bVII"
    assert fn.is_diatonic is False


def test_build_chord_spells_common_seventh_chords_correctly():
    # Each case is a real, standard chord spelling — the letter-stacking
    # approach must reuse each letter's normal "skip one" pattern (root,
    # 3rd, 5th, 7th), never reusing a letter or defaulting to a "simpler"
    # enharmonic spelling.
    assert build_chord("C", "major7").notes == ["C", "E", "G", "B"]
    assert build_chord("G", "dominant7").notes == ["G", "B", "D", "F"]
    assert build_chord("D", "minor7").notes == ["D", "F", "A", "C"]
    assert build_chord("B", "half-diminished7").notes == ["B", "D", "F", "A"]
    # A fully diminished 7th stacks minor thirds all the way up — B D F Ab,
    # not B D F G#, since G# would break the letter-per-third pattern.
    assert build_chord("B", "diminished7").notes == ["B", "D", "F", "Ab"]
    assert build_chord("C", "minor-major7").notes == ["C", "Eb", "G", "B"]


def test_build_chord_pitch_classes_match_notes():
    chord = build_chord("D", "minor7")
    assert chord.pitch_classes == [2, 5, 9, 0]


def test_diatonic_sevenths_c_major():
    scale = build_scale("C", "major")
    sevenths = diatonic_sevenths(scale)

    expected = [
        (1, "C", "major7", "Imaj7"),
        (2, "D", "minor7", "ii7"),
        (3, "E", "minor7", "iii7"),
        (4, "F", "major7", "IVmaj7"),
        (5, "G", "dominant7", "V7"),
        (6, "A", "minor7", "vi7"),
        (7, "B", "half-diminished7", "viiø7"),
    ]
    actual = [(s.degree, s.root, s.quality, s.roman_numeral) for s in sevenths]
    assert actual == expected


def test_diatonic_sevenths_a_natural_minor():
    scale = build_scale("A", "minor")
    sevenths = diatonic_sevenths(scale)

    # Natural minor (no raised leading tone): i7 iiø7 IIImaj7 iv7 v7 VImaj7 VII7
    qualities = [s.quality for s in sevenths]
    assert qualities == [
        "minor7", "half-diminished7", "major7", "minor7", "minor7", "major7", "dominant7",
    ]
    numerals = [s.roman_numeral for s in sevenths]
    assert numerals == ["i7", "iiø7", "IIImaj7", "iv7", "v7", "VImaj7", "VII7"]


def test_analyze_chord_function_handles_seventh_chords():
    fn = analyze_chord_function("G", "dominant7", "C", "major")
    assert fn.roman_numeral == "V7"
    assert fn.is_diatonic is False  # analyze_chord_function checks against diatonic *triads*, not sevenths
