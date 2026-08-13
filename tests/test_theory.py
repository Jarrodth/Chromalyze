from chromalyze.theory import (
    analyze_chord_function,
    build_scale,
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
