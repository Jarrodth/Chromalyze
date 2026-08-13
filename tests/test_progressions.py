from chromalyze.progressions import NAMED_PROGRESSIONS, identify_progression, realize_progression


def test_realize_pop_progression_in_c_major():
    chords = realize_progression(NAMED_PROGRESSIONS["pop"], "C")
    assert [(c.roman_numeral, c.root, c.quality) for c in chords] == [
        ("I", "C", "major"),
        ("V", "G", "major"),
        ("vi", "A", "minor"),
        ("IV", "F", "major"),
    ]
    # Real spelled chord tones, not just roots:
    assert chords[2].notes == ["A", "C", "E"]


def test_realize_pop_progression_transposes_to_a_different_key():
    # Same shape (I-V-vi-IV), realized in G major instead of C major.
    chords = realize_progression(NAMED_PROGRESSIONS["pop"], "G")
    assert [(c.roman_numeral, c.root, c.quality) for c in chords] == [
        ("I", "G", "major"),
        ("V", "D", "major"),
        ("vi", "E", "minor"),
        ("IV", "C", "major"),
    ]


def test_realize_jazz_ii_v_i_uses_seventh_chords():
    chords = realize_progression(NAMED_PROGRESSIONS["jazz_ii_v_i"], "C")
    assert [(c.roman_numeral, c.root, c.quality) for c in chords] == [
        ("ii7", "D", "minor7"),
        ("V7", "G", "dominant7"),
        ("Imaj7", "C", "major7"),
    ]


def test_realize_andalusian_cadence_in_a_minor():
    # The famous Am-G-F-E progression: VII and VI are natural minor's own
    # chords, but the closing V is a borrowed/altered major dominant (E
    # major, with G#) rather than natural minor's own (diatonic, minor)
    # v chord — real Andalusian-cadence harmony, not a simplification of it.
    chords = realize_progression(NAMED_PROGRESSIONS["andalusian_cadence"], "A")
    assert [(c.roman_numeral, c.root, c.quality) for c in chords] == [
        ("i", "A", "minor"),
        ("VII", "G", "major"),
        ("VI", "F", "major"),
        ("V", "E", "major"),
    ]
    assert chords[3].notes == ["E", "G#", "B"]


def test_realize_twelve_bar_blues_is_all_dominant_sevenths():
    chords = realize_progression(NAMED_PROGRESSIONS["twelve_bar_blues"], "C")
    assert len(chords) == 12
    assert all(c.quality == "dominant7" for c in chords)
    assert [c.root for c in chords] == ["C", "C", "C", "C", "F", "F", "C", "C", "G", "F", "C", "G"]


def test_identify_progression_recognizes_pop_progression_in_c_major():
    chords = [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")]
    matches = identify_progression(chords, key_tonic="C", key_mode="major")
    assert "pop" in matches


def test_identify_progression_distinguishes_similar_progressions():
    # I-vi-IV-V ('50s) is a real permutation of pop's chords but a
    # different progression — identify_progression must not conflate them.
    fifties_chords = [("C", "major"), ("A", "minor"), ("F", "major"), ("G", "major")]
    matches = identify_progression(fifties_chords, key_tonic="C", key_mode="major")
    assert "fifties" in matches
    assert "pop" not in matches


def test_identify_progression_returns_empty_for_no_match():
    random_chords = [("C", "major"), ("Db", "major"), ("F#", "minor")]
    assert identify_progression(random_chords, key_tonic="C", key_mode="major") == []


def test_every_named_progression_realizes_without_error():
    # Cheap but effective completeness check: every catalog entry must
    # actually build real chords for a real tonic, not just look right on
    # paper — most of the catalog is major-context, but the minor ones
    # (e.g. andalusian_cadence, minor_pop) must also work when realized.
    for name, progression in NAMED_PROGRESSIONS.items():
        chords = realize_progression(progression, "C" if progression.mode == "major" else "A")
        assert len(chords) == len(progression.steps), name
        for chord in chords:
            assert chord.notes, name
