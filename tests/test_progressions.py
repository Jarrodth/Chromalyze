from chromalyze.chords import ChordSegment
from chromalyze.progressions import (
    NAMED_PROGRESSIONS,
    best_progression_match,
    clean_chords_with_detected_loop,
    clean_chords_with_progression,
    identify_progression,
    realize_progression,
    resolve_quality_oscillation,
)


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


def test_best_progression_match_finds_pop_progression_looped():
    # Pop progression (I-V-vi-IV) played twice in a row in C major.
    chords = [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")] * 2
    match = best_progression_match(chords, key_tonic="C", key_mode="major")
    assert match is not None
    assert match.name == "pop"
    assert match.phase == 0
    assert match.match_ratio == 1.0


def test_best_progression_match_finds_the_right_starting_phase():
    # Same loop, but the real sequence starts mid-loop, on the V chord.
    chords = [("G", "major"), ("A", "minor"), ("F", "major"), ("C", "major")] * 2
    match = best_progression_match(chords, key_tonic="C", key_mode="major")
    assert match is not None
    assert match.name == "pop"
    assert match.phase == 1


def test_best_progression_match_tolerates_a_few_real_misreads():
    chords = [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")] * 3
    # Corrupt one entry, as a stand-in for a real chroma misread — the
    # overall loop should still be recognizable.
    chords[5] = ("Db", "minor")
    match = best_progression_match(chords, key_tonic="C", key_mode="major")
    assert match is not None
    assert match.name == "pop"


def test_best_progression_match_returns_none_for_unstructured_chords():
    chords = [("C", "major"), ("Db", "major"), ("F#", "minor"), ("B", "major")]
    assert best_progression_match(chords, key_tonic="C", key_mode="major") is None


def test_best_progression_match_returns_none_for_empty_input():
    assert best_progression_match([], key_tonic="C", key_mode="major") is None


def _segment(start, end, chord, confidence):
    return ChordSegment(start=start, end=end, chord=chord, correlation=0.9, confidence=confidence)


def test_clean_chords_with_progression_corrects_a_low_confidence_outlier():
    # Pop progression (C-G-Am-F) played twice, but the 7th segment (should
    # be Am) was misdetected as a low-confidence Ab.
    segments = [
        _segment(0.0, 1.0, "C", 0.5),
        _segment(1.0, 2.0, "G", 0.5),
        _segment(2.0, 3.0, "Am", 0.5),
        _segment(3.0, 4.0, "F", 0.5),
        _segment(4.0, 5.0, "C", 0.5),
        _segment(5.0, 6.0, "G", 0.5),
        _segment(6.0, 7.0, "Ab", 0.01),  # low confidence, doesn't fit the loop
        _segment(7.0, 8.0, "F", 0.5),
    ]

    result = clean_chords_with_progression(segments, key_tonic="C", key_mode="major")

    assert any(m.name == "pop" for m in result.matches)
    corrected = [s for s in result.chords if s.progression_corrected]
    assert len(corrected) == 1
    assert corrected[0].chord == "Am"
    assert corrected[0].start == 6.0 and corrected[0].end == 7.0


def test_clean_chords_with_progression_leaves_confident_deviations_alone():
    segments = [
        _segment(0.0, 1.0, "C", 0.5),
        _segment(1.0, 2.0, "G", 0.5),
        _segment(2.0, 3.0, "Am", 0.5),
        _segment(3.0, 4.0, "F", 0.5),
        _segment(4.0, 5.0, "C", 0.5),
        _segment(5.0, 6.0, "G", 0.5),
        _segment(6.0, 7.0, "Ab", 0.4),  # confident — treated as a genuine deviation, not corrected
        _segment(7.0, 8.0, "F", 0.5),
    ]

    result = clean_chords_with_progression(segments, key_tonic="C", key_mode="major")

    assert all(not s.progression_corrected for s in result.chords)
    assert [s.chord for s in result.chords] == ["C", "G", "Am", "F", "C", "G", "Ab", "F"]


def test_clean_chords_with_progression_merges_a_corrected_segment_into_its_neighbors():
    # The 12-bar blues' first four bars are all the same chord (I7,
    # collapsed to a plain "C" triad label) — a real detector splitting
    # that stretch into several small windows, with one of them misread,
    # is exactly the "buggy jumpy chord display" case this is meant to fix.
    labels = ["C", "C", "Db", "C", "F", "F", "C", "C", "G", "F", "C", "G"]
    segments = [
        _segment(float(i), float(i + 1), label, 0.01 if label == "Db" else 0.5) for i, label in enumerate(labels)
    ]

    result = clean_chords_with_progression(segments, key_tonic="C", key_mode="major")

    assert any(m.name == "twelve_bar_blues" for m in result.matches)
    assert result.chords[0].chord == "C"
    assert result.chords[0].start == 0.0
    assert result.chords[0].end == 4.0
    assert result.chords[0].progression_corrected is True


def test_clean_chords_with_progression_returns_none_match_when_nothing_fits():
    segments = [
        _segment(0.0, 1.0, "C", 0.9),
        _segment(1.0, 2.0, "Db", 0.9),
        _segment(2.0, 3.0, "F#m", 0.9),
    ]
    result = clean_chords_with_progression(segments, key_tonic="C", key_mode="major")
    assert result.matches == []
    assert result.chords == segments


def test_clean_chords_with_progression_handles_empty_input():
    result = clean_chords_with_progression([], key_tonic="C", key_mode="major")
    assert result.chords == []
    assert result.matches == []


def test_clean_chords_with_progression_handles_a_long_sectional_sequence():
    # A song where the first half loops the pop progression and the
    # second half loops the '50s progression — no single progression
    # explains the whole thing, but each half should still get its own
    # chunk-local corrections.
    pop_labels = ["C", "G", "Am", "F"] * 4  # 16 segments, "Am" at index 2, 6, 10, 14
    fifties_labels = ["C", "Am", "F", "G"] * 4  # 16 segments, "Am" at index 1, 5, 9, 13
    labels = pop_labels + fifties_labels
    # Corrupt one low-confidence "Am" segment in each half.
    labels[6] = "Db"  # pop half
    labels[16 + 5] = "Eb"  # fifties half

    segments = [
        _segment(float(i), float(i + 1), label, 0.01 if i in (6, 21) else 0.5) for i, label in enumerate(labels)
    ]

    result = clean_chords_with_progression(segments, key_tonic="C", key_mode="major", window_size=16)

    assert {m.name for m in result.matches} == {"pop", "fifties"}
    corrected = {s.start: s.chord for s in result.chords if s.progression_corrected}
    assert corrected == {6.0: "Am", 21.0: "Am"}


def test_resolve_quality_oscillation_collapses_a_flip_flopping_root():
    # F is diatonically major in C major (IV) — a power-chord ambiguity
    # flip-flopping between F and Fm should collapse to the diatonic F.
    segments = [
        _segment(0.0, 1.0, "C", 0.5),
        _segment(1.0, 2.0, "F", 0.02),
        _segment(2.0, 3.0, "Fm", 0.02),
        _segment(3.0, 4.0, "F", 0.02),
        _segment(4.0, 5.0, "Fm", 0.02),
        _segment(5.0, 6.0, "G", 0.5),
    ]

    result = resolve_quality_oscillation(segments, key_tonic="C", key_mode="major")

    assert [s.chord for s in result] == ["C", "F", "G"]
    assert result[1].start == 1.0 and result[1].end == 5.0
    assert result[1].quality_resolved is True


def test_resolve_quality_oscillation_can_resolve_to_minor():
    # A is the tonic of A minor — diatonically minor (i), not major — so
    # oscillation on the tonic root should resolve to "Am", not "A". Not
    # a blanket "prefer major" rule: whichever reading is actually the
    # key's own diatonic triad wins, in either direction.
    segments = [
        _segment(0.0, 1.0, "D", 0.5),
        _segment(1.0, 2.0, "Am", 0.02),
        _segment(2.0, 3.0, "A", 0.02),
        _segment(3.0, 4.0, "Am", 0.02),
        _segment(4.0, 5.0, "E", 0.5),
    ]

    result = resolve_quality_oscillation(segments, key_tonic="A", key_mode="minor")

    assert [s.chord for s in result] == ["D", "Am", "E"]


def test_resolve_quality_oscillation_leaves_a_confident_segment_alone():
    segments = [
        _segment(0.0, 1.0, "F", 0.02),
        _segment(1.0, 2.0, "Fm", 0.4),  # confident — a genuine reading, not touched
        _segment(2.0, 3.0, "F", 0.02),
    ]

    result = resolve_quality_oscillation(segments, key_tonic="C", key_mode="major")

    assert [s.chord for s in result] == ["F", "Fm", "F"]
    assert result[1].quality_resolved is False


def test_resolve_quality_oscillation_falls_back_to_majority_for_a_chromatic_root():
    # Db is diatonic in C major as neither major nor minor — nothing to
    # prefer on theory grounds, so majority (2x major vs 1x minor) wins.
    segments = [
        _segment(0.0, 1.0, "C", 0.5),
        _segment(1.0, 2.0, "Db", 0.02),
        _segment(2.0, 3.0, "Dbm", 0.02),
        _segment(3.0, 4.0, "Db", 0.02),
        _segment(4.0, 5.0, "G", 0.5),
    ]

    result = resolve_quality_oscillation(segments, key_tonic="C", key_mode="major")

    assert [s.chord for s in result] == ["C", "Db", "G"]


def test_resolve_quality_oscillation_leaves_a_consistent_run_alone():
    segments = [
        _segment(0.0, 1.0, "F", 0.02),
        _segment(1.0, 2.0, "F", 0.02),
        _segment(2.0, 3.0, "G", 0.5),
    ]

    result = resolve_quality_oscillation(segments, key_tonic="C", key_mode="major")

    assert [s.chord for s in result] == ["F", "G"]
    assert all(not s.quality_resolved for s in result)


def test_resolve_quality_oscillation_handles_short_input():
    segments = [_segment(0.0, 1.0, "C", 0.5)]
    assert resolve_quality_oscillation(segments, key_tonic="C", key_mode="major") == segments


def test_clean_chords_with_detected_loop_finds_a_non_catalog_loop():
    # D-Bm-G-A repeated 4 times — not one of NAMED_PROGRESSIONS' 12
    # entries, so clean_chords_with_progression couldn't touch this, but
    # it's still a real, self-consistent repeating loop.
    labels = ["D", "Bm", "G", "A"] * 4
    labels[9] = "Eb"  # was "Bm"

    segments = [_segment(float(i), float(i + 1), label, 0.01 if i == 9 else 0.5) for i, label in enumerate(labels)]

    result = clean_chords_with_detected_loop(segments)

    assert len(result.loops) == 1
    assert result.loops[0].period == 4
    corrected = [s for s in result.chords if s.loop_corrected]
    assert len(corrected) == 1
    assert corrected[0].chord == "Bm"
    assert corrected[0].start == 9.0


def test_clean_chords_with_detected_loop_prefers_the_shortest_period():
    # A clean period-2 loop is also technically "explained" by period 4,
    # 6, or 8 (same content, just re-described with a longer cycle) — the
    # shortest period that reaches the best ratio should win.
    labels = ["C", "G"] * 8
    labels[5] = "Ab"  # was "G"
    segments = [_segment(float(i), float(i + 1), label, 0.01 if i == 5 else 0.5) for i, label in enumerate(labels)]

    result = clean_chords_with_detected_loop(segments)

    assert len(result.loops) == 1
    assert result.loops[0].period == 2
    corrected = [s for s in result.chords if s.loop_corrected]
    assert len(corrected) == 1
    assert corrected[0].chord == "G"


def test_clean_chords_with_detected_loop_leaves_confident_deviations_alone():
    labels = ["D", "Bm", "G", "A"] * 4
    labels[9] = "Eb"
    segments = [_segment(float(i), float(i + 1), label, 0.4 if i == 9 else 0.5) for i, label in enumerate(labels)]

    result = clean_chords_with_detected_loop(segments)

    assert all(not s.loop_corrected for s in result.chords)
    assert result.chords[9].chord == "Eb"


def test_clean_chords_with_detected_loop_returns_nothing_for_unstructured_chords():
    labels = ["C", "D", "E", "F", "G", "A", "B", "C#", "D#", "F#", "Bb", "Eb"]  # no repetition at all
    segments = [_segment(float(i), float(i + 1), label, 0.01) for i, label in enumerate(labels)]

    result = clean_chords_with_detected_loop(segments)

    assert result.loops == []
    assert [s.chord for s in result.chords] == labels


def test_clean_chords_with_detected_loop_handles_short_input():
    segments = [_segment(0.0, 1.0, "C", 0.5), _segment(1.0, 2.0, "G", 0.5)]
    result = clean_chords_with_detected_loop(segments)
    assert result.chords == segments
    assert result.loops == []


def test_clean_chords_with_detected_loop_corrects_across_overlapping_windows():
    # A loop much longer than one window (40 segments vs the default
    # window_size of 16) — this only exercises the sliding-window voting
    # machinery (several overlapping windows, not just one, have to
    # agree) rather than one window covering everything in a single shot.
    labels = ["D", "Bm", "G", "A"] * 10
    labels[22] = "Eb"  # was "G" (22 % 4 == 2)

    segments = [_segment(float(i), float(i + 1), label, 0.01 if i == 22 else 0.5) for i, label in enumerate(labels)]

    result = clean_chords_with_detected_loop(segments)

    corrected = [s for s in result.chords if s.loop_corrected]
    assert len(corrected) == 1
    assert corrected[0].chord == "G"
    assert corrected[0].start == 22.0
