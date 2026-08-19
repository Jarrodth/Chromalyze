from chromalyze.instruments import (
    BASS_5_STRING_STANDARD,
    BASS_STANDARD,
    GUITAR_7_STRING_STANDARD,
    GUITAR_8_STRING_STANDARD,
    GUITAR_DROP_D,
    GUITAR_OPEN_G,
    GUITAR_STANDARD,
    Tuning,
    fretboard_positions,
    practical_chord_voicing,
    practical_power_chord_voicing,
)
from chromalyze.theory import NOTE_NAME_TO_PITCH_CLASS, build_chord, build_power_chord, build_scale


def _shape_string(positions, string_count):
    """Renders a voicing as real chord-chart shorthand, e.g. "X32010" for
    open C major — low string first, "X" for a muted/unplayed string —
    so test assertions read the same way any real chord chart does.
    """
    by_string = {p.string_index: p.fret for p in positions}
    return "".join(str(by_string[i]) if i in by_string else "X" for i in range(string_count))


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


def _voicing_for(root, quality):
    chord = build_chord(root, quality)
    root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
    return practical_chord_voicing(chord.pitch_classes, root_pc, GUITAR_STANDARD)


def test_practical_chord_voicing_open_c_major():
    # The textbook open C major chord: X-3-2-0-1-0.
    positions = _voicing_for("C", "major")
    assert _shape_string(positions, 6) == "X32010"


def test_practical_chord_voicing_open_a_minor():
    # The textbook open A minor chord: X-0-2-2-1-0.
    positions = _voicing_for("A", "minor")
    assert _shape_string(positions, 6) == "X02210"


def test_practical_chord_voicing_open_e_minor():
    # The textbook open E minor chord: 0-2-2-0-0-0.
    positions = _voicing_for("E", "minor")
    assert _shape_string(positions, 6) == "022000"


def test_practical_chord_voicing_open_d_major():
    # The textbook open D major chord: X-X-0-2-3-2.
    positions = _voicing_for("D", "major")
    assert _shape_string(positions, 6) == "XX0232"


def test_practical_chord_voicing_open_g_major():
    # The textbook open G major chord: 3-2-0-0-0-3 (low position — not the
    # common "3-2-0-0-3-3" barred variant some players use instead, since
    # this algorithm always prefers the lowest fret available per string).
    positions = _voicing_for("G", "major")
    assert _shape_string(positions, 6) == "320003"


def test_practical_chord_voicing_root_always_in_the_bass():
    for root in ["C", "D", "E", "F", "G", "A", "B"]:
        for quality in ["major", "minor"]:
            positions = _voicing_for(root, quality)
            root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
            bass_position = min(positions, key=lambda p: p.string_index)
            assert bass_position.pitch_class == root_pc, f"{root} {quality}: bass note wasn't the root"


def test_practical_chord_voicing_covers_every_chord_tone():
    chord = build_chord("F", "major")
    root_pc = NOTE_NAME_TO_PITCH_CLASS["F"]
    positions = practical_chord_voicing(chord.pitch_classes, root_pc, GUITAR_STANDARD)
    assert {p.pitch_class for p in positions} == set(chord.pitch_classes)


def test_practical_chord_voicing_on_bass_tuning():
    # E muted, A open (root), D fret 2 (fifth), G fret 2 (root octave) — a
    # real, practical bass voicing, not just any pitch-correct combination.
    chord = build_chord("A", "minor")
    root_pc = NOTE_NAME_TO_PITCH_CLASS["A"]
    positions = practical_chord_voicing(chord.pitch_classes, root_pc, BASS_STANDARD)
    assert _shape_string(positions, 4) == "X022"


def test_build_power_chord_is_correctly_spelled_root_and_fifth():
    # A power chord is neither major nor minor — just the root and its
    # perfect fifth. Real letter spelling matters here too: the fifth is
    # 4 letters above the root (C-D-E-F-G), not the 2-letter tertian jump
    # build_chord uses for triads, so this checks it lands on the right
    # letter, not just the right pitch class.
    cases = {"C": ("C", "G"), "D": ("D", "A"), "F": ("F", "C"), "F#": ("F#", "C#"), "Bb": ("Bb", "F")}
    for root, expected_notes in cases.items():
        chord = build_power_chord(root)
        assert chord.quality == "power"
        assert tuple(chord.notes) == expected_notes
        root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
        assert chord.pitch_classes == [root_pc, (root_pc + 7) % 12]


def test_practical_power_chord_voicing_matches_real_canonical_shapes():
    # The two movable shapes every guitarist already knows by heart —
    # rooted on the low E string or the A string, never higher (a power
    # chord voiced on, say, the G or B string, however low the fret, isn't
    # how anyone actually plays one).
    expected = {
        "E": (0, 0, 1, 2),  # open E5: low E open, A string 2nd fret
        "F": (0, 1, 1, 3),  # movable F5: low E 1st fret, A string 3rd fret
        "G": (0, 3, 1, 5),  # movable G5: low E 3rd fret, A string 5th fret
        "A": (1, 0, 2, 2),  # open A5: A open, D string 2nd fret
        "B": (1, 2, 2, 4),  # movable B5: A string 2nd fret, D string 4th fret
        "C": (1, 3, 2, 5),  # movable C5: A string 3rd fret, D string 5th fret
        "D": (1, 5, 2, 7),  # movable D5: A string 5th fret, D string 7th fret
    }
    for root, (root_string, root_fret, fifth_string, fifth_fret) in expected.items():
        root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
        positions = practical_power_chord_voicing(root_pc, GUITAR_STANDARD)
        assert len(positions) == 2
        by_string = {p.string_index: p.fret for p in positions}
        assert by_string == {root_string: root_fret, fifth_string: fifth_fret}, root


def test_practical_power_chord_voicing_only_ever_two_notes():
    # Real power chords mute everything except root and fifth — unlike
    # practical_chord_voicing's full-hand-span approach (right for real
    # triads, wrong here), this should never ring more than 2 strings.
    for root in ["C", "D", "E", "F", "G", "A", "B", "F#", "Bb", "Eb"]:
        root_pc = NOTE_NAME_TO_PITCH_CLASS[root]
        positions = practical_power_chord_voicing(root_pc, GUITAR_STANDARD)
        assert len(positions) == 2
        pitch_classes = {p.pitch_class for p in positions}
        assert pitch_classes == {root_pc, (root_pc + 7) % 12}
        # The two notes must be on adjacent strings, root on the lower one.
        strings = sorted(p.string_index for p in positions)
        assert strings[1] == strings[0] + 1
        root_position = next(p for p in positions if p.pitch_class == root_pc)
        assert root_position.string_index == strings[0]


def test_practical_power_chord_voicing_never_roots_above_the_low_two_strings():
    # The actual regression this guards: every root, for every pitch
    # class, must land on the low E string or the A string — never a
    # higher string, no matter how low a fret that might reach.
    for root_pc in range(12):
        positions = practical_power_chord_voicing(root_pc, GUITAR_STANDARD)
        root_position = min(positions, key=lambda p: p.string_index)
        assert root_position.string_index in (0, 1), root_pc


def test_practical_power_chord_voicing_handles_a_non_standard_string_gap():
    # Open G tuning's second string pair (G to D) is a perfect 5th, not
    # the usual perfect 4th — the fret-offset-to-the-fifth math still
    # needs to come from the tuning's real interval, not a hardcoded +2,
    # even though the G-string-specific case standard tuning used to
    # exercise is no longer reachable now the root search is restricted
    # to the low two strings.
    root_pc = NOTE_NAME_TO_PITCH_CLASS["A"]  # A5, rooted on open-G-tuning's 2nd string (G)
    positions = practical_power_chord_voicing(root_pc, GUITAR_OPEN_G)
    by_string = {p.string_index: p.fret for p in positions}
    # String 1 (G, pitch class 7) to string 2 (D, pitch class 2) is a
    # perfect 5th (7 semitones) above — exactly the same interval as
    # root-to-fifth, so the fifth lands on the *same* fret as the root
    # (2nd fret: G+2=A the root, D+2=E the fifth), not the usual +2 more.
    assert by_string == {1: 2, 2: 2}
