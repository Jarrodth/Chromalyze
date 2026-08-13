from chromalyze.scales import build_named_scale
from chromalyze.theory import diatonic_triads


def test_major_pentatonic_c():
    scale = build_named_scale("C", "major_pentatonic")
    assert scale.notes == ["C", "D", "E", "G", "A"]
    assert scale.pitch_classes == [0, 2, 4, 7, 9]


def test_minor_pentatonic_a():
    scale = build_named_scale("A", "minor_pentatonic")
    assert scale.notes == ["A", "C", "D", "E", "G"]
    assert scale.pitch_classes == [9, 0, 2, 4, 7]


def test_blues_scale_c():
    # The classic C blues scale, including the reused "G" letter for both
    # the blue note (Gb) and the real 5th (G) — a blues scale isn't
    # diatonic, so unlike a 7-note scale there's no rule against that.
    scale = build_named_scale("C", "blues")
    assert scale.notes == ["C", "Eb", "F", "Gb", "G", "Bb"]
    assert scale.pitch_classes == [0, 3, 5, 6, 7, 10]


def test_harmonic_minor_a():
    # A harmonic minor: natural minor with a raised 7th (G -> G#).
    scale = build_named_scale("A", "harmonic_minor")
    assert scale.notes == ["A", "B", "C", "D", "E", "F", "G#"]
    assert scale.pitch_classes == [9, 11, 0, 2, 4, 5, 8]


def test_melodic_minor_a():
    # A melodic (jazz/ascending) minor: natural minor with a raised 6th
    # and 7th (F -> F#, G -> G#).
    scale = build_named_scale("A", "melodic_minor")
    assert scale.notes == ["A", "B", "C", "D", "E", "F#", "G#"]
    assert scale.pitch_classes == [9, 11, 0, 2, 4, 6, 8]


def test_harmonic_minor_diatonic_triads_have_the_signature_augmented_biii_and_major_v():
    # Harmonic minor's raised 7th is exactly what gives it real harmonic
    # function natural minor doesn't have: a proper major-quality V (not
    # natural minor's v) and a distinctive augmented triad on bIII. This
    # is diatonic_triads (from theory.py) running on a Scale it was never
    # written with in mind — proof the generic stack-and-classify approach
    # really does generalize beyond the 7 modes it was built for.
    scale = build_named_scale("A", "harmonic_minor")
    triads = diatonic_triads(scale)
    qualities = [t.quality for t in triads]
    assert qualities == ["minor", "diminished", "augmented", "minor", "major", "major", "diminished"]
    numerals = [t.roman_numeral for t in triads]
    assert numerals == ["i", "ii°", "III+", "iv", "V", "VI", "vii°"]
