from chromalyze.chords import detect_chords
from chromalyze.preprocessing import load_audio


def test_detect_chords_progression(chord_progression_clip):
    path, expected_chords, seconds_per_chord = chord_progression_clip
    y, sr = load_audio(path)
    segments = detect_chords(y, sr, segment_seconds=0.5)

    detected_chords = [s.chord for s in segments]
    assert detected_chords == expected_chords


def test_detect_chords_segment_boundaries_match_known_progression(chord_progression_clip):
    path, expected_chords, seconds_per_chord = chord_progression_clip
    y, sr = load_audio(path)
    segments = detect_chords(y, sr, segment_seconds=0.5)

    for i, seg in enumerate(segments):
        expected_start = i * seconds_per_chord
        expected_end = (i + 1) * seconds_per_chord
        assert abs(seg.start - expected_start) < 0.1
        assert abs(seg.end - expected_end) < 0.1


def test_detect_chords_correlation_is_high_for_clean_triads(chord_progression_clip):
    path, _, _ = chord_progression_clip
    y, sr = load_audio(path)
    segments = detect_chords(y, sr, segment_seconds=0.5)

    for seg in segments:
        assert seg.correlation > 0.9


def test_detect_chords_merges_consecutive_identical_segments(chord_progression_clip):
    path, expected_chords, _ = chord_progression_clip
    y, sr = load_audio(path)
    # A small segment size would produce many raw windows per chord if
    # merging didn't happen — the result should still collapse to exactly
    # one entry per real chord change.
    segments = detect_chords(y, sr, segment_seconds=0.25)
    assert len(segments) == len(expected_chords)


def test_detect_chords_on_silence_returns_no_confident_chord(tmp_path):
    import numpy as np
    import soundfile as sf

    path = str(tmp_path / "silence.wav")
    sf.write(path, np.zeros(22050 * 2, dtype=np.float32), 22050)

    y, sr = load_audio(path)
    segments = detect_chords(y, sr, segment_seconds=1.0)

    # Silence has zero-variance chroma — every candidate's correlation is
    # undefined (NaN), which detect_chords maps to -inf so it can never look
    # like a confident real match.
    for seg in segments:
        assert seg.correlation == float("-inf")
