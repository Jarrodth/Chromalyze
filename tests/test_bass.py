from chromalyze.bass import detect_bass_root_trace
from chromalyze.preprocessing import load_audio


def test_detect_bass_root_trace_finds_the_right_roots(bass_line_clip):
    path, expected_roots, seconds_per_note = bass_line_clip
    y, sr = load_audio(path)
    segments = detect_bass_root_trace(y, sr, segment_seconds=0.5)

    assert [s.root for s in segments] == expected_roots


def test_detect_bass_root_trace_segment_boundaries_match_known_notes(bass_line_clip):
    path, expected_roots, seconds_per_note = bass_line_clip
    y, sr = load_audio(path)
    segments = detect_bass_root_trace(y, sr, segment_seconds=0.5)

    for i, seg in enumerate(segments):
        expected_start = i * seconds_per_note
        expected_end = (i + 1) * seconds_per_note
        assert abs(seg.start - expected_start) < 0.1
        assert abs(seg.end - expected_end) < 0.1


def test_detect_bass_root_trace_confidence_is_positive_for_clean_notes(bass_line_clip):
    path, _, _ = bass_line_clip
    y, sr = load_audio(path)
    segments = detect_bass_root_trace(y, sr, segment_seconds=0.5)

    for seg in segments:
        assert seg.confidence > 0.1


def test_detect_bass_root_trace_merges_consecutive_identical_segments(bass_line_clip):
    path, expected_roots, _ = bass_line_clip
    y, sr = load_audio(path)
    # A small segment size would produce many raw windows per note if
    # merging didn't happen — the result should still collapse to exactly
    # one entry per real note change.
    segments = detect_bass_root_trace(y, sr, segment_seconds=0.25)
    assert len(segments) == len(expected_roots)


def test_detect_bass_root_trace_respects_given_beat_times(bass_line_clip):
    path, expected_roots, seconds_per_note = bass_line_clip
    y, sr = load_audio(path)
    # Fabricated beat grid aligned to the note boundaries — proves
    # detect_bass_root_trace actually uses beat_times when given, same
    # beat-synchronous segmentation detect_chords supports.
    beat_times = [seconds_per_note * i for i in range(1, len(expected_roots))]
    segments = detect_bass_root_trace(y, sr, beat_times=beat_times)

    assert [s.root for s in segments] == expected_roots
