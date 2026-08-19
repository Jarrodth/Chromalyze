from chromalyze.beats import detect_beats
from chromalyze.chords import detect_chords, detect_chords_from_stems
from chromalyze.preprocessing import load_audio
from chromalyze.stems import combine_stems


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


def test_beat_synchronous_segmentation_is_more_accurate_than_fixed_windows(rhythmic_chord_progression_clip):
    path, expected_chords, expected_boundaries = rhythmic_chord_progression_clip
    y, sr = load_audio(path)

    # Real beat detection on real rhythmic audio — not fabricated ground
    # truth — to prove the actual pipeline benefits, not just the
    # segmentation math in isolation.
    beats = detect_beats(y, sr)

    fixed_window_segments = detect_chords(y, sr, segment_seconds=1.0)
    beat_synced_segments = detect_chords(y, sr, beat_times=beats.beat_times)

    # Beat-synchronous boundaries should land much closer to the track's
    # true chord-change points (0, 1.5, 3.0, 4.5s) than arbitrary
    # fixed-time windows, which have no reason to align with them.
    true_change_points = expected_boundaries[1:-1]

    def max_boundary_error(segments):
        detected_points = [s.start for s in segments[1:]]  # skip the very first segment's start (always 0)
        if not detected_points:
            return float("inf")
        return max(min(abs(d - t) for d in detected_points) for t in true_change_points)

    fixed_error = max_boundary_error(fixed_window_segments)
    synced_error = max_boundary_error(beat_synced_segments)

    assert synced_error < fixed_error
    assert synced_error < 0.1  # within 100ms of the true chord changes
    assert [s.chord for s in beat_synced_segments] == expected_chords


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
        assert seg.confidence == 0.0


def test_detect_chords_confidence_is_positive_for_clean_triads(chord_progression_clip):
    path, _, _ = chord_progression_clip
    y, sr = load_audio(path)
    segments = detect_chords(y, sr, segment_seconds=0.5)

    # A clean, unambiguous triad should beat every other root/quality
    # candidate by a clear margin, not just the winner alone scoring well.
    for seg in segments:
        assert seg.confidence > 0.1


def test_isolated_single_beat_chord_guess_is_smoothed_using_context(rhythmic_clip_with_single_beat_blip):
    path, expected_chord = rhythmic_clip_with_single_beat_blip
    y, sr = load_audio(path)
    beats = detect_beats(y, sr)

    segments = detect_chords(y, sr, beat_times=beats.beat_times)

    # The lone stray beat should be corrected to match the chord on both
    # sides of it, collapsing the whole clip back into one segment instead
    # of surfacing a spurious one-beat blip in the middle.
    assert [s.chord for s in segments] == [expected_chord]


def test_detect_chords_from_stems_excludes_drum_contamination(contaminated_stems_clip):
    stems, sr, expected_chord = contaminated_stems_clip

    clean_segments = detect_chords_from_stems(stems, sr, segment_seconds=1.0)
    assert all(s.chord == expected_chord for s in clean_segments)

    # Same signal, but recombined with drums left in at full strength (the
    # naive full-mix approach) — proves the drum exclusion is actually
    # doing something, not just that the new code path runs.
    naive_mix = combine_stems(stems, drum_attenuation=1.0)
    contaminated_segments = detect_chords(naive_mix, sr, segment_seconds=1.0)
    assert any(s.chord != expected_chord for s in contaminated_segments)


def test_detect_chords_from_stems_bandpass_removes_sub_bass_rumble(rumble_contaminated_stem):
    stems, sr, expected_chord = rumble_contaminated_stem

    # apply_bandpass defaults to True — loud sub-bass rumble below
    # DEFAULT_CHORD_BANDPASS_LOW_HZ shouldn't derail detection.
    filtered_segments = detect_chords_from_stems(stems, sr, segment_seconds=1.0)
    assert all(s.chord == expected_chord for s in filtered_segments)

    # Same signal with bandpass explicitly disabled — proves the rumble
    # really would derail detection without it, not just that the
    # parameter exists.
    unfiltered_segments = detect_chords_from_stems(stems, sr, segment_seconds=1.0, apply_bandpass=False)
    assert any(s.chord != expected_chord for s in unfiltered_segments)
