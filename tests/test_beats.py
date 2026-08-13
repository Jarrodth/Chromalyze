from chromalyze.beats import detect_beats
from chromalyze.preprocessing import load_audio


def test_detect_beats_returns_beat_times_in_range(click_track_120bpm):
    path, _ = click_track_120bpm
    y, sr = load_audio(path)
    result = detect_beats(y, sr)

    assert len(result.beat_times) > 0
    # 10 real seconds of audio — no beat timestamp should fall outside it.
    assert all(0 <= t <= 10.0 for t in result.beat_times)
    # Timestamps should be strictly increasing.
    assert result.beat_times == sorted(result.beat_times)


def test_detect_beats_spacing_matches_known_tempo(click_track_120bpm):
    path, expected_bpm = click_track_120bpm
    y, sr = load_audio(path)
    result = detect_beats(y, sr)

    expected_interval = 60.0 / expected_bpm
    intervals = [b - a for a, b in zip(result.beat_times, result.beat_times[1:])]
    # Beat trackers can occasionally skip or double a beat, so check the
    # median interval (robust to a handful of outliers) rather than every
    # single gap matching exactly.
    intervals_sorted = sorted(intervals)
    median_interval = intervals_sorted[len(intervals_sorted) // 2]
    assert abs(median_interval - expected_interval) < expected_interval * 0.1


def test_detect_beats_bpm_matches_standalone_detect_bpm(click_track_90bpm):
    from chromalyze.tempo import detect_bpm

    path, _ = click_track_90bpm
    y, sr = load_audio(path)
    result = detect_beats(y, sr)
    standalone_bpm = detect_bpm(y, sr)

    assert result.bpm == standalone_bpm
