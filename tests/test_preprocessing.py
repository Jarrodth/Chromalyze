from chromalyze.preprocessing import load_audio


def test_load_audio_returns_correct_sample_rate(click_track_120bpm):
    path, _ = click_track_120bpm
    y, sr = load_audio(path, sr=22050)
    assert sr == 22050


def test_load_audio_returns_mono_signal_of_expected_length(click_track_120bpm):
    path, _ = click_track_120bpm
    y, sr = load_audio(path, sr=22050)
    assert y.ndim == 1
    # File was written as 10.0s of real audio — loaded length should match
    # within a tiny tolerance (resampling/framing can shift it by a sample
    # or two, never by a meaningful fraction of a second).
    expected_samples = 10.0 * sr
    assert abs(len(y) - expected_samples) < sr * 0.05


def test_load_audio_resamples_to_requested_rate(click_track_120bpm):
    path, _ = click_track_120bpm
    y, sr = load_audio(path, sr=8000)
    assert sr == 8000
    assert abs(len(y) - 10.0 * 8000) < 8000 * 0.05
