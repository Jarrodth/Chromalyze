import numpy as np

from chromalyze.preprocessing import bandpass_filter, load_audio


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


def _sine(freq, sr, duration_seconds=2.0):
    t = np.arange(int(duration_seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _rms(y):
    return float(np.sqrt(np.mean(y**2)))


def test_bandpass_filter_passes_a_frequency_inside_the_band():
    sr = 22050
    y = _sine(1000.0, sr)  # well within the default 70-4500Hz band
    filtered = bandpass_filter(y, sr)
    # Zero-phase filtering preserves amplitude for an in-band tone —
    # only its edges are affected by the filter's settling.
    assert _rms(filtered) > 0.9 * _rms(y)


def test_bandpass_filter_attenuates_sub_bass_rumble():
    sr = 22050
    y = _sine(40.0, sr)  # below the default 70Hz low cutoff
    filtered = bandpass_filter(y, sr)
    assert _rms(filtered) < 0.1 * _rms(y)


def test_bandpass_filter_attenuates_high_frequency_noise():
    sr = 22050
    y = _sine(8000.0, sr)  # above the default 4500Hz high cutoff
    filtered = bandpass_filter(y, sr)
    assert _rms(filtered) < 0.1 * _rms(y)


def test_bandpass_filter_preserves_signal_length():
    sr = 22050
    y = _sine(1000.0, sr)
    filtered = bandpass_filter(y, sr)
    assert len(filtered) == len(y)


def test_bandpass_filter_respects_custom_cutoffs():
    sr = 22050
    y = _sine(100.0, sr)
    # Default band would pass 100Hz; a tighter custom band excluding it
    # should attenuate it instead.
    filtered = bandpass_filter(y, sr, low_hz=200.0, high_hz=4500.0)
    assert _rms(filtered) < 0.1 * _rms(y)
