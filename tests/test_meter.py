from chromalyze.beats import detect_beats
from chromalyze.meter import detect_beats_per_measure
from chromalyze.preprocessing import load_audio


def test_detect_beats_per_measure_recovers_4_4(accented_click_track_4_4):
    y, sr = load_audio(accented_click_track_4_4)
    beats = detect_beats(y, sr)
    result = detect_beats_per_measure(y, sr, beats.beat_times)

    assert result.beats_per_measure == 4
    assert result.confidence > 0.5  # a clean, strongly-accented pattern should be a confident match


def test_detect_beats_per_measure_recovers_3_4(accented_click_track_3_4):
    y, sr = load_audio(accented_click_track_3_4)
    beats = detect_beats(y, sr)
    result = detect_beats_per_measure(y, sr, beats.beat_times)

    assert result.beats_per_measure == 3
    assert result.confidence > 0.5


def test_detect_beats_per_measure_low_confidence_with_no_accent_pattern(unaccented_click_track):
    # A click track with every beat at the same loudness has no real
    # answer — the estimator must not report high confidence in a guess.
    y, sr = load_audio(unaccented_click_track)
    beats = detect_beats(y, sr)
    result = detect_beats_per_measure(y, sr, beats.beat_times)

    assert result.confidence < 0.1


def test_detect_beats_per_measure_defaults_with_too_few_beats():
    import numpy as np

    y = np.zeros(22050, dtype=np.float32)  # 1 second of silence, ~0 beats
    result = detect_beats_per_measure(y, 22050, beat_times=[0.5, 1.0, 1.5])

    assert result.beats_per_measure == 4  # honest default, not a guess dressed up as a real answer
    assert result.confidence == 0.0
