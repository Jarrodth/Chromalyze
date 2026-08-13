from chromalyze.preprocessing import load_audio
from chromalyze.tempo import detect_bpm


def test_detect_bpm_on_known_120bpm_track(click_track_120bpm):
    path, expected_bpm = click_track_120bpm
    y, sr = load_audio(path)
    bpm = detect_bpm(y, sr)
    # Beat trackers can occasionally lock onto a half/double-tempo multiple
    # of the true tempo — check for that explicitly rather than just a wide
    # tolerance, so a genuine detection failure doesn't get masked as a
    # false pass.
    ratio = bpm / expected_bpm
    assert any(abs(ratio - m) < 0.03 for m in (0.5, 1.0, 2.0)), (
        f"detected {bpm} bpm is not a plausible multiple of {expected_bpm} bpm"
    )


def test_detect_bpm_on_known_90bpm_track(click_track_90bpm):
    path, expected_bpm = click_track_90bpm
    y, sr = load_audio(path)
    bpm = detect_bpm(y, sr)
    ratio = bpm / expected_bpm
    assert any(abs(ratio - m) < 0.03 for m in (0.5, 1.0, 2.0)), (
        f"detected {bpm} bpm is not a plausible multiple of {expected_bpm} bpm"
    )


def test_detect_bpm_returns_a_plain_float(click_track_120bpm):
    path, _ = click_track_120bpm
    y, sr = load_audio(path)
    bpm = detect_bpm(y, sr)
    assert isinstance(bpm, float)
