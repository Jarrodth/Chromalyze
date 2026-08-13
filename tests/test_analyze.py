from chromalyze import analyze


def test_analyze_populates_implemented_fields(click_track_120bpm):
    path, expected_bpm = click_track_120bpm
    result = analyze(path)

    assert result.bpm is not None
    assert result.beats is not None
    assert len(result.beats) > 0
    ratio = result.bpm / expected_bpm
    assert any(abs(ratio - m) < 0.03 for m in (0.5, 1.0, 2.0))


def test_analyze_leaves_unimplemented_fields_as_none(click_track_120bpm):
    path, _ = click_track_120bpm
    result = analyze(path)

    assert result.key is None
    assert result.chords is None
    assert result.scale is None
