from chromalyze.key import detect_key
from chromalyze.preprocessing import load_audio


def test_detect_key_c_major(tonal_clip_c_major):
    path, expected_tonic, expected_mode = tonal_clip_c_major
    y, sr = load_audio(path)
    result = detect_key(y, sr)

    assert result.tonic == expected_tonic
    assert result.mode == expected_mode
    assert result.key == "C major"


def test_detect_key_a_minor(tonal_clip_a_minor):
    path, expected_tonic, expected_mode = tonal_clip_a_minor
    y, sr = load_audio(path)
    result = detect_key(y, sr)

    assert result.tonic == expected_tonic
    assert result.mode == expected_mode
    assert result.key == "A minor"


def test_detect_key_e_major(tonal_clip_e_major):
    path, expected_tonic, expected_mode = tonal_clip_e_major
    y, sr = load_audio(path)
    result = detect_key(y, sr)

    assert result.tonic == expected_tonic
    assert result.mode == expected_mode


def test_detect_key_f_sharp_minor(tonal_clip_f_sharp_minor):
    path, expected_tonic, expected_mode = tonal_clip_f_sharp_minor
    y, sr = load_audio(path)
    result = detect_key(y, sr)

    assert result.tonic == expected_tonic
    assert result.mode == expected_mode


def test_detect_key_correlation_is_a_plain_float(tonal_clip_c_major):
    path, _, _ = tonal_clip_c_major
    y, sr = load_audio(path)
    result = detect_key(y, sr)
    assert isinstance(result.correlation, float)
    # A clean, strongly tonal synthetic clip should correlate well with its
    # true key profile — not just "highest of 24 mediocre scores".
    assert result.correlation > 0.5


def test_confidence_is_high_for_a_clear_cadential_clip(tonal_clip_c_major):
    path, _, _ = tonal_clip_c_major
    y, sr = load_audio(path)
    result = detect_key(y, sr)
    assert result.confidence > 0.15


def test_confidence_is_low_for_a_genuinely_ambiguous_clip(ambiguous_scale_clip_c):
    y, sr = load_audio(ambiguous_scale_clip_c)
    result = detect_key(y, sr)
    # A plain scale run (no chords, no cadence) shares every pitch class
    # with its relative minor — there's genuinely nothing in this signal to
    # tell the two apart, and the confidence score should reflect that.
    assert result.confidence < 0.1


def test_ambiguous_clip_confidence_is_lower_than_cadential_clip(tonal_clip_c_major, ambiguous_scale_clip_c):
    path, _, _ = tonal_clip_c_major
    y_cadence, sr_cadence = load_audio(path)
    y_ambiguous, sr_ambiguous = load_audio(ambiguous_scale_clip_c)

    cadence_result = detect_key(y_cadence, sr_cadence)
    ambiguous_result = detect_key(y_ambiguous, sr_ambiguous)

    assert ambiguous_result.confidence < cadence_result.confidence
