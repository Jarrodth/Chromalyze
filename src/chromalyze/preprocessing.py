"""Audio loading and preprocessing — the shared first step every other
analysis stage builds on."""

from __future__ import annotations

import librosa
import numpy as np
import scipy.signal

DEFAULT_SAMPLE_RATE = 22050

# Chosen for chord detection specifically, not as a generic "vocal
# isolation" or "telephone band" range:
#
# - Low end (70Hz, just below D2) sits below standard 4-string bass
#   guitar's lowest open string (E1, ~41Hz) *and* below its lowest
#   commonly-fretted low notes, so real bass/guitar note fundamentals
#   survive — this only clears true sub-bass rumble (room noise, mic
#   handling, any residual kick-drum bleed the drum-stem exclusion in
#   stems.py didn't fully catch). A tighter cutoff nearer 130-150Hz would
#   remove the actual fundamental of most of a bass line's real notes —
#   worse, a note's *harmonics* (all that's left once its fundamental is
#   gone) naturally emphasize a major third and fifth above it, which is
#   exactly the kind of bias that makes minor chords misread as major
#   (see resolve_quality_oscillation in progressions.py for the same
#   failure mode from a different cause).
# - High end (4500Hz) clears content that's almost entirely harmonics and
#   non-pitched noise for chord purposes — distortion adds a lot of harsh
#   upper-harmonic energy that can smear across many chroma bins as noise
#   rather than real pitch information, so this matters most for heavily
#   distorted/overdriven guitar tones.
DEFAULT_CHORD_BANDPASS_LOW_HZ = 70.0
DEFAULT_CHORD_BANDPASS_HIGH_HZ = 4500.0


def load_audio(path: str, sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono signal resampled to `sr`.

    Returns (samples, sample_rate).
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def bandpass_filter(
    y: np.ndarray,
    sr: int,
    low_hz: float = DEFAULT_CHORD_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_CHORD_BANDPASS_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, `low_hz` to `high_hz` — an
    optional step before chroma-based analysis (detect_chords, detect_key)
    to clear content that's mostly noise for pitch-class purposes without
    touching the frequency range real chord tones actually live in. See
    DEFAULT_CHORD_BANDPASS_LOW_HZ/DEFAULT_CHORD_BANDPASS_HIGH_HZ for why
    those specific defaults were chosen.

    Zero-phase (sosfiltfilt, filtering forward then backward) rather than
    a single causal pass, so the filter doesn't shift where transients
    land in time — that would misalign filtered audio against beat times
    computed from the unfiltered signal.
    """
    nyquist = sr / 2
    low = max(low_hz, 1.0) / nyquist
    high = min(high_hz, nyquist - 1.0) / nyquist
    sos = scipy.signal.butter(order, [low, high], btype="band", output="sos")
    return scipy.signal.sosfiltfilt(sos, y).astype(np.float32)
