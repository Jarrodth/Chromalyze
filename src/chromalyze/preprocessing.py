"""Audio loading and preprocessing — the shared first step every other
analysis stage builds on."""

from __future__ import annotations

import librosa
import numpy as np

DEFAULT_SAMPLE_RATE = 22050


def load_audio(path: str, sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono signal resampled to `sr`.

    Returns (samples, sample_rate).
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr
