"""Beat tracking — locates individual beat timestamps and, as a natural
byproduct of the same algorithm, an overall tempo estimate."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class BeatResult:
    bpm: float
    beat_times: list[float]  # seconds, one per detected beat


def detect_beats(y: np.ndarray, sr: int) -> BeatResult:
    """Estimate tempo and beat locations for a loaded audio signal."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    # librosa can return tempo as a 0-d/1-element array depending on version.
    bpm = float(np.atleast_1d(tempo)[0])
    return BeatResult(bpm=bpm, beat_times=beat_times)
