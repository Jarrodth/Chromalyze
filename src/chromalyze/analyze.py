"""Top-level entry point — ties every analysis stage together into one call."""

from __future__ import annotations

from dataclasses import dataclass

from .beats import detect_beats
from .key import detect_key
from .preprocessing import DEFAULT_SAMPLE_RATE, load_audio


@dataclass
class AnalysisResult:
    bpm: float | None
    beats: list[float] | None
    key: str | None
    key_confidence: float | None  # see KeyResult.confidence in key.py
    chords: list[str] | None  # not yet implemented
    scale: str | None  # not yet implemented


def analyze(audio_file: str) -> AnalysisResult:
    """Run every available analysis stage on `audio_file` and return the
    combined result. Fields for stages not yet built are None.
    """
    y, sr = load_audio(audio_file, sr=DEFAULT_SAMPLE_RATE)
    beats = detect_beats(y, sr)
    key = detect_key(y, sr)

    return AnalysisResult(
        bpm=beats.bpm,
        beats=beats.beat_times,
        key=key.key,
        key_confidence=key.confidence,
        chords=None,
        scale=None,
    )
