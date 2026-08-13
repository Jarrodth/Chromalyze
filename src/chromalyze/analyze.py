"""Top-level entry point — ties every analysis stage together into one call."""

from __future__ import annotations

from dataclasses import dataclass

from .beats import detect_beats
from .chords import ChordSegment, detect_chords
from .key import detect_key
from .preprocessing import DEFAULT_SAMPLE_RATE, load_audio
from .theory import Scale, build_scale


@dataclass
class AnalysisResult:
    bpm: float | None
    beats: list[float] | None
    key: str | None
    key_confidence: float | None  # see KeyResult.confidence in key.py
    chords: list[ChordSegment] | None
    scale: Scale | None


def analyze(audio_file: str) -> AnalysisResult:
    """Run every available analysis stage on `audio_file` and return the
    combined result.
    """
    y, sr = load_audio(audio_file, sr=DEFAULT_SAMPLE_RATE)
    beats = detect_beats(y, sr)
    key = detect_key(y, sr)
    chords = detect_chords(y, sr, beat_times=beats.beat_times)
    scale = build_scale(key.tonic, key.mode)

    return AnalysisResult(
        bpm=beats.bpm,
        beats=beats.beat_times,
        key=key.key,
        key_confidence=key.confidence,
        chords=chords,
        scale=scale,
    )
