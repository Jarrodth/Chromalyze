"""Chord recognition — the same chroma template-matching principle as key
detection (see key.py), applied over short time segments instead of a
whole-track average, with simple triad templates instead of the fuzzier
Krumhansl-Schmuckler key profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .key import MAJOR_TONIC_NAMES

# Chord templates are a small, precise set of tones (unlike a key profile,
# which weights all 7 scale degrees to varying degrees) — 1 at each chord
# tone's semitone distance above the root, 0 elsewhere.
MAJOR_TRIAD_TEMPLATE = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
MINOR_TRIAD_TEMPLATE = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

DEFAULT_SEGMENT_SECONDS = 1.0
_CHROMA_HOP_LENGTH = 512  # librosa.feature.chroma_cqt's own default


@dataclass
class ChordSegment:
    start: float  # seconds
    end: float  # seconds
    chord: str  # e.g. "C", "Am"
    correlation: float  # best-match correlation score for this segment


def _chord_name(root_pc: int, quality: str) -> str:
    root_name = MAJOR_TONIC_NAMES[root_pc]
    return root_name if quality == "major" else f"{root_name}m"


def _best_chord_for_chroma(chroma_vector: np.ndarray) -> tuple[str, float]:
    best_correlation = -np.inf
    best_root = 0
    best_quality = "major"

    for quality, template in (("major", MAJOR_TRIAD_TEMPLATE), ("minor", MINOR_TRIAD_TEMPLATE)):
        for root in range(12):
            rotated = np.roll(template, root)
            correlation = np.corrcoef(chroma_vector, rotated)[0, 1]
            if np.isnan(correlation):
                # A silent/near-silent segment has ~zero variance — no
                # meaningful chord to report, so it can never win.
                correlation = -np.inf
            if correlation > best_correlation:
                best_correlation = correlation
                best_root = root
                best_quality = quality

    return _chord_name(best_root, best_quality), float(best_correlation)


def _merge_adjacent(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Collapse consecutive segments that landed on the same chord into one
    longer segment, rather than returning artificially fragmented output
    when a chord persists across several fixed-size windows.
    """
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.chord == merged[-1].chord:
            previous = merged[-1]
            merged[-1] = ChordSegment(
                start=previous.start,
                end=seg.end,
                chord=seg.chord,
                correlation=(previous.correlation + seg.correlation) / 2,
            )
        else:
            merged.append(seg)
    return merged


def detect_chords(y: np.ndarray, sr: int, segment_seconds: float = DEFAULT_SEGMENT_SECONDS) -> list[ChordSegment]:
    """Estimate a chord for each fixed-length segment of the track, then
    merge consecutive segments that land on the same chord.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=_CHROMA_HOP_LENGTH)
    total_duration = len(y) / sr

    raw_segments: list[ChordSegment] = []
    start = 0.0
    while start < total_duration:
        end = min(start + segment_seconds, total_duration)
        frame_mask = (frame_times >= start) & (frame_times < end)
        if np.any(frame_mask):
            chroma_vector = chroma[:, frame_mask].mean(axis=1)
            chord, correlation = _best_chord_for_chroma(chroma_vector)
            raw_segments.append(ChordSegment(start=start, end=end, chord=chord, correlation=correlation))
        start = end

    return _merge_adjacent(raw_segments)
