"""Bass-line root-note tracing.

A bass part is close to monophonic — one note at a time, rarely a full
chord — so its single dominant chroma pitch class per window is a real,
independent read on a song's actual chord root at that moment. Useful as
a cross-check against chord recognition's own root guesses (see
chords.py), which reads the root off triad template-matching against
the full (or drum-excluded) mix instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .chords import _CHROMA_HOP_LENGTH, _segment_boundaries
from .key import MAJOR_TONIC_NAMES

DEFAULT_SEGMENT_SECONDS = 1.0


@dataclass
class BassRootSegment:
    start: float  # seconds
    end: float  # seconds
    root: str  # e.g. "A", "F#" — no quality: a single bass note alone can't imply major/minor
    confidence: float  # gap between the best and second-best pitch class's chroma energy for this segment


def _best_root_for_chroma(chroma_vector: np.ndarray) -> tuple[str, float]:
    order = np.argsort(chroma_vector)[::-1]
    best_pc, second_pc = int(order[0]), int(order[1])
    confidence = float(chroma_vector[best_pc] - chroma_vector[second_pc])
    return MAJOR_TONIC_NAMES[best_pc], confidence


def _merge_adjacent_roots(segments: list[BassRootSegment]) -> list[BassRootSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.root == merged[-1].root:
            previous = merged[-1]
            merged[-1] = BassRootSegment(
                start=previous.start,
                end=seg.end,
                root=seg.root,
                confidence=(previous.confidence + seg.confidence) / 2,
            )
        else:
            merged.append(seg)
    return merged


def detect_bass_root_trace(
    y: np.ndarray,
    sr: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    beat_times: list[float] | None = None,
) -> list[BassRootSegment]:
    """Trace the dominant pitch class of a bass line over time.

    Same beat-window chroma averaging as detect_chords (see
    `_segment_boundaries` there), but reads off the single loudest pitch
    class per window instead of correlating against triad templates — a
    bass line has no "quality" to guess at, it's (almost always) one note
    at a time, not a chord.

    Intended for a bass-only stem, not a full mix: chroma from a full mix
    would be dominated by whatever's loudest overall, not specifically
    the bass line. Pair with `combine_stems`/stem separation to get one.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=_CHROMA_HOP_LENGTH)
    total_duration = len(y) / sr

    raw_segments: list[BassRootSegment] = []
    for start, end in _segment_boundaries(total_duration, segment_seconds, beat_times):
        frame_mask = (frame_times >= start) & (frame_times < end)
        if np.any(frame_mask):
            chroma_vector = chroma[:, frame_mask].mean(axis=1)
            root, confidence = _best_root_for_chroma(chroma_vector)
            raw_segments.append(BassRootSegment(start=start, end=end, root=root, confidence=confidence))

    return _merge_adjacent_roots(raw_segments)
