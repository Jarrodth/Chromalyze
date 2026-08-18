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
from .stems import combine_stems

# Chord templates are a small, precise set of tones (unlike a key profile,
# which weights all 7 scale degrees to varying degrees) — 1 at each chord
# tone's semitone distance above the root, 0 elsewhere.
MAJOR_TRIAD_TEMPLATE = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
MINOR_TRIAD_TEMPLATE = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

DEFAULT_SEGMENT_SECONDS = 1.0
_CHROMA_HOP_LENGTH = 512  # librosa.feature.chroma_cqt's own default

# A raw segment shorter than this fraction of the track's median segment
# duration is treated as "isolated" by _smooth_isolated_segments below.
_ISOLATED_DURATION_RATIO = 0.5


@dataclass
class ChordSegment:
    start: float  # seconds
    end: float  # seconds
    chord: str  # e.g. "C", "Am"
    correlation: float  # best-match correlation score for this segment
    confidence: float  # gap between the best and second-best candidate — same convention as KeyResult.confidence in key.py


def _chord_name(root_pc: int, quality: str) -> str:
    root_name = MAJOR_TONIC_NAMES[root_pc]
    return root_name if quality == "major" else f"{root_name}m"


def _best_chord_for_chroma(chroma_vector: np.ndarray) -> tuple[str, float, float]:
    scores = []
    for quality, template in (("major", MAJOR_TRIAD_TEMPLATE), ("minor", MINOR_TRIAD_TEMPLATE)):
        for root in range(12):
            rotated = np.roll(template, root)
            correlation = np.corrcoef(chroma_vector, rotated)[0, 1]
            if np.isnan(correlation):
                # A silent/near-silent segment has ~zero variance — no
                # meaningful chord to report, so it can never win.
                correlation = -np.inf
            scores.append((correlation, root, quality))

    scores.sort(key=lambda s: s[0], reverse=True)
    best_correlation, best_root, best_quality = scores[0]
    second_correlation = scores[1][0]

    # Raw gap between the winner and the runner-up, deliberately left
    # unnormalized rather than forced into a 0-1 range — see detect_key()
    # in key.py for the same convention and the reasoning behind it. A
    # silent/near-silent segment ties every candidate at -inf, where the
    # gap is meaningless rather than a real (dis)confirmation, so it's
    # reported as 0.0 instead of -inf minus -inf's NaN.
    confidence = 0.0 if best_correlation == -np.inf else float(best_correlation - second_correlation)

    return _chord_name(best_root, best_quality), float(best_correlation), confidence


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
                confidence=(previous.confidence + seg.confidence) / 2,
            )
        else:
            merged.append(seg)
    return merged


def _smooth_isolated_segments(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Correct isolated short chord guesses using their surrounding context.

    Takes already run-length-merged segments (see _merge_adjacent), where a
    real chord that holds across several consecutive raw windows/beats has
    already collapsed into one long segment — so a segment that's still
    much shorter than the track's other segments is exactly the case where
    only a single window/beat landed on that reading. When it's also
    flanked on both sides by a different chord its two neighbors agree on
    (..., X, Y, X, ...), that's far more likely to be a passing tone, a
    bent note, or a stray overtone than a real chord change and back — so
    it's reassigned to match its neighbors. The caller re-merges afterward
    to fold it into the surrounding segment instead of reporting a
    spurious short-lived blip.

    Comparisons read from the original (pre-smoothing) sequence, so two
    isolated segments in a row are each judged against their real
    neighbors rather than against a value this same pass already rewrote.
    """
    if len(segments) < 3:
        return segments

    durations = [seg.end - seg.start for seg in segments]
    isolated_threshold = float(np.median(durations)) * _ISOLATED_DURATION_RATIO

    smoothed = list(segments)
    for i in range(1, len(segments) - 1):
        previous_seg, current_seg, next_seg = segments[i - 1], segments[i], segments[i + 1]
        is_isolated = (
            (current_seg.end - current_seg.start) < isolated_threshold
            and previous_seg.chord == next_seg.chord
            and previous_seg.chord != current_seg.chord
        )
        if is_isolated:
            smoothed[i] = ChordSegment(
                start=current_seg.start,
                end=current_seg.end,
                chord=previous_seg.chord,
                correlation=current_seg.correlation,
                confidence=current_seg.confidence,
            )
    return smoothed


def _segment_boundaries(
    total_duration: float, segment_seconds: float, beat_times: list[float] | None
) -> list[tuple[float, float]]:
    if not beat_times:
        boundaries = []
        start = 0.0
        while start < total_duration:
            end = min(start + segment_seconds, total_duration)
            boundaries.append((start, end))
            start = end
        return boundaries

    # Beat-synchronous: segment between consecutive beats, plus a lead-in
    # before the first beat and a tail after the last one if the track
    # extends past them. Real chord changes happen on beats, not at
    # arbitrary fixed-time offsets, so aligning segment boundaries to the
    # actual beat grid avoids analyzing a window that straddles part of one
    # chord and part of the next — a window like that captures a genuine
    # blend of both chords' notes, which can correlate best with a third
    # chord that was never actually played.
    points = sorted({0.0, *(b for b in beat_times if 0.0 < b < total_duration), total_duration})
    return list(zip(points[:-1], points[1:]))


def detect_chords(
    y: np.ndarray,
    sr: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    beat_times: list[float] | None = None,
) -> list[ChordSegment]:
    """Estimate a chord for each segment of the track, merge consecutive
    segments that land on the same chord, then smooth over any surviving
    segment that's isolated — much shorter than its neighbors and flanked
    by two segments that agree with each other — using its surrounding
    context, and merge once more so a corrected segment folds back into
    the chord around it instead of surfacing as its own tiny entry.

    If `beat_times` is given (e.g. from detect_beats), segments are aligned
    to those beat boundaries instead of arbitrary fixed-length windows —
    see `_segment_boundaries`. Falls back to fixed `segment_seconds`
    windows if no beat times are given, so this still works standalone
    without requiring beat detection to have already run.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=_CHROMA_HOP_LENGTH)
    total_duration = len(y) / sr

    raw_segments: list[ChordSegment] = []
    for start, end in _segment_boundaries(total_duration, segment_seconds, beat_times):
        frame_mask = (frame_times >= start) & (frame_times < end)
        if np.any(frame_mask):
            chroma_vector = chroma[:, frame_mask].mean(axis=1)
            chord, correlation, confidence = _best_chord_for_chroma(chroma_vector)
            raw_segments.append(
                ChordSegment(start=start, end=end, chord=chord, correlation=correlation, confidence=confidence)
            )

    # Merge first so segment duration reflects how many consecutive raw
    # windows actually agreed on a chord (what "isolated" means below),
    # then smooth, then merge again to absorb any corrected segment.
    return _merge_adjacent(_smooth_isolated_segments(_merge_adjacent(raw_segments)))


def detect_chords_from_stems(
    stems: dict[str, np.ndarray],
    sr: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    beat_times: list[float] | None = None,
    drum_attenuation: float = 0.0,
) -> list[ChordSegment]:
    """Like `detect_chords`, but takes separated stems (e.g. Demucs output —
    {"vocals": y, "drums": y, "bass": y, "other": y}) instead of a single
    mixed signal.

    The stems are recombined with `drums` removed (or heavily attenuated,
    if `drum_attenuation` is raised above its default of 0.0 — see
    `combine_stems`) before chroma extraction ever runs, so template
    matching never sees the one stem least likely to carry the actual
    harmony. Everything else — vocals, bass, and whatever's left in
    "other" (or, on a 6-stem separation, "guitar"/"piano" too) — is kept
    at full strength, since any of them can plausibly be carrying the
    chord that's actually being played.
    """
    mixed = combine_stems(stems, drum_attenuation=drum_attenuation)
    return detect_chords(mixed, sr, segment_seconds=segment_seconds, beat_times=beat_times)
