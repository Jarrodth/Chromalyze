"""Beats-per-measure estimation — a real, but deliberately limited,
attempt at meter detection.

True downbeat tracking (knowing which beat is "beat 1" of a measure) is a
research-grade MIR problem — state-of-the-art tools solve it with trained
neural nets (e.g. madmom's DBN-based downbeat tracker), which is real extra
weight and complexity this package doesn't take on. This is a much
simpler heuristic built on top of the beat times `detect_beats` already
gives us: downbeats are usually the most rhythmically accented beat in a
measure (a kick drum on beat 1, for example), so autocorrelating a
per-beat "accent" signal over candidate measure lengths finds the
periodicity that best explains the accent pattern.

This works reasonably on music with a clear, consistent rhythmic accent
(most drum-forward pop/rock/electronic music) and does NOT work reliably
on syncopated, accent-ambiguous, or sparsely-percussive music — hence the
confidence score alongside the estimate, the same honesty as key.py's
`key_confidence`: a low score means "don't trust this," not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

# Real time signatures in the music this is meant to handle are
# overwhelmingly some combination of 2, 3, or 4 beats per measure —
# checking a wider candidate range mostly just invites false positives
# from noise, not genuine detection of rarer meters.
_CANDIDATE_BEATS_PER_MEASURE = [2, 3, 4]

# By far the most common meter — the honest fallback when there isn't
# enough signal to estimate anything.
_DEFAULT_BEATS_PER_MEASURE = 4

_MIN_BEATS_REQUIRED = 8


@dataclass
class TimeSignatureResult:
    beats_per_measure: int  # e.g. 4 for 4/4, 3 for 3/4
    confidence: float  # gap between the best and 2nd-best candidate; low means "not confident"


_ACCENT_WINDOW_SECONDS = 0.08


def _accent_per_beat(y: np.ndarray, sr: int, beat_times: list[float]) -> np.ndarray:
    """A rough per-beat "how accented is this" value: peak absolute
    amplitude in a window around each beat time. A real downbeat is
    usually the loudest moment near its beat (a kick drum, a strummed
    chord), so this is a workable accent proxy — but it only works
    combined with a wide-enough window: detected beat times from
    `detect_beats` carry real jitter (tens of milliseconds is normal for a
    dynamic-programming beat tracker), and sampling too narrow a window —
    or worse, a single nearest frame of onset-strength — around each beat
    time systematically misses the true peak and produces spurious
    period-2-ish noise instead of the real accent pattern. Empirically
    verified against synthesized click tracks with a known accent pattern
    before settling on this approach and window size.
    """
    window_samples = int(_ACCENT_WINDOW_SECONDS * sr)
    accents = []
    for t in beat_times:
        center = int(t * sr)
        start = max(0, center - window_samples)
        end = min(len(y), center + window_samples)
        segment = y[start:end]
        accents.append(float(np.max(np.abs(segment))) if len(segment) else 0.0)
    return np.array(accents)


def _periodicity_score(accents: np.ndarray, period: int) -> float:
    """How well `accents` repeats with period `period`: a single-lag
    normalized autocorrelation, not a full spectral method. 1.0 is a
    perfect repeat, 0.0 (or below) is no meaningful correlation.
    """
    if len(accents) <= period:
        return 0.0
    a = accents[:-period] - accents[:-period].mean()
    b = accents[period:] - accents[period:].mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def detect_beats_per_measure(y: np.ndarray, sr: int, beat_times: list[float]) -> TimeSignatureResult:
    """Best-effort estimate of how many beats form one measure, from
    rhythmic accent periodicity. `beat_times` should come from
    `detect_beats` on the same audio.
    """
    if len(beat_times) < _MIN_BEATS_REQUIRED:
        return TimeSignatureResult(beats_per_measure=_DEFAULT_BEATS_PER_MEASURE, confidence=0.0)

    accents = _accent_per_beat(y, sr, beat_times)
    scores = {period: _periodicity_score(accents, period) for period in _CANDIDATE_BEATS_PER_MEASURE}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_period, best_score = ranked[0]

    # A genuine period-P accent pattern is mathematically guaranteed to
    # also score well at every multiple of P (e.g. real 2/4 is trivially
    # periodic at 4 too) — comparing against one of the winner's own
    # harmonics would make a correct, confident answer look falsely
    # ambiguous. Confidence instead measures the gap to the best
    # non-harmonic runner-up, the candidate that's actually a competing
    # hypothesis rather than a mathematical echo of the winner.
    runner_up_score = next((score for period, score in ranked[1:] if period % best_period != 0), 0.0)

    return TimeSignatureResult(beats_per_measure=best_period, confidence=float(best_score - runner_up_score))
