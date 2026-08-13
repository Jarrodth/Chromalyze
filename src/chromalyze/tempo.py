"""Standalone BPM detection.

Tempo falls out of the same computation as beat tracking (see beats.py), so
this is a thin wrapper for callers who only want the number and don't care
about individual beat timestamps — kept as its own function/module since
it's a distinct concept callers may reasonably want in isolation.
"""

from __future__ import annotations

import numpy as np

from .beats import detect_beats


def detect_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate tempo (BPM) for a loaded audio signal."""
    return detect_beats(y, sr).bpm
