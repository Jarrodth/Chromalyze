"""Key detection via the Krumhansl-Schmuckler algorithm.

Librosa doesn't ship a key detector directly, but it does provide the
chroma (pitch-class energy) features the classic K-S method is built on:
correlate the track's average chroma vector against empirically-derived
major/minor key profiles for all 12 possible tonics, and take the best
match.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

# Krumhansl-Kessler (1982) empirical key profiles — the standard reference
# values used across most MIR key-detection implementations. Each is
# indexed by semitone distance above the tonic (index 0 = tonic).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Conventional (minimal-accidental) note spellings, chosen separately for
# major vs. minor since the two follow different real-world usage — e.g.
# "C# minor" is common while "Db major" (not "C# major") is common.
MAJOR_TONIC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


@dataclass
class KeyResult:
    tonic: str  # e.g. "C", "Eb", "F#"
    mode: str  # "major" or "minor"
    key: str  # e.g. "C major", "F# minor"
    correlation: float  # best-match correlation score (higher = more confident)
    confidence: float  # gap between the best and second-best candidate — see detect_key()


def detect_key(y: np.ndarray, sr: int) -> KeyResult:
    """Estimate the musical key of a loaded audio signal."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            # Rotate the tonic-relative profile so index i lines up with
            # absolute pitch class (tonic + i) % 12, matching chroma's own
            # indexing (0 = C, 1 = C#, ... 11 = B).
            rotated_profile = np.roll(profile, tonic)
            correlation = np.corrcoef(chroma_mean, rotated_profile)[0, 1]
            scores.append((correlation, tonic, mode))

    scores.sort(key=lambda s: s[0], reverse=True)
    best_correlation, best_tonic, best_mode = scores[0]
    second_correlation = scores[1][0]

    # Raw gap between the winner and the runner-up, deliberately left
    # unnormalized rather than forced into a 0-1 range — real correlation
    # values here typically fall around 0.3-0.9, and a fake-precise 0-1
    # "confidence" would imply more rigor than a Pearson correlation gap
    # actually has. In practice: a gap under ~0.05 usually means the two
    # candidates are the classic relative-major/minor pair (they share every
    # pitch class, so their profile correlations land close together); a
    # gap over ~0.15 is a clear, unambiguous winner.
    confidence = float(best_correlation - second_correlation)

    names = MAJOR_TONIC_NAMES if best_mode == "major" else MINOR_TONIC_NAMES
    tonic_name = names[best_tonic]
    return KeyResult(
        tonic=tonic_name,
        mode=best_mode,
        key=f"{tonic_name} {best_mode}",
        correlation=float(best_correlation),
        confidence=confidence,
    )
