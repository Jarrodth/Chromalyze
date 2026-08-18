"""Recombining separated-stem waveforms (e.g. Demucs output) back into a
single signal for downstream analysis. The one consumer today is chord
detection — see `chords.detect_chords_from_stems` — where a loud drum kit
in the full mix is a common source of chroma contamination that a
per-stem remix can sidestep before chroma extraction ever runs.
"""

from __future__ import annotations

import numpy as np

# Stem names treated as percussive/non-pitched for recombination purposes.
# Demucs names its drum stem exactly this whether it's running the 4-stem
# model (vocals/drums/bass/other) or the 6-stem "htdemucs_6s" model
# (vocals/drums/bass/guitar/piano/other) — either way, every other stem
# carries some harmonic content a chord could plausibly be read from, so
# only this one is singled out.
DRUM_STEM_NAMES = frozenset({"drums"})


def combine_stems(stems: dict[str, np.ndarray], drum_attenuation: float = 0.0) -> np.ndarray:
    """Sum a dict of separated stem waveforms (e.g. {"vocals": y1, "drums":
    y2, "bass": y3, "other": y4}) back into one signal, scaling any stem
    named in DRUM_STEM_NAMES by `drum_attenuation` instead of including it
    at full strength.

    `drum_attenuation` is a linear gain applied only to drum stems before
    summing: 0.0 (the default) removes them entirely, 1.0 puts them back at
    full strength (equivalent to just re-mixing the original stems
    unchanged), and anything in between heavily attenuates without fully
    zeroing them out — for tracks where totally dropping the kick/snare
    transients would leave the remaining stems sounding unnaturally thin,
    without reintroducing the full contamination full-strength drums cause.

    Stems from the same separation job aren't always exactly the same
    length (rounding can differ per stem) — they're trimmed to the
    shortest before summing.
    """
    if not stems:
        raise ValueError("combine_stems requires at least one stem")

    min_len = min(len(waveform) for waveform in stems.values())
    mixed = np.zeros(min_len, dtype=np.float32)
    for name, waveform in stems.items():
        gain = drum_attenuation if name in DRUM_STEM_NAMES else 1.0
        if gain:
            mixed += gain * waveform[:min_len]
    return mixed
