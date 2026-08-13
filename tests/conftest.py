"""Shared test fixtures — real, synthetic audio with a known ground truth,
not mocked signals, so tests actually exercise librosa's real algorithms."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def make_click_track(bpm: float, duration_seconds: float, sr: int, path: str) -> None:
    """Write a real WAV file containing short percussive clicks at exact
    `bpm`-spaced intervals — a clean, unambiguous onset pattern that beat
    trackers handle reliably, much like a real metronome recording.
    """
    total_samples = int(duration_seconds * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    click_interval_samples = int(60.0 / bpm * sr)
    click_length = int(0.02 * sr)  # 20ms decaying burst per click
    decay = np.exp(-np.linspace(0, 12, click_length))
    click_waveform = decay * np.sin(2 * np.pi * 1200 * np.arange(click_length) / sr)

    position = 0
    while position + click_length < total_samples:
        audio[position : position + click_length] += click_waveform
        position += click_interval_samples

    sf.write(path, audio, sr)


@pytest.fixture
def click_track_120bpm(tmp_path):
    path = str(tmp_path / "click_120bpm.wav")
    make_click_track(bpm=120.0, duration_seconds=10.0, sr=22050, path=path)
    return path, 120.0


@pytest.fixture
def click_track_90bpm(tmp_path):
    path = str(tmp_path / "click_90bpm.wav")
    make_click_track(bpm=90.0, duration_seconds=10.0, sr=22050, path=path)
    return path, 90.0
