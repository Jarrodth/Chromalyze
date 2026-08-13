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


def make_accented_click_track(
    bpm: float, beats_per_measure: int, duration_seconds: float, sr: int, path: str, accent_ratio: float = 3.0
) -> None:
    """Like `make_click_track`, but every `beats_per_measure`-th click
    (the "downbeat") is louder than the rest by `accent_ratio` — a real,
    if simplistic, stand-in for a drum pattern's kick-on-beat-1 accent,
    needed to test meter.py's beats-per-measure estimator against a known
    ground truth.
    """
    total_samples = int(duration_seconds * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    click_interval_samples = int(60.0 / bpm * sr)
    click_length = int(0.02 * sr)
    decay = np.exp(-np.linspace(0, 12, click_length))
    click_waveform = decay * np.sin(2 * np.pi * 1200 * np.arange(click_length) / sr)

    position = 0
    beat_index = 0
    while position + click_length < total_samples:
        amplitude = accent_ratio if beat_index % beats_per_measure == 0 else 1.0
        audio[position : position + click_length] += amplitude * click_waveform
        position += click_interval_samples
        beat_index += 1

    sf.write(path, audio, sr)


@pytest.fixture
def accented_click_track_4_4(tmp_path):
    path = str(tmp_path / "accented_click_4_4.wav")
    make_accented_click_track(bpm=120.0, beats_per_measure=4, duration_seconds=30.0, sr=22050, path=path)
    return path


@pytest.fixture
def accented_click_track_3_4(tmp_path):
    path = str(tmp_path / "accented_click_3_4.wav")
    make_accented_click_track(bpm=120.0, beats_per_measure=3, duration_seconds=30.0, sr=22050, path=path)
    return path


@pytest.fixture
def unaccented_click_track(tmp_path):
    path = str(tmp_path / "unaccented_click.wav")
    make_click_track(bpm=120.0, duration_seconds=30.0, sr=22050, path=path)
    return path


def note_freq(pitch_class: int, octave: int = 4) -> float:
    """Frequency in Hz for a pitch class (0=C .. 11=B) in the given octave,
    12-tone equal temperament, A4 = 440Hz."""
    semitones_from_a4 = (pitch_class - 9) + 12 * (octave - 4)
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


def make_tonal_clip(tonic_pc: int, mode: str, duration_seconds: float, sr: int, path: str) -> None:
    """Write a real WAV file playing a I-IV-V-I (or i-iv-V-i for minor)
    chord progression in the given key — strong, unambiguous tonal/cadential
    content a key detector should have no trouble with, built from real
    additive sine synthesis at real 12-TET frequencies, not a pre-made
    audio file.
    """
    if mode == "major":
        # (root semitone offset, chord quality) for I, IV, V, I
        chords = [(0, "major"), (5, "major"), (7, "major"), (0, "major")]
    else:
        # i, iv, V (major dominant — standard harmonic-minor-style cadence,
        # gives a much stronger/less ambiguous key cue than a plain v)
        chords = [(0, "minor"), (5, "minor"), (7, "major"), (0, "minor")]

    chord_seconds = duration_seconds / len(chords)
    chord_samples = int(chord_seconds * sr)
    t = np.arange(chord_samples) / sr

    segments = []
    for root_offset, quality in chords:
        intervals = [0, 4, 7] if quality == "major" else [0, 3, 7]
        chord_wave = np.zeros(chord_samples, dtype=np.float32)
        for interval in intervals:
            pc = (tonic_pc + root_offset + interval) % 12
            freq = note_freq(pc, octave=3)
            chord_wave += np.sin(2 * np.pi * freq * t).astype(np.float32)
        chord_wave /= len(intervals)
        segments.append(chord_wave)

    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def tonal_clip_c_major(tmp_path):
    path = str(tmp_path / "tonal_c_major.wav")
    make_tonal_clip(tonic_pc=0, mode="major", duration_seconds=8.0, sr=22050, path=path)
    return path, "C", "major"


@pytest.fixture
def tonal_clip_a_minor(tmp_path):
    path = str(tmp_path / "tonal_a_minor.wav")
    make_tonal_clip(tonic_pc=9, mode="minor", duration_seconds=8.0, sr=22050, path=path)
    return path, "A", "minor"


@pytest.fixture
def tonal_clip_e_major(tmp_path):
    path = str(tmp_path / "tonal_e_major.wav")
    make_tonal_clip(tonic_pc=4, mode="major", duration_seconds=8.0, sr=22050, path=path)
    return path, "E", "major"


@pytest.fixture
def tonal_clip_f_sharp_minor(tmp_path):
    path = str(tmp_path / "tonal_fsharp_minor.wav")
    make_tonal_clip(tonic_pc=6, mode="minor", duration_seconds=8.0, sr=22050, path=path)
    return path, "F#", "minor"


def make_ambiguous_scale_clip(tonic_pc: int, duration_seconds: float, sr: int, path: str) -> None:
    """Write a real WAV file playing the 7 notes of a major scale in
    ascending order, each held equally, with no chords and no cadence —
    deliberately the hardest possible case for key detection, since a major
    scale and its relative natural minor scale share every pitch class.
    There's nothing in this signal to tell the detector which of the two
    is actually the tonic, unlike the I-IV-V-I clips above.
    """
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    note_seconds = duration_seconds / len(major_intervals)
    note_samples = int(note_seconds * sr)
    t = np.arange(note_samples) / sr

    segments = []
    for interval in major_intervals:
        pc = (tonic_pc + interval) % 12
        freq = note_freq(pc, octave=4)
        segments.append(np.sin(2 * np.pi * freq * t).astype(np.float32))

    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def ambiguous_scale_clip_c(tmp_path):
    path = str(tmp_path / "ambiguous_scale_c.wav")
    make_ambiguous_scale_clip(tonic_pc=0, duration_seconds=7.0, sr=22050, path=path)
    return path


def make_chord_progression_clip(
    chords: list[tuple[int, str]], seconds_per_chord: float, sr: int, path: str
) -> None:
    """Write a real WAV file playing a sequence of held triads — real
    additive sine synthesis at real 12-TET frequencies, not a pre-made
    audio file. `chords` is a list of (root pitch class, "major"/"minor").
    """
    samples_per_chord = int(seconds_per_chord * sr)
    t = np.arange(samples_per_chord) / sr

    segments = []
    for root_pc, quality in chords:
        intervals = [0, 4, 7] if quality == "major" else [0, 3, 7]
        chord_wave = np.zeros(samples_per_chord, dtype=np.float32)
        for interval in intervals:
            pc = (root_pc + interval) % 12
            freq = note_freq(pc, octave=3)
            chord_wave += np.sin(2 * np.pi * freq * t).astype(np.float32)
        chord_wave /= len(intervals)
        segments.append(chord_wave)

    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def chord_progression_clip(tmp_path):
    path = str(tmp_path / "chord_progression.wav")
    # C - Am - F - G, 2 seconds each — a textbook I-vi-IV-V progression.
    chords = [(0, "major"), (9, "minor"), (5, "major"), (7, "major")]
    make_chord_progression_clip(chords, seconds_per_chord=2.0, sr=22050, path=path)
    return path, ["C", "Am", "F", "G"], 2.0


def make_rhythmic_chord_progression_clip(
    chords: list[tuple[int, str]], bpm: float, beats_per_chord: int, sr: int, path: str
) -> None:
    """Write a real WAV file playing a chord progression as rhythmic
    "strums" (short decaying bursts, not held tones) at a steady bpm — real
    onset content a beat tracker can actually lock onto, unlike
    make_chord_progression_clip's smooth sustained chords. Needed to test
    beat-synchronous chord segmentation for real, since that only helps
    once there's something for detect_beats to genuinely detect.
    """
    beat_interval_samples = int(60.0 / bpm * sr)
    strum_samples = int(0.25 * sr)
    decay = np.exp(-np.linspace(0, 4, strum_samples))
    t = np.arange(strum_samples) / sr

    segments = []
    for root_pc, quality in chords:
        intervals = [0, 4, 7] if quality == "major" else [0, 3, 7]
        for _ in range(beats_per_chord):
            strum = np.zeros(strum_samples, dtype=np.float32)
            for interval in intervals:
                pc = (root_pc + interval) % 12
                freq = note_freq(pc, octave=3)
                strum += (decay * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            strum /= len(intervals)
            pad = np.zeros(max(0, beat_interval_samples - strum_samples), dtype=np.float32)
            segments.append(np.concatenate([strum, pad]))

    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def rhythmic_chord_progression_clip(tmp_path):
    path = str(tmp_path / "rhythmic_chord_progression.wav")
    # G - Em - C - D, 3 beats each at 120bpm (1.5s per chord).
    chords = [(7, "major"), (4, "minor"), (0, "major"), (2, "major")]
    make_rhythmic_chord_progression_clip(chords, bpm=120.0, beats_per_chord=3, sr=22050, path=path)
    expected_boundaries = [0.0, 1.5, 3.0, 4.5, 6.0]
    return path, ["G", "Em", "C", "D"], expected_boundaries
