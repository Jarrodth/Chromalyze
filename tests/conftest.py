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


def make_rhythmic_clip_with_blip(
    chord: tuple[int, str],
    blip_chord: tuple[int, str],
    bpm: float,
    beats_before: int,
    blip_beats: int,
    beats_after: int,
    sr: int,
    path: str,
) -> None:
    """Like make_rhythmic_chord_progression_clip, but strums `chord` for
    `beats_before` beats, briefly switches to `blip_chord` for just
    `blip_beats` beats (a stand-in for a passing tone or a stray overtone
    getting misread as its own chord), then returns to `chord` for
    `beats_after` more beats — real onset content with a real, isolated,
    single-window misread in the middle, needed to test that detect_chords'
    minimum-duration smoothing pass corrects it back using the matching
    chord on both sides, instead of reporting a spurious short blip.
    """
    beat_interval_samples = int(60.0 / bpm * sr)
    strum_samples = int(0.25 * sr)
    decay = np.exp(-np.linspace(0, 4, strum_samples))
    t = np.arange(strum_samples) / sr

    def strum_for(root_pc: int, quality: str) -> np.ndarray:
        intervals = [0, 4, 7] if quality == "major" else [0, 3, 7]
        strum = np.zeros(strum_samples, dtype=np.float32)
        for interval in intervals:
            pc = (root_pc + interval) % 12
            freq = note_freq(pc, octave=3)
            strum += (decay * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        strum /= len(intervals)
        pad = np.zeros(max(0, beat_interval_samples - strum_samples), dtype=np.float32)
        return np.concatenate([strum, pad])

    segments = (
        [strum_for(*chord) for _ in range(beats_before)]
        + [strum_for(*blip_chord) for _ in range(blip_beats)]
        + [strum_for(*chord) for _ in range(beats_after)]
    )
    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def rhythmic_clip_with_single_beat_blip(tmp_path):
    path = str(tmp_path / "rhythmic_blip.wav")
    # C major held for 6 beats, one stray G major beat, then C major for 6
    # more beats at 120bpm — the only chord that should ever be reported is
    # C, once beat-synchronous smoothing corrects the lone G beat.
    make_rhythmic_clip_with_blip(
        chord=(0, "major"),
        blip_chord=(7, "major"),
        bpm=120.0,
        beats_before=6,
        blip_beats=1,
        beats_after=6,
        sr=22050,
        path=path,
    )
    return path, "C"


@pytest.fixture
def contaminated_stems_clip():
    """Two raw stem waveforms (no file I/O needed — detect_chords_from_stems
    takes arrays directly, same as detect_chords): a quiet, clean C major
    triad standing in for the harmony-carrying stems (vocals/bass/other),
    and a much louder F# major triad standing in for a drum stem. F# major
    (F#/A#/C#) is pitch-class-disjoint from C major (C/E/G), so a naive
    full-mix sum reliably gets pulled toward the wrong chord if the loud
    "drums" stem isn't excluded — real, direct evidence that recombining
    stems with drums removed actually changes the detected chord, not just
    that the plumbing runs without error.
    """
    sr = 22050
    duration_seconds = 2.0
    samples = int(duration_seconds * sr)
    t = np.arange(samples) / sr

    def triad_wave(root_pc: int, amplitude: float) -> np.ndarray:
        wave = np.zeros(samples, dtype=np.float32)
        for interval in (0, 4, 7):
            pc = (root_pc + interval) % 12
            freq = note_freq(pc, octave=3)
            wave += np.sin(2 * np.pi * freq * t).astype(np.float32)
        return (amplitude * wave / 3).astype(np.float32)

    harmony = triad_wave(root_pc=0, amplitude=1.0)  # C major
    drums = triad_wave(root_pc=6, amplitude=6.0)  # F# major, much louder

    stems = {"vocals": harmony, "drums": drums}
    return stems, sr, "C"


@pytest.fixture
def rumble_contaminated_stem():
    """A single raw stem: a clean, quiet C major triad plus loud sub-bass
    rumble (a stand-in for mic handling noise/room rumble/kick-drum bleed
    that survived stem separation) at F#1 (~46Hz) — a pitch class
    completely disjoint from C major, chosen specifically below
    DEFAULT_CHORD_BANDPASS_LOW_HZ (70Hz) so bandpass_filter should remove
    it entirely while leaving the real chord untouched.
    """
    sr = 22050
    duration_seconds = 2.0
    samples = int(duration_seconds * sr)
    t = np.arange(samples) / sr

    chord = np.zeros(samples, dtype=np.float32)
    for interval in (0, 4, 7):
        pc = interval % 12
        chord += np.sin(2 * np.pi * note_freq(pc, octave=3) * t).astype(np.float32)
    chord /= 3

    rumble_freq = note_freq(6, octave=1)  # F#1, ~46Hz
    rumble = 8.0 * np.sin(2 * np.pi * rumble_freq * t).astype(np.float32)

    stems = {"vocals": (chord + rumble).astype(np.float32)}
    return stems, sr, "C"


def make_bass_line_clip(roots: list[int], seconds_per_note: float, sr: int, path: str) -> None:
    """Write a real WAV file playing a sequence of held single low notes
    (real additive sine synthesis, an octave down from the octave-3 chords
    elsewhere in this file, plus a quiet octave-up harmonic for a more
    realistic bass timbre) — a stand-in for an isolated bass stem, where
    only one pitch class is ever sounding at a time, not a full chord.
    `roots` is a list of root pitch classes (0=C .. 11=B).
    """
    samples_per_note = int(seconds_per_note * sr)
    t = np.arange(samples_per_note) / sr

    segments = []
    for root_pc in roots:
        fundamental = note_freq(root_pc, octave=2)
        wave = np.sin(2 * np.pi * fundamental * t) + 0.3 * np.sin(2 * np.pi * fundamental * 2 * t)
        segments.append((wave / 1.3).astype(np.float32))

    audio = np.concatenate(segments)
    sf.write(path, audio, sr)


@pytest.fixture
def bass_line_clip(tmp_path):
    path = str(tmp_path / "bass_line.wav")
    # A-C-F-G, one held note each, 2 seconds per note.
    roots = [9, 0, 5, 7]
    make_bass_line_clip(roots, seconds_per_note=2.0, sr=22050, path=path)
    return path, ["A", "C", "F", "G"], 2.0
