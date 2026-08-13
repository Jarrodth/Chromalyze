from .analyze import AnalysisResult, analyze
from .beats import BeatResult, detect_beats
from .chords import ChordSegment, detect_chords
from .instruments import FretPosition, PRESET_TUNINGS, Tuning, fretboard_positions
from .intervals import Interval, common_interval_reference, interval_between, interval_from_semitones
from .key import KeyResult, detect_key
from .preprocessing import load_audio
from .tempo import detect_bpm
from .theory import (
    CHORD_INTERVALS,
    Chord,
    ChordFunction,
    DiatonicChord,
    DiatonicSeventhChord,
    Scale,
    analyze_chord_function,
    build_chord,
    build_scale,
    diatonic_sevenths,
    diatonic_triads,
    parallel_key,
    relative_key,
)

__all__ = [
    "analyze",
    "AnalysisResult",
    "detect_beats",
    "BeatResult",
    "detect_chords",
    "ChordSegment",
    "detect_key",
    "KeyResult",
    "load_audio",
    "detect_bpm",
    "build_scale",
    "Scale",
    "diatonic_triads",
    "DiatonicChord",
    "diatonic_sevenths",
    "DiatonicSeventhChord",
    "build_chord",
    "Chord",
    "CHORD_INTERVALS",
    "analyze_chord_function",
    "ChordFunction",
    "relative_key",
    "parallel_key",
    "Tuning",
    "PRESET_TUNINGS",
    "fretboard_positions",
    "FretPosition",
    "Interval",
    "interval_between",
    "interval_from_semitones",
    "common_interval_reference",
]
