from .analyze import AnalysisResult, analyze
from .beats import BeatResult, detect_beats
from .chords import ChordSegment, detect_chords
from .instruments import FretPosition, PRESET_TUNINGS, Tuning, fretboard_positions
from .key import KeyResult, detect_key
from .preprocessing import load_audio
from .tempo import detect_bpm
from .theory import (
    ChordFunction,
    DiatonicChord,
    Scale,
    analyze_chord_function,
    build_scale,
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
    "analyze_chord_function",
    "ChordFunction",
    "relative_key",
    "parallel_key",
    "Tuning",
    "PRESET_TUNINGS",
    "fretboard_positions",
    "FretPosition",
]
