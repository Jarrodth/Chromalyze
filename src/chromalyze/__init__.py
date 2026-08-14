from .analyze import AnalysisResult, analyze
from .beats import BeatResult, detect_beats
from .caged import (
    CAGED_ORDER,
    CAGED_SHAPES,
    CagedChordPosition,
    CagedChordVoicing,
    CagedScaleBox,
    caged_chord_shapes,
    caged_scale_boxes,
)
from .chords import ChordSegment, detect_chords
from .instruments import (
    FretPosition,
    PRESET_TUNINGS,
    Tuning,
    fretboard_positions,
    practical_chord_voicing,
    practical_power_chord_voicing,
)
from .intervals import Interval, common_interval_reference, interval_between, interval_from_semitones
from .key import KeyResult, detect_key
from .meter import TimeSignatureResult, detect_beats_per_measure
from .preprocessing import load_audio
from .progressions import (
    NAMED_PROGRESSIONS,
    NamedProgression,
    ProgressionChord,
    ProgressionStep,
    identify_progression,
    realize_progression,
)
from .scales import NAMED_SCALE_INTERVALS, build_named_scale
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
    build_power_chord,
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
    "build_power_chord",
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
    "practical_chord_voicing",
    "practical_power_chord_voicing",
    "Interval",
    "interval_between",
    "interval_from_semitones",
    "common_interval_reference",
    "build_named_scale",
    "NAMED_SCALE_INTERVALS",
    "NAMED_PROGRESSIONS",
    "NamedProgression",
    "ProgressionStep",
    "ProgressionChord",
    "realize_progression",
    "identify_progression",
    "CAGED_ORDER",
    "CAGED_SHAPES",
    "CagedChordPosition",
    "CagedChordVoicing",
    "CagedScaleBox",
    "caged_chord_shapes",
    "caged_scale_boxes",
    "TimeSignatureResult",
    "detect_beats_per_measure",
]
