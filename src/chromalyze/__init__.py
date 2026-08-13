from .analyze import AnalysisResult, analyze
from .beats import BeatResult, detect_beats
from .key import KeyResult, detect_key
from .preprocessing import load_audio
from .tempo import detect_bpm

__all__ = [
    "analyze",
    "AnalysisResult",
    "detect_beats",
    "BeatResult",
    "detect_key",
    "KeyResult",
    "load_audio",
    "detect_bpm",
]
