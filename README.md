# Chromalyze

Standalone audio analysis engine — tempo, beats, key, chords, and scale from
a raw audio file. Built on top of [Librosa](https://librosa.org/), licensed
ISC (same as Librosa) so it can be embedded in closed-source applications
without copyleft obligations.

Not tied to any particular application — the public API takes a file path
and returns plain data, nothing else.

## Status

This is under active, staged development. Currently implemented:

- Preprocessing (mono load + resample)
- BPM detection
- Beat tracking
- Key detection (Krumhansl-Schmuckler profile correlation over Librosa chroma)

Not yet implemented (present in the result shape as `None` for now):

- Chord recognition
- Music theory layer (scale degrees, roman numerals, etc.)

## Usage

```python
from chromalyze import analyze

result = analyze("song.wav")
result.bpm     # 128.3
result.beats   # [0.49, 0.95, 1.42, ...] — seconds
result.key             # "C major"
result.key_confidence  # 0.27 — gap between best and 2nd-best key candidate;
                        # under ~0.05 usually means genuine relative
                        # major/minor ambiguity (they share every pitch
                        # class), over ~0.15 is a clear, confident match
result.chords          # None (not yet implemented)
result.scale           # None (not yet implemented)
```

Individual stages are also available directly:

```python
from chromalyze import load_audio, detect_bpm, detect_beats, detect_key

y, sr = load_audio("song.wav")
bpm = detect_bpm(y, sr)
beats = detect_beats(y, sr)  # BeatResult(bpm=..., beat_times=[...])
key = detect_key(y, sr)      # KeyResult(tonic="C", mode="major", key="C major", correlation=0.97, confidence=0.27)
```

## Development

```bash
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
```

## License

ISC — see [LICENSE](LICENSE).
