# Chromalyze

Standalone audio analysis engine — tempo, beats, key, chords, and scale from
a raw audio file. Built on top of [Librosa](https://librosa.org/), licensed
ISC (same as Librosa) so it can be embedded in closed-source applications
without copyleft obligations.

Not tied to any particular application — the public API takes a file path
and returns plain data, nothing else.

## Status

Audio analysis (tempo/beats/key/chords) and the full planned music theory
layer are both complete:

- Preprocessing (mono load + resample)
- BPM detection
- Beat tracking
- Key detection (Krumhansl-Schmuckler profile correlation over Librosa chroma)
- Chord recognition (chroma template matching, segmented on real detected
  beats rather than arbitrary fixed-time windows — see `detect_chords`)
- Music theory layer: scale spelling (proper key-signature-aware letter
  spelling, not a fixed chromatic lookup), diatonic triads *and* seventh
  chords with roman numerals for any of the 7 modes, arbitrary chord
  spelling from any root + quality, relative/parallel key, and roman-numeral
  analysis of arbitrary (including chromatic/borrowed) chords against a key
- Interval theory: letter-aware naming and quality (e.g. "Major Third",
  "Augmented Fourth") for the distance between any two notes, plus a
  semitone-only lookup for when no note spelling is available
- Instrument/tuning layer: maps a scale or chord onto real fretboard
  positions for guitar (standard, drop tunings, alternate tunings, 7- and
  8-string) and bass (standard, drop tunings, 5- and 6-string) — piano/
  keyboard needs no equivalent, since a scale's pitch classes map directly
  onto piano keys with nothing instrument-specific to account for
- Named non-diatonic scales: major/minor pentatonic, blues, harmonic minor,
  melodic minor (jazz/ascending form) — see `build_named_scale`
- Chord progression library: a catalog of well-known progressions (Pop,
  '50s, jazz ii-V-I, 12-bar blues, Andalusian cadence, and more), realizable
  in any key and matchable against a real detected chord sequence — see
  `realize_progression`/`identify_progression`
- CAGED system (standard 6-string guitar tuning only): the 5 moveable
  C/A/G/E/D major-chord shapes for any root, and the 5 scale "box" patterns
  built around them — see `caged_chord_shapes`/`caged_scale_boxes`
- Beats-per-measure estimation (a best-effort meter heuristic, not full
  time-signature detection — see caveats below) — see `detect_beats_per_measure`

## Usage

```python
from chromalyze import analyze

result = analyze("song.wav")
result.bpm             # 128.3
result.beats            # [0.49, 0.95, 1.42, ...] — seconds
result.key              # "C major"
result.key_confidence   # 0.27 — gap between best and 2nd-best key candidate;
                         # under ~0.05 usually means genuine relative
                         # major/minor ambiguity (they share every pitch
                         # class), over ~0.15 is a clear, confident match
result.chords           # [ChordSegment(start=0.0, end=1.53, chord="G", correlation=0.99), ...]
result.scale            # Scale(tonic="C", mode="ionian", notes=["C","D","E","F","G","A","B"], pitch_classes=[0,2,4,5,7,9,11])
result.beats_per_measure             # 4 — best-effort estimate, see "Beats-per-measure" caveats below
result.beats_per_measure_confidence  # 0.0-ish means don't trust it; a clean, accented track scores much higher
```

Individual stages are also available directly:

```python
from chromalyze import load_audio, detect_bpm, detect_beats, detect_key, detect_chords

y, sr = load_audio("song.wav")
bpm = detect_bpm(y, sr)
beats = detect_beats(y, sr)  # BeatResult(bpm=..., beat_times=[...])
key = detect_key(y, sr)      # KeyResult(tonic="C", mode="major", key="C major", correlation=0.97, confidence=0.27)

# Beat-synchronous segmentation (recommended — see analyze() above) avoids
# analyzing a window that straddles two different chords, which can
# produce a spurious chord that was never actually played:
chords = detect_chords(y, sr, beat_times=beats.beat_times)
# Or, standalone, without beat detection: fixed-length windows instead.
chords = detect_chords(y, sr, segment_seconds=1.0)

# Best-effort "how many beats per measure" estimate:
from chromalyze import detect_beats_per_measure
meter = detect_beats_per_measure(y, sr, beats.beat_times)  # TimeSignatureResult(beats_per_measure=4, confidence=1.2)
```

**A real caveat on `detect_beats_per_measure`**: this is *not* full time-signature
detection. True downbeat tracking (knowing which beat is "beat 1" of a
measure) is a research-grade MIR problem state-of-the-art tools solve with
trained neural nets (e.g. madmom's DBN-based downbeat tracker) — real extra
weight and complexity this package doesn't take on. Instead, this is a
simpler heuristic: downbeats are usually the most rhythmically accented
beat in a measure (a kick drum on beat 1, for example), so it autocorrelates
a per-beat loudness signal to find the periodicity (2, 3, or 4 beats) that
best explains the accent pattern. Verified against synthesized click tracks
with a known accent pattern before shipping — it reliably recovers the
right answer on a clean, consistently-accented track, and reports near-zero
confidence (rather than a false answer) when there's no real accent
pattern to find. It will do noticeably worse on syncopated, sparsely
percussive, or accent-ambiguous real music — always check `confidence`
before trusting `beats_per_measure`, the same way you'd check
`key_confidence` before trusting `key`.

### Music theory

```python
from chromalyze import build_scale, diatonic_triads, diatonic_sevenths, build_chord, analyze_chord_function, relative_key, parallel_key

scale = build_scale("G", "major")       # Scale(notes=["G","A","B","C","D","E","F#"], ...)
triads = diatonic_triads(scale)         # [DiatonicChord(degree=1, root="G", quality="major", roman_numeral="I"), ...]
sevenths = diatonic_sevenths(scale)     # [DiatonicSeventhChord(degree=1, root="G", quality="major7", roman_numeral="Imaj7"), ...]
relative_key("G", "major")              # ("E", "aeolian") — relative minor
parallel_key("G", "major")              # ("G", "aeolian") — parallel minor

# Build any chord (triad or seventh) from a root + quality, independent of
# any scale or key — spelled with real letter names, not just pitch classes:
build_chord("D", "minor7")              # Chord(root="D", quality="minor7", notes=["D","F","A","C"], pitch_classes=[2,5,9,0])
build_chord("B", "diminished7")         # notes=["B","D","F","Ab"] — spelled Ab, not G#, to keep one letter per chord tone

# Label a detected chord's harmonic function within a key, including
# chromatic/borrowed chords (e.g. a "bVII" chord borrowed from the parallel minor):
analyze_chord_function("F", "major", key_tonic="G", key_mode="major")
# ChordFunction(roman_numeral="bVII", is_diatonic=False)
```

`CHORD_INTERVALS` (importable directly) lists every quality `build_chord` and
the diatonic-chord builders understand: `major`, `minor`, `diminished`,
`augmented`, `major7`, `dominant7`, `minor7`, `minor-major7`,
`half-diminished7`, `diminished7`, `augmented-major7`, `augmented7`.

### Named scales

```python
from chromalyze import build_named_scale, NAMED_SCALE_INTERVALS, diatonic_triads

build_named_scale("C", "major_pentatonic")  # Scale(notes=["C","D","E","G","A"], ...)
build_named_scale("A", "minor_pentatonic")  # Scale(notes=["A","C","D","E","G"], ...)
build_named_scale("C", "blues")             # Scale(notes=["C","Eb","F","Gb","G","Bb"], ...)
build_named_scale("A", "harmonic_minor")    # Scale(notes=["A","B","C","D","E","F","G#"], ...)
build_named_scale("A", "melodic_minor")     # Scale(notes=["A","B","C","D","E","F#","G#"], ...)

# Harmonic/melodic minor are ordinary 7-note scales, so the diatonic-chord
# tools from the "Music theory" section above work on them directly — e.g.
# harmonic minor's raised 7th gives it a real dominant V (not natural
# minor's v) and a distinctive augmented bIII chord:
diatonic_triads(build_named_scale("A", "harmonic_minor"))
# [..., DiatonicChord(degree=3, root="C", quality="augmented", roman_numeral="III+"), ...,
#      DiatonicChord(degree=5, root="E", quality="major", roman_numeral="V"), ...]

sorted(NAMED_SCALE_INTERVALS)  # ['blues', 'harmonic_minor', 'major_pentatonic', 'melodic_minor', 'minor_pentatonic']
```

### Chord progressions

```python
from chromalyze import NAMED_PROGRESSIONS, realize_progression, identify_progression

# Realize a named progression in any key — same I-V-vi-IV shape, two keys:
realize_progression(NAMED_PROGRESSIONS["pop"], "C")
# [ProgressionChord(roman_numeral="I", root="C", quality="major", notes=["C","E","G"], ...),
#  ProgressionChord(roman_numeral="V", root="G", ...), ("vi", "A", "minor"), ("IV", "F", "major")]
realize_progression(NAMED_PROGRESSIONS["pop"], "G")   # G-D-Em-C — same shape, different key

realize_progression(NAMED_PROGRESSIONS["andalusian_cadence"], "A")
# i(Am) - VII(G) - VI(F) - V(E major) — the closing E is a borrowed/altered
# dominant (with G#), not natural A minor's own (diatonic, minor-quality) v

# Check whether a real chord sequence (e.g. from detect_chords) matches a
# known progression, given the key it's in:
identify_progression([("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")], key_tonic="C", key_mode="major")
# ["pop"]
```

`NAMED_PROGRESSIONS` also includes `fifties`, `three_chord`, `jazz_ii_v_i`,
`pachelbels_canon`, `authentic_cadence`, `plagal_cadence`,
`deceptive_cadence`, `twelve_bar_blues`, `minor_three_chord`, and
`minor_pop` — each with a `.description`.

### Intervals

```python
from chromalyze import interval_between, interval_from_semitones, common_interval_reference

interval_between("C", "E")     # Interval(degree=3, quality="major", semitones=4, name="Major Third", short_name="M3")
interval_between("F", "B")     # Interval(degree=4, quality="augmented", semitones=6, name="Augmented Fourth", short_name="A4")
interval_between("B", "F")     # Interval(degree=5, quality="diminished", semitones=6, name="Diminished Fifth", short_name="d5")
# F->B and B->F are both 6 semitones apart, but letter distance (a 4th vs
# a 5th) makes them genuinely different intervals, not just two names for
# the same thing — this only works with real note names, which is why
# `interval_between` takes spelled notes rather than bare pitch classes.

# No note spelling on hand (e.g. straight from a raw pitch-class
# difference)? Falls back to music's single most conventional name:
interval_from_semitones(6)     # "Augmented Fourth" — the tritone's usual spelling

# A ready-made reference table, handy for teaching:
for i in common_interval_reference():
    print(i.short_name, i.name)  # P1 Perfect Unison, m2 Minor Second, M2 Major Second, ...
```

### Instruments

```python
from chromalyze import build_scale, fretboard_positions, PRESET_TUNINGS

scale = build_scale("C", "major")
positions = fretboard_positions(scale.pitch_classes, PRESET_TUNINGS["guitar_7string_standard"], num_frets=24)
# [FretPosition(string_index=0, fret=0, pitch_class=11, scale_degree=7), ...]

# Custom/alternate tunings aren't limited to the presets:
from chromalyze import Tuning
my_tuning = Tuning.from_note_names("My Tuning", ["D", "A", "D", "G", "A", "D"])  # DADGAD
```

Presets in `PRESET_TUNINGS`: `guitar_standard`, `guitar_drop_d`, `guitar_drop_c`,
`guitar_drop_b`, `guitar_half_step_down`, `guitar_dadgad`, `guitar_open_d`,
`guitar_open_g`, `guitar_7string_standard`, `guitar_7string_drop_a`,
`guitar_8string_standard`, `bass_standard`, `bass_drop_d`, `bass_half_step_down`,
`bass_5string_standard`, `bass_5string_high_c`, `bass_6string_standard`.

### CAGED (standard 6-string guitar tuning)

```python
from chromalyze import caged_chord_shapes, caged_scale_boxes, build_scale

# The 5 moveable ways to play a C major chord, in canonical C-A-G-E-D
# order — landing at the exact frets every guitarist learns them at:
for voicing in caged_chord_shapes("C"):
    print(voicing.shape, [(p.string_index, p.fret, p.role) for p in voicing.positions])
# C  -> frets starting at 0 (the literal open C chord)
# A  -> frets starting at 3  (the classic "A-shape barre" C at the 3rd fret)
# G  -> frets starting at 5
# E  -> frets starting at 8  (the classic "E-shape barre" C at the 8th fret)
# D  -> frets starting at 10

# The 5 CAGED scale-box patterns for a scale, same C-A-G-E-D order,
# covering the full neck between them:
scale = build_scale("C", "major")
for box in caged_scale_boxes("C", scale.pitch_classes):
    print(box.shape, box.min_fret, "-", box.max_fret, len(box.positions), "positions")
```

CAGED is specific to standard EADGBE tuning — the 5 shapes look different
from each other specifically because of that tuning's string-interval
pattern, so this doesn't extend to drop tunings, 7/8-string, or bass.

## Development

```bash
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
```

## License

ISC — see [LICENSE](LICENSE).
