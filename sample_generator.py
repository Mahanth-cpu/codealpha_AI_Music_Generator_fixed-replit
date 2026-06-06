"""
Sample MIDI Generator — No Training Required
Generates a demo MIDI file to showcase the output pipeline.
Run: python sample_generator.py
"""

from __future__ import annotations

import os
import random
from typing import List

# ── Try MIDIUtil first, then pretty_midi ──
try:
    from midiutil import MIDIFile
    MIDIUTIL_OK = True
except ImportError:
    MIDIUTIL_OK = False

try:
    import pretty_midi
    PRETTY_MIDI_OK = True
except ImportError:
    PRETTY_MIDI_OK = False


# ─────────────────── Music Theory Constants ───────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

# Common progressions (chord root semitone offsets from tonic)
PROGRESSIONS = {
    "I-IV-V-I":    [0, 5, 7, 0],
    "I-V-vi-IV":   [0, 7, 9, 5],
    "ii-V-I":      [2, 7, 0],
    "I-vi-IV-V":   [0, 9, 5, 7],
}

# Simple melodic patterns (intervals from current note)
MELODY_PATTERNS = [
    [0, 2, 4, 5, 4, 2, 0],       # ascending scale fragment
    [0, -2, 0, 2, 4, 2, 0],      # arch shape
    [0, 4, 7, 4, 0, -2, 0],      # arpeggio
    [0, 2, 0, -2, 0, 2, 4],      # neighbour motion
]


def semitone_to_midi(root_midi: int, semitone_offset: int) -> int:
    return max(21, min(108, root_midi + semitone_offset))


def build_chord(root_midi: int, quality: str = "major") -> List[int]:
    """Return MIDI note numbers for a triad."""
    intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dom7":  [0, 4, 7, 10],
    }
    return [root_midi + i for i in intervals.get(quality, [0, 4, 7])]


def generate_melody(
    length:    int   = 64,
    root_midi: int   = 60,   # C4
    scale:     str   = "major",
) -> List[int]:
    """Generate a melody as a list of MIDI note numbers."""
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    scale_intervals = major_scale if scale == "major" else minor_scale

    scale_notes = [root_midi + i for i in scale_intervals]
    scale_notes += [n + 12 for n in scale_notes]  # add octave up

    melody: List[int] = []
    current = scale_notes[0]

    for _ in range(length):
        pattern = random.choice(MELODY_PATTERNS)
        step    = random.choice(pattern)
        nxt     = current + step

        # Snap to nearest scale note
        closest = min(scale_notes, key=lambda n: abs(n - nxt))
        melody.append(closest)
        current = closest

    return melody


def write_midi_midiutil(
    melody:     List[int],
    chords:     List[List[int]],
    filepath:   str,
    tempo_bpm:  int = 120,
    beats_per_chord: int = 4,
) -> None:
    midi = MIDIFile(2)   # track 0 = melody, track 1 = chords
    midi.addTempo(0, 0, tempo_bpm)
    midi.addTempo(1, 0, tempo_bpm)

    # Melody track
    beat = 0.0
    for pitch in melody:
        dur = random.choice([0.5, 0.5, 0.5, 1.0, 1.0, 0.25])
        midi.addNote(0, 0, pitch, beat, dur, 70 + random.randint(-10, 10))
        beat += dur

    # Chord track
    beat = 0.0
    for chord_notes in chords:
        for pitch in chord_notes:
            midi.addNote(1, 0, pitch, beat, beats_per_chord - 0.1,
                         55 + random.randint(-5, 5))
        beat += beats_per_chord

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as fh:
        midi.writeFile(fh)


def write_midi_prettymidi(
    melody:     List[int],
    chords:     List[List[int]],
    filepath:   str,
    tempo_bpm:  int = 120,
    beats_per_chord: int = 4,
) -> None:
    seconds_per_beat = 60.0 / tempo_bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)

    piano = pretty_midi.Instrument(program=0, name="Piano")
    t = 0.0
    for pitch in melody:
        dur = random.choice([0.25, 0.5, 0.5, 0.5, 1.0]) * seconds_per_beat
        piano.notes.append(
            pretty_midi.Note(velocity=70, pitch=pitch, start=t, end=t + dur)
        )
        t += dur

    chords_inst = pretty_midi.Instrument(program=48, name="Strings")
    t = 0.0
    chord_dur = beats_per_chord * seconds_per_beat
    for chord_notes in chords:
        for pitch in chord_notes:
            chords_inst.notes.append(
                pretty_midi.Note(velocity=55, pitch=pitch,
                                 start=t, end=t + chord_dur - 0.05)
            )
        t += chord_dur

    pm.instruments.extend([piano, chords_inst])
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    pm.write(filepath)


def generate_sample(
    output_path: str = "generated_music/sample_output.mid",
    tempo_bpm:   int = 120,
    num_bars:    int = 16,
    key:         str = "C",
    mode:        str = "major",
) -> str:
    """
    Generate a sample MIDI composition.
    Returns the path to the saved file.
    """
    random.seed(42)

    # Convert key to MIDI root note
    root_midi = NOTE_NAMES.index(key.upper()) + 60  # octave 4

    # Build chord progression
    prog_name, prog_offsets = random.choice(list(PROGRESSIONS.items()))
    print(f"  Progression : {prog_name}")
    print(f"  Key         : {key} {mode}")
    print(f"  Tempo       : {tempo_bpm} BPM")

    chords = []
    for _ in range(num_bars):
        offset = random.choice(prog_offsets)
        quality = "major" if mode == "major" and offset in [0, 5, 7] else "minor"
        chords.append(build_chord(root_midi + offset - 12, quality))  # one octave lower

    # Build melody
    melody = generate_melody(
        length    = num_bars * 8,
        root_midi = root_midi,
        scale     = mode,
    )

    # Write
    if MIDIUTIL_OK:
        write_midi_midiutil(melody, chords, output_path, tempo_bpm)
        method = "midiutil"
    elif PRETTY_MIDI_OK:
        write_midi_prettymidi(melody, chords, output_path, tempo_bpm)
        method = "pretty_midi"
    else:
        raise RuntimeError("Install midiutil or pretty_midi: pip install midiutil pretty_midi")

    print(f"  Method      : {method}")
    print(f"  Notes       : {len(melody)}")
    print(f"  Saved →       {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a sample MIDI file")
    p.add_argument("--output",  default="generated_music/sample_output.mid")
    p.add_argument("--tempo",   type=int, default=120)
    p.add_argument("--bars",    type=int, default=16)
    p.add_argument("--key",     default="C",     choices=NOTE_NAMES)
    p.add_argument("--mode",    default="major", choices=["major", "minor"])
    args = p.parse_args()

    print("\n🎵 Generating sample MIDI…")
    path = generate_sample(
        output_path = args.output,
        tempo_bpm   = args.tempo,
        num_bars    = args.bars,
        key         = args.key,
        mode        = args.mode,
    )
    print(f"\n✅ Done! Open {path} in any MIDI player.\n")
