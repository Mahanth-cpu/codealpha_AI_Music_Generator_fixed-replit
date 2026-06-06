"""
Music Generation Module — AI Music Generator
Converts model predictions → MIDI files
Usage: python generate.py --num_notes 200 --temperature 1.0 --output my_music
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import (
    ensure_dir, load_pickle, load_json, save_json,
    temperature_sample, logger,
)

# ────── Optional imports ──────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

try:
    import music21
    from music21 import note as m21_note, chord as m21_chord, stream, duration, tempo
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
#  MIDI Writer
# ──────────────────────────────────────────────────────────────────────────────

class MIDIWriter:
    """Convert a list of note/chord tokens into a MIDI file."""

    NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    def __init__(
        self,
        tempo_bpm:    int   = 120,
        output_dir:   str   = "generated_music",
        default_duration: float = 0.5,
    ) -> None:
        self.tempo_bpm        = tempo_bpm
        self.output_dir       = output_dir
        self.default_duration = default_duration
        ensure_dir(output_dir)

    # ------------------------------------------------------------------
    def write(self, notes: List[str], filename: str) -> str:
        """
        Write *notes* to a MIDI file.
        Tries music21 first, falls back to midiutil, then pretty_midi.
        Returns the path to the saved file.
        """
        if not filename.lower().endswith(".mid"):
            filename += ".mid"
        filepath = os.path.join(self.output_dir, filename)

        if MUSIC21_AVAILABLE:
            self._write_music21(notes, filepath)
        elif MIDIUTIL_AVAILABLE:
            self._write_midiutil(notes, filepath)
        elif PRETTY_MIDI_AVAILABLE:
            self._write_prettymidi(notes, filepath)
        else:
            raise RuntimeError(
                "No MIDI library available. "
                "Install music21, midiutil, or pretty_midi."
            )

        logger.info(f"MIDI saved → {filepath}")
        return filepath

    # ── music21 writer ───────────────────────────────────────────────

    def _write_music21(self, notes: List[str], filepath: str) -> None:
        output_stream = stream.Stream()
        output_stream.append(tempo.MetronomeMark(number=self.tempo_bpm))
        note_dur = duration.Duration(self.default_duration)

        for token in notes:
            try:
                if "." in token:
                    # chord
                    pitches = token.split(".")
                    c = m21_chord.Chord(pitches)
                    c.duration = note_dur
                    output_stream.append(c)
                else:
                    n = m21_note.Note(token)
                    n.duration = note_dur
                    output_stream.append(n)
            except Exception:
                rest = m21_note.Rest()
                rest.duration = note_dur
                output_stream.append(rest)

        output_stream.write("midi", fp=filepath)

    # ── midiutil writer ──────────────────────────────────────────────

    def _write_midiutil(self, notes: List[str], filepath: str) -> None:
        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.tempo_bpm)
        beat = self.default_duration
        time_pos = 0.0

        for token in notes:
            try:
                if "." in token:
                    for pitch_str in token.split("."):
                        midi_num = self._pitch_to_midi(pitch_str)
                        if midi_num:
                            midi.addNote(0, 0, midi_num, time_pos, beat, 80)
                else:
                    midi_num = self._pitch_to_midi(token)
                    if midi_num:
                        midi.addNote(0, 0, midi_num, time_pos, beat, 80)
            except Exception:
                pass
            time_pos += beat

        with open(filepath, "wb") as fh:
            midi.writeFile(fh)

    # ── pretty_midi writer ───────────────────────────────────────────

    def _write_prettymidi(self, notes: List[str], filepath: str) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo_bpm)
        piano = pretty_midi.Instrument(program=0)
        beat = self.default_duration
        t = 0.0

        for token in notes:
            try:
                pitches = token.split(".") if "." in token else [token]
                for p in pitches:
                    midi_num = self._pitch_to_midi(p)
                    if midi_num:
                        piano.notes.append(
                            pretty_midi.Note(velocity=80, pitch=midi_num,
                                             start=t, end=t + beat)
                        )
            except Exception:
                pass
            t += beat

        pm.instruments.append(piano)
        pm.write(filepath)

    # ── helpers ──────────────────────────────────────────────────────

    def _pitch_to_midi(self, pitch_str: str) -> Optional[int]:
        """Convert a pitch string like 'C4' or 'F#5' to a MIDI number."""
        try:
            s = pitch_str.strip()
            # Extract letter + accidental
            if len(s) >= 2 and s[-1].isdigit():
                octave = int(s[-1])
                name   = s[:-1]
            elif len(s) >= 3 and s[-2:].lstrip("-").isdigit():
                octave = int(s[-2:])
                name   = s[:-2]
            else:
                return None
            if name not in self.NOTE_ORDER:
                return None
            return self.NOTE_ORDER.index(name) + (octave + 1) * 12
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
#  Generator
# ──────────────────────────────────────────────────────────────────────────────

class MusicGenerator:
    """
    Load a trained model + vocabulary and generate new note sequences.
    """

    MODEL_PATH  = "models/{arch}_best.keras"
    CACHE_PATH  = "data/processed_notes.pkl"
    CONFIG_PATH = "models/model_config.json"

    def __init__(
        self,
        output_dir: str = "generated_music",
        tempo_bpm:  int = 120,
    ) -> None:
        self.output_dir = output_dir
        self.tempo_bpm  = tempo_bpm

        self.model       = None
        self.note_to_int: Dict[str, int] = {}
        self.int_to_note: Dict[int, str] = {}
        self.n_vocab     = 0
        self.sequence_length = 64
        self.all_notes: List[str] = []

        self._writer = MIDIWriter(tempo_bpm=tempo_bpm, output_dir=output_dir)

    # ------------------------------------------------------------------
    def load(self, architecture: str = "stacked_lstm") -> "MusicGenerator":
        """Load model weights and vocabulary from disk."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for generation.")

        # Load vocabulary from cache
        try:
            cache = load_pickle(self.CACHE_PATH)
            self.note_to_int     = cache["note_to_int"]
            self.int_to_note     = cache["int_to_note"]
            self.n_vocab         = cache["n_vocab"]
            self.all_notes       = cache["all_notes"]
        except FileNotFoundError:
            raise FileNotFoundError(
                "Vocabulary cache not found. Run preprocessing first."
            )

        # Load model config for sequence length
        try:
            cfg = load_json(self.CONFIG_PATH)
            self.sequence_length = cfg.get("sequence_length", 64)
        except FileNotFoundError:
            logger.warning("model_config.json not found — using default seq_len=64.")

        # Load model weights
        model_path = self.MODEL_PATH.format(arch=architecture)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Train the model first with: python train.py"
            )

        self.model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from '{model_path}'")
        return self

    # ------------------------------------------------------------------
    def generate(
        self,
        num_notes:   int   = 200,
        temperature: float = 1.0,
        seed_notes:  Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate *num_notes* tokens using temperature sampling.

        Parameters
        ----------
        num_notes   : how many notes to generate
        temperature : creativity (0.5 = conservative, 1.5 = experimental)
        seed_notes  : optional list of seed notes; random if None
        """
        if self.model is None:
            raise RuntimeError("Call load() before generate().")

        # Build seed sequence
        if seed_notes is None or len(seed_notes) < self.sequence_length:
            start = random.randint(0, max(0, len(self.all_notes) - self.sequence_length))
            seed_notes = self.all_notes[start : start + self.sequence_length]
            if len(seed_notes) < self.sequence_length:
                seed_notes = list(self.note_to_int.keys())[:self.sequence_length]

        pattern = list(seed_notes[-self.sequence_length:])
        generated: List[str] = []

        logger.info(
            f"Generating {num_notes} notes "
            f"(temperature={temperature}, seed_len={len(pattern)})"
        )
        t0 = time.time()

        for _ in range(num_notes):
            # Encode current window
            x = np.array(
                [[self.note_to_int.get(n, 0) / self.n_vocab for n in pattern]],
                dtype=np.float32,
            ).reshape(1, self.sequence_length, 1)

            preds = self.model.predict(x, verbose=0)[0]
            idx   = temperature_sample(preds, temperature)
            token = self.int_to_note.get(idx, pattern[-1])

            generated.append(token)
            pattern.append(token)
            pattern = pattern[1:]  # slide window

        elapsed = time.time() - t0
        logger.info(f"Generated {len(generated)} notes in {elapsed:.2f}s")
        return generated

    # ------------------------------------------------------------------
    def generate_and_save(
        self,
        num_notes:   int   = 200,
        temperature: float = 1.0,
        output_name: str   = "generated",
        seed_notes:  Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Generate music and write to MIDI.
        Returns (midi_filepath, note_list).
        """
        notes = self.generate(num_notes, temperature, seed_notes)
        midi_path = self._writer.write(notes, output_name)

        # Save metadata
        meta = {
            "output_file":   midi_path,
            "num_notes":     len(notes),
            "temperature":   temperature,
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_tokens": notes[:20],
        }
        meta_path = midi_path.replace(".mid", "_meta.json")
        save_json(meta, meta_path)

        return midi_path, notes

    # ------------------------------------------------------------------
    def generate_variations(
        self,
        num_notes:    int         = 150,
        temperatures: List[float] = None,
        base_name:    str         = "variation",
    ) -> List[Tuple[str, float]]:
        """
        Generate multiple variations at different creativity levels.
        Returns list of (filepath, temperature).
        """
        if temperatures is None:
            temperatures = [0.5, 1.0, 1.5]

        results = []
        seed = self.all_notes[:self.sequence_length] if self.all_notes else None

        for t in temperatures:
            name  = f"{base_name}_temp{str(t).replace('.','_')}"
            path, _ = self.generate_and_save(
                num_notes=num_notes, temperature=t,
                output_name=name, seed_notes=seed,
            )
            results.append((path, t))
            logger.info(f"Saved variation (temp={t}) → {path}")

        return results

    # ------------------------------------------------------------------
    def get_generation_stats(self, notes: List[str]) -> Dict:
        """Compute simple statistics about a generated note sequence."""
        from collections import Counter
        counter = Counter(notes)
        unique  = len(counter)
        chords  = sum(1 for n in notes if "." in n)
        return {
            "total_notes":    len(notes),
            "unique_tokens":  unique,
            "diversity_pct":  round(unique / max(len(notes), 1) * 100, 2),
            "chord_count":    chords,
            "note_count":     len(notes) - chords,
            "top_5_tokens":   counter.most_common(5),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate music with trained LSTM")
    parser.add_argument("--arch",        default="stacked_lstm",
                        choices=["basic_lstm", "bidirectional_lstm",
                                 "stacked_lstm", "transformer"])
    parser.add_argument("--num_notes",   type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Creativity (0.5=conservative, 1.5=experimental)")
    parser.add_argument("--output",      default="generated_music",
                        help="Output filename (without .mid)")
    parser.add_argument("--variations",  action="store_true",
                        help="Generate 3 variations at different temperatures")
    args = parser.parse_args()

    gen = MusicGenerator()
    gen.load(architecture=args.arch)

    if args.variations:
        results = gen.generate_variations(num_notes=args.num_notes, base_name=args.output)
        print("\nGenerated variations:")
        for path, temp in results:
            print(f"  temp={temp} → {path}")
    else:
        path, notes = gen.generate_and_save(
            num_notes   = args.num_notes,
            temperature = args.temperature,
            output_name = args.output,
        )
        stats = gen.get_generation_stats(notes)
        print(f"\n✅ MIDI saved → {path}")
        print(f"   Total notes   : {stats['total_notes']}")
        print(f"   Unique tokens : {stats['unique_tokens']}")
        print(f"   Diversity     : {stats['diversity_pct']}%")
        print(f"   Chords        : {stats['chord_count']}")
