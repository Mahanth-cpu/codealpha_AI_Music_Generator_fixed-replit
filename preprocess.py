"""
Data Collection & Preprocessing Module
AI Music Generator — LSTM-based approach
"""

from __future__ import annotations

import os
import time
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    ensure_dir, find_midi_files, save_pickle, load_pickle,
    save_json, load_json, create_sequences, encode_sequences,
    plot_note_distribution, plot_song_lengths, plot_piano_roll,
    logger,
)

# ────── Optional music21 import ──────
try:
    import music21
    from music21 import converter, instrument, note, chord, stream
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("music21 not installed — using stub extractor.")


# ──────────────────────────────────────────────────────────────────────────────
#  MIDI Parsing
# ──────────────────────────────────────────────────────────────────────────────

class MIDIParser:
    """Parse a single MIDI file into a list of note/chord strings."""

    def __init__(self, min_notes: int = 10) -> None:
        self.min_notes = min_notes

    # ------------------------------------------------------------------
    def parse(self, filepath: str) -> Optional[List[str]]:
        """
        Return list of note tokens for one MIDI file.
        Returns None if the file is corrupt or too short.
        """
        if not MUSIC21_AVAILABLE:
            return self._stub_parse(filepath)

        try:
            score = converter.parse(filepath)
        except Exception as exc:
            logger.warning(f"Could not parse '{filepath}': {exc}")
            return None

        try:
            parts = instrument.partitionByInstrument(score)
            elements = parts.parts[0].recurse() if parts and parts.parts else score.flat.notes
        except Exception:
            elements = score.flat.notes

        tokens: List[str] = []
        for element in elements:
            if isinstance(element, note.Note):
                tokens.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                tokens.append(".".join(str(p) for p in element.pitches))

        if len(tokens) < self.min_notes:
            logger.debug(f"Skipping '{filepath}' — only {len(tokens)} notes.")
            return None

        return tokens

    # ------------------------------------------------------------------
    @staticmethod
    def _stub_parse(filepath: str) -> List[str]:
        """Fallback: generate random tokens when music21 is absent."""
        rng = np.random.default_rng(seed=abs(hash(filepath)) % (2**31))
        pitch_classes = [
            "C4", "D4", "E4", "F4", "G4", "A4", "B4",
            "C5", "D5", "E5", "F5", "G5",
            "C4.E4.G4", "D4.F4.A4", "G4.B4.D5",
        ]
        length = int(rng.integers(100, 400))
        return rng.choice(pitch_classes, size=length).tolist()


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset Statistics
# ──────────────────────────────────────────────────────────────────────────────

class DatasetStats:
    """Compute and format dataset-level statistics."""

    def __init__(self, all_notes: List[str], song_lengths: List[int]) -> None:
        self.all_notes = all_notes
        self.song_lengths = song_lengths
        self.counter = Counter(all_notes)

    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        return {
            "num_songs": len(self.song_lengths),
            "total_notes": len(self.all_notes),
            "unique_tokens": len(self.counter),
            "avg_song_length": float(np.mean(self.song_lengths)) if self.song_lengths else 0,
            "min_song_length": int(np.min(self.song_lengths)) if self.song_lengths else 0,
            "max_song_length": int(np.max(self.song_lengths)) if self.song_lengths else 0,
            "top_10_notes": self.counter.most_common(10),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 55)
        print("  📊  DATASET STATISTICS")
        print("=" * 55)
        print(f"  Songs          : {s['num_songs']}")
        print(f"  Total tokens   : {s['total_notes']:,}")
        print(f"  Unique tokens  : {s['unique_tokens']:,}")
        print(f"  Avg song len   : {s['avg_song_length']:.1f} notes")
        print(f"  Min / Max len  : {s['min_song_length']} / {s['max_song_length']}")
        print("\n  Top-10 tokens:")
        for tok, cnt in s["top_10_notes"]:
            bar = "█" * min(30, cnt // max(1, s["total_notes"] // 300))
            print(f"    {tok:<20} {cnt:>6}  {bar}")
        print("=" * 55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Preprocessor (main class)
# ──────────────────────────────────────────────────────────────────────────────

class MusicPreprocessor:
    """
    End-to-end pipeline:
        1. Load MIDI files
        2. Extract notes / chords
        3. Build vocabulary mappings
        4. Create LSTM sequences
        5. Save / load processed data
    """

    CACHE_FILE   = "data/processed_notes.pkl"
    VOCAB_FILE   = "data/vocabulary.json"
    STATS_FILE   = "data/dataset_stats.json"

    def __init__(
        self,
        sequence_length: int = 64,
        min_note_frequency: int = 1,
    ) -> None:
        self.sequence_length    = sequence_length
        self.min_note_frequency = min_note_frequency

        # Populated after processing
        self.all_notes:    List[str]       = []
        self.song_lengths: List[int]       = []
        self.note_to_int:  Dict[str, int]  = {}
        self.int_to_note:  Dict[int, str]  = {}
        self.n_vocab:      int             = 0
        self.stats:        Optional[DatasetStats] = None

        self._parser = MIDIParser()

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return bool(self.all_notes) and self.n_vocab > 0

    # ── Public API ──────────────────────────────────────────────────

    def load_midi_directory(
        self,
        directory: str,
        max_files: Optional[int] = None,
        force_reload: bool = False,
    ) -> "MusicPreprocessor":
        """Load and parse all MIDI files in *directory*."""
        cache = self.CACHE_FILE
        if not force_reload and os.path.exists(cache):
            logger.info("Loading cached note data…")
            self._load_cache()
            return self

        midi_files = find_midi_files(directory)
        if not midi_files:
            raise ValueError(f"No MIDI files found in '{directory}'.")

        if max_files:
            midi_files = midi_files[:max_files]

        logger.info(f"Parsing {len(midi_files)} MIDI file(s)…")
        t0 = time.time()
        failed = 0

        for fp in midi_files:
            tokens = self._parser.parse(fp)
            if tokens is None:
                failed += 1
                continue
            self.all_notes.extend(tokens)
            self.song_lengths.append(len(tokens))

        elapsed = time.time() - t0
        logger.info(
            f"Parsed {len(self.song_lengths)} files in {elapsed:.1f}s "
            f"({failed} failed). Total tokens: {len(self.all_notes):,}"
        )

        if not self.all_notes:
            raise ValueError("No notes extracted from any MIDI file.")

        self._build_vocabulary()
        self._save_cache()
        return self

    # ------------------------------------------------------------------
    def load_notes_list(self, notes: List[str]) -> "MusicPreprocessor":
        """Directly set all_notes (useful for testing / Streamlit uploads)."""
        self.all_notes  = notes
        self.song_lengths = [len(notes)]
        self._build_vocabulary()
        return self

    # ------------------------------------------------------------------
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) NumPy arrays for model training."""
        if not self.is_ready:
            raise RuntimeError("No data loaded. Call load_midi_directory() first.")
        inputs, targets = create_sequences(self.all_notes, self.sequence_length)
        X, y = encode_sequences(inputs, targets, self.note_to_int, self.n_vocab)
        logger.info(f"Training data — X: {X.shape}, y: {y.shape}")
        return X, y

    # ------------------------------------------------------------------
    def get_stats(self) -> DatasetStats:
        if self.stats is None:
            self.stats = DatasetStats(self.all_notes, self.song_lengths)
        return self.stats

    # ------------------------------------------------------------------
    def run_eda(self, asset_dir: str = "assets") -> Dict[str, str]:
        """Run EDA and save plots; return dict of {plot_name: filepath}."""
        ensure_dir(asset_dir)
        stats = self.get_stats()
        paths: Dict[str, str] = {}

        note_path = os.path.join(asset_dir, "note_distribution.png")
        plot_note_distribution(dict(stats.counter), note_path)
        paths["note_distribution"] = note_path

        chord_counts = {k: v for k, v in stats.counter.items() if "." in k}
        if chord_counts:
            chord_path = os.path.join(asset_dir, "chord_distribution.png")
            plot_note_distribution(chord_counts, chord_path, top_n=15)
            paths["chord_distribution"] = chord_path

        length_path = os.path.join(asset_dir, "song_lengths.png")
        plot_song_lengths(self.song_lengths, length_path)
        paths["song_lengths"] = length_path

        roll_path = os.path.join(asset_dir, "piano_roll.png")
        plot_piano_roll(self.all_notes, roll_path)
        paths["piano_roll"] = roll_path

        stats.print_summary()
        return paths

    # ── Private helpers ─────────────────────────────────────────────

    def _build_vocabulary(self) -> None:
        """Build note↔int mappings after filtering rare tokens."""
        counter = Counter(self.all_notes)
        vocab = sorted(
            tok for tok, cnt in counter.items()
            if cnt >= self.min_note_frequency
        )
        self.note_to_int = {n: i for i, n in enumerate(vocab)}
        self.int_to_note = {i: n for n, i in self.note_to_int.items()}
        self.n_vocab = len(vocab)

        # Filter out-of-vocab tokens
        self.all_notes = [n for n in self.all_notes if n in self.note_to_int]

        logger.info(f"Vocabulary size: {self.n_vocab}")

    def _save_cache(self) -> None:
        ensure_dir("data")
        save_pickle({
            "all_notes":    self.all_notes,
            "song_lengths": self.song_lengths,
            "note_to_int":  self.note_to_int,
            "int_to_note":  self.int_to_note,
            "n_vocab":      self.n_vocab,
        }, self.CACHE_FILE)
        save_json(self.get_stats().summary(), self.STATS_FILE)
        logger.info("Preprocessor cache saved.")

    def _load_cache(self) -> None:
        data = load_pickle(self.CACHE_FILE)
        self.all_notes    = data["all_notes"]
        self.song_lengths = data["song_lengths"]
        self.note_to_int  = data["note_to_int"]
        self.int_to_note  = data["int_to_note"]
        self.n_vocab      = data["n_vocab"]
        logger.info(f"Cache loaded — {self.n_vocab} vocab tokens.")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MIDI dataset")
    parser.add_argument("--data_dir",   default="data",       help="Directory containing MIDI files")
    parser.add_argument("--seq_len",    type=int, default=64,  help="Sequence window length")
    parser.add_argument("--max_files",  type=int, default=None)
    parser.add_argument("--force",      action="store_true",   help="Force re-parse even if cache exists")
    parser.add_argument("--eda",        action="store_true",   help="Run EDA and save plots")
    args = parser.parse_args()

    pre = MusicPreprocessor(sequence_length=args.seq_len)
    pre.load_midi_directory(args.data_dir, max_files=args.max_files, force_reload=args.force)
    pre.get_stats().print_summary()

    if args.eda:
        paths = pre.run_eda()
        print("\nEDA plots saved:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
