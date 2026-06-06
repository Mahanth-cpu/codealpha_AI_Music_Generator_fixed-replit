"""
Utility functions for AI Music Generator
Author: AI Music Generator Project
"""

import os
import logging
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


# ─────────────────────────── Logging Setup ───────────────────────────

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"music_gen_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("MusicGen")
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ─────────────────────────── File Helpers ───────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, filepath: str) -> None:
    """Serialise object with pickle."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "wb") as fh:
        pickle.dump(obj, fh)
    logger.debug(f"Saved pickle → {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load a pickle file; raise FileNotFoundError if missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle not found: {filepath}")
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data as JSON."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    logger.debug(f"Saved JSON → {filepath}")


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_md5(filepath: str) -> str:
    """Return MD5 hash of a file (used to detect duplicates)."""
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_midi_files(directory: str) -> List[str]:
    """Recursively find all MIDI files (.mid / .midi) in a directory."""
    midi_files: List[str] = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))
    logger.info(f"Found {len(midi_files)} MIDI file(s) in '{directory}'")
    return sorted(midi_files)


# ─────────────────────────── Sequence Helpers ───────────────────────────

def create_sequences(
    notes: List[str],
    sequence_length: int = 64,
) -> Tuple[List[List[str]], List[str]]:
    """
    Build sliding-window input→target pairs.

    Returns
    -------
    inputs  : list of note sequences (each length = sequence_length)
    targets : corresponding next-note labels
    """
    inputs, targets = [], []
    for i in range(len(notes) - sequence_length):
        inputs.append(notes[i : i + sequence_length])
        targets.append(notes[i + sequence_length])
    logger.debug(f"Created {len(inputs)} sequence pairs (window={sequence_length})")
    return inputs, targets


def encode_sequences(
    inputs: List[List[str]],
    targets: List[str],
    note_to_int: Dict[str, int],
    n_vocab: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode string sequences → NumPy arrays ready for Keras.

    X shape : (samples, seq_len, 1)  — normalised integers
    y shape : (samples, n_vocab)     — one-hot
    """
    seq_len = len(inputs[0])
    X = np.zeros((len(inputs), seq_len, 1), dtype=np.float32)
    y = np.zeros((len(targets), n_vocab), dtype=np.float32)

    for i, (seq, tgt) in enumerate(zip(inputs, targets)):
        for t, note in enumerate(seq):
            X[i, t, 0] = note_to_int[note] / n_vocab
        y[i, note_to_int[tgt]] = 1.0

    return X, y


# ─────────────────────────── Temperature Sampling ───────────────────────────

def temperature_sample(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an index from a probability array using temperature scaling.

    temperature < 1 → more conservative / repetitive
    temperature > 1 → more creative / random
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if temperature == 0:
        return int(np.argmax(predictions))
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions - predictions.max())
    probabilities = exp_preds / exp_preds.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


# ─────────────────────────── Plotting Helpers ───────────────────────────

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]


def plot_note_distribution(
    note_counts: Dict[str, int],
    save_path: str,
    top_n: int = 20,
) -> None:
    """Bar chart of the most frequent notes / chords."""
    sorted_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*sorted_notes) if sorted_notes else ([], [])

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(labels)), values, color=PALETTE[: len(labels)], edgecolor="#333", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Note / Chord", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Top-{top_n} Most Frequent Notes & Chords", fontsize=13, fontweight="bold")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved note distribution plot → {save_path}")


def plot_song_lengths(
    lengths: List[int],
    save_path: str,
) -> None:
    """Histogram of per-song note counts."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=30, color=PALETTE[1], edgecolor="#333", linewidth=0.6, alpha=0.85)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(lengths):.0f}")
    ax.set_xlabel("Notes per Song", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Song Length Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved song length plot → {save_path}")


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Dual-axis plot of training loss + accuracy."""
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history_dict.get("loss", []), color=PALETTE[0], linewidth=2, label="Train")
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], color=PALETTE[1], linewidth=2, linestyle="--", label="Validation")
    axes[0].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    acc_key = "accuracy" if "accuracy" in history_dict else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history_dict else "val_acc"
    axes[1].plot(epochs, history_dict.get(acc_key, []), color=PALETTE[2], linewidth=2, label="Train")
    if val_acc_key in history_dict:
        axes[1].plot(epochs, history_dict[val_acc_key], color=PALETTE[3], linewidth=2, linestyle="--", label="Validation")
    axes[1].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history plot → {save_path}")


def plot_piano_roll(
    notes: List[str],
    save_path: str,
    max_notes: int = 200,
) -> None:
    """Simple piano-roll visualisation."""
    NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    def note_to_midi_approx(n: str) -> Optional[int]:
        try:
            base = n.split("-")[0] if "-" in n else n.split(".")[0]
            if len(base) < 2:
                return None
            octave_str = "".join(c for c in base if c.isdigit() or (c == "-" and base.index(c) > 0))
            pitch = "".join(c for c in base if not c.isdigit())
            if pitch in NOTE_ORDER and octave_str:
                return NOTE_ORDER.index(pitch) + (int(octave_str) + 1) * 12
        except Exception:
            return None
        return None

    sample = notes[:max_notes]
    midi_vals, positions = [], []
    for i, note in enumerate(sample):
        m = note_to_midi_approx(note)
        if m is not None:
            midi_vals.append(m)
            positions.append(i)

    if not midi_vals:
        logger.warning("Piano roll: no parseable notes found.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(positions, midi_vals, c=midi_vals, cmap="plasma", s=15, alpha=0.8)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title("Piano Roll Visualisation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved piano roll → {save_path}")


# ─────────────────────────── Misc Helpers ───────────────────────────

def format_duration(seconds: float) -> str:
    """Convert seconds → human-readable 'Xm Ys' string."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def get_device_info() -> Dict[str, Any]:
    """Return GPU / CPU availability info for TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return {
            "tensorflow_version": tf.__version__,
            "gpu_available": len(gpus) > 0,
            "gpu_devices": [g.name for g in gpus],
            "num_gpus": len(gpus),
        }
    except ImportError:
        return {"tensorflow_version": "not installed", "gpu_available": False}
