"""
Training Pipeline — AI Music Generator
Usage: python train.py --data_dir data --arch stacked_lstm --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from utils import (
    ensure_dir, save_json, load_json,
    plot_training_history, get_device_info, logger,
)
from preprocess import MusicPreprocessor
from model import MusicModelFactory, build_callbacks


# ──────────────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────────────

class MusicTrainer:
    """
    Orchestrates:
        preprocessing → model building → training → history persistence
    """

    HISTORY_FILE = "models/training_history.json"

    def __init__(
        self,
        architecture:    str   = "stacked_lstm",
        sequence_length: int   = 64,
        epochs:          int   = 50,
        batch_size:      int   = 64,
        lstm_units:      int   = 256,
        dropout_rate:    float = 0.3,
        learning_rate:   float = 1e-3,
        validation_split: float = 0.15,
        patience:        int   = 10,
        model_dir:       str   = "models",
    ) -> None:
        self.architecture     = architecture
        self.sequence_length  = sequence_length
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lstm_units       = lstm_units
        self.dropout_rate     = dropout_rate
        self.learning_rate    = learning_rate
        self.validation_split = validation_split
        self.patience         = patience
        self.model_dir        = model_dir

        self.model       = None
        self.preprocessor: Optional[MusicPreprocessor] = None
        self.history_dict: Dict = {}

    # ------------------------------------------------------------------
    def prepare_data(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        force_reload: bool = False,
    ) -> "MusicTrainer":
        self.preprocessor = MusicPreprocessor(sequence_length=self.sequence_length)
        self.preprocessor.load_midi_directory(
            data_dir, max_files=max_files, force_reload=force_reload
        )
        self.preprocessor.get_stats().print_summary()
        return self

    # ------------------------------------------------------------------
    def build_model(self) -> "MusicTrainer":
        if self.preprocessor is None or not self.preprocessor.is_ready:
            raise RuntimeError("Call prepare_data() before build_model().")

        dev = get_device_info()
        logger.info(f"Device info: {dev}")

        self.model = MusicModelFactory.build(
            architecture    = self.architecture,
            n_vocab         = self.preprocessor.n_vocab,
            sequence_length = self.sequence_length,
            lstm_units      = self.lstm_units,
            dropout_rate    = self.dropout_rate,
            learning_rate   = self.learning_rate,
        )
        self.model.summary(print_fn=logger.info)
        return self

    # ------------------------------------------------------------------
    def train(self) -> "MusicTrainer":
        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        X, y = self.preprocessor.get_training_data()

        callbacks = build_callbacks(
            model_dir  = self.model_dir,
            patience   = self.patience,
            model_name = f"{self.architecture}_best",
        )

        logger.info(
            f"Training '{self.architecture}' — "
            f"{self.epochs} epochs, batch {self.batch_size}, "
            f"X={X.shape}"
        )
        t0 = time.time()

        history = self.model.fit(
            X, y,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            validation_split = self.validation_split,
            callbacks       = callbacks,
            verbose         = 1,
        )
        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed/60:.1f} minutes.")

        self.history_dict = history.history
        self._save_history()
        self._save_plots()
        return self

    # ------------------------------------------------------------------
    def run(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        force_reload: bool = False,
    ) -> "MusicTrainer":
        """Convenience: prepare → build → train in one call."""
        return (
            self.prepare_data(data_dir, max_files, force_reload)
                .build_model()
                .train()
        )

    # ── Private helpers ─────────────────────────────────────────────

    def _save_history(self) -> None:
        ensure_dir(self.model_dir)
        save_json(self.history_dict, self.HISTORY_FILE)

        # Also save model config for later inference
        cfg = {
            "architecture":    self.architecture,
            "sequence_length": self.sequence_length,
            "n_vocab":         self.preprocessor.n_vocab,
            "lstm_units":      self.lstm_units,
        }
        save_json(cfg, os.path.join(self.model_dir, "model_config.json"))
        logger.info("Training history & config saved.")

    def _save_plots(self) -> None:
        plot_path = os.path.join("assets", "training_history.png")
        plot_training_history(self.history_dict, plot_path)

    # ------------------------------------------------------------------
    def print_final_metrics(self) -> None:
        h = self.history_dict
        if not h:
            print("No training history available.")
            return
        print("\n" + "=" * 45)
        print("  📈  FINAL TRAINING METRICS")
        print("=" * 45)
        print(f"  Train loss     : {h['loss'][-1]:.4f}")
        if "val_loss" in h:
            print(f"  Val   loss     : {h['val_loss'][-1]:.4f}")
        acc_key = "accuracy" if "accuracy" in h else "acc"
        if acc_key in h:
            print(f"  Train accuracy : {h[acc_key][-1]*100:.2f}%")
        val_acc = h.get("val_accuracy", h.get("val_acc", []))
        if val_acc:
            print(f"  Val   accuracy : {val_acc[-1]*100:.2f}%")
        print(f"  Total epochs   : {len(h['loss'])}")
        print("=" * 45 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AI Music Generator")
    p.add_argument("--data_dir",   default="data",          help="MIDI dataset directory")
    p.add_argument("--arch",       default="stacked_lstm",
                   choices=MusicModelFactory.ARCHITECTURES,  help="Model architecture")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len",    type=int, default=64)
    p.add_argument("--lstm_units", type=int, default=256)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int, default=10)
    p.add_argument("--max_files",  type=int, default=None)
    p.add_argument("--force",      action="store_true")
    p.add_argument("--eda",        action="store_true", help="Run EDA before training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = MusicTrainer(
        architecture    = args.arch,
        sequence_length = args.seq_len,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lstm_units      = args.lstm_units,
        dropout_rate    = args.dropout,
        learning_rate   = args.lr,
        patience        = args.patience,
    )

    trainer.prepare_data(args.data_dir, args.max_files, args.force)

    if args.eda:
        trainer.preprocessor.run_eda()

    trainer.build_model().train()
    trainer.print_final_metrics()
