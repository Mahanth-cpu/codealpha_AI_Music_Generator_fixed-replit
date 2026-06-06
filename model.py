"""
Model Architectures for AI Music Generator
Implements: Basic LSTM, Bidirectional LSTM, Stacked LSTM, Transformer
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, Tuple

from utils import logger

# ────── TensorFlow import ──────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — model stubs only.")


# ──────────────────────────────────────────────────────────────────────────────
#  Model Factory
# ──────────────────────────────────────────────────────────────────────────────

class MusicModelFactory:
    """Build one of several LSTM (or Transformer) architectures."""

    ARCHITECTURES = ["basic_lstm", "bidirectional_lstm", "stacked_lstm", "transformer"]

    # ------------------------------------------------------------------
    @staticmethod
    def build(
        architecture: str,
        n_vocab: int,
        sequence_length: int,
        **kwargs,
    ) -> "keras.Model":
        """
        Parameters
        ----------
        architecture    : one of ARCHITECTURES
        n_vocab         : vocabulary / output size
        sequence_length : LSTM time-steps
        **kwargs        : passed to the specific builder
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to build models.")

        builders = {
            "basic_lstm":         MusicModelFactory._basic_lstm,
            "bidirectional_lstm": MusicModelFactory._bidirectional_lstm,
            "stacked_lstm":       MusicModelFactory._stacked_lstm,
            "transformer":        MusicModelFactory._transformer,
        }

        if architecture not in builders:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from {MusicModelFactory.ARCHITECTURES}"
            )

        model = builders[architecture](n_vocab, sequence_length, **kwargs)
        logger.info(
            f"Built '{architecture}' — "
            f"params: {model.count_params():,}"
        )
        return model

    # ── A. Basic LSTM ────────────────────────────────────────────────

    @staticmethod
    def _basic_lstm(
        n_vocab: int,
        sequence_length: int,
        lstm_units: int = 256,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        **_,
    ) -> "keras.Model":
        model = keras.Sequential(
            [
                keras.Input(shape=(sequence_length, 1), name="input"),
                layers.LSTM(lstm_units, return_sequences=False, name="lstm1"),
                layers.BatchNormalization(name="bn1"),
                layers.Dropout(dropout_rate, name="drop1"),
                layers.Dense(lstm_units // 2, activation="relu", name="dense1"),
                layers.BatchNormalization(name="bn2"),
                layers.Dropout(dropout_rate / 2, name="drop2"),
                layers.Dense(n_vocab, activation="softmax", name="output"),
            ],
            name="BasicLSTM",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── B. Bidirectional LSTM ────────────────────────────────────────

    @staticmethod
    def _bidirectional_lstm(
        n_vocab: int,
        sequence_length: int,
        lstm_units: int = 256,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        **_,
    ) -> "keras.Model":
        inp = keras.Input(shape=(sequence_length, 1), name="input")
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True), name="bilstm1"
        )(inp)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(dropout_rate, name="drop1")(x)
        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2), name="bilstm2"
        )(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(dropout_rate / 2, name="drop2")(x)
        x = layers.Dense(n_vocab // 2, activation="relu", name="dense1")(x)
        out = layers.Dense(n_vocab, activation="softmax", name="output")(x)

        model = keras.Model(inp, out, name="BidirectionalLSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── C. Stacked LSTM ──────────────────────────────────────────────

    @staticmethod
    def _stacked_lstm(
        n_vocab: int,
        sequence_length: int,
        lstm_units: int = 512,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
        learning_rate: float = 5e-4,
        **_,
    ) -> "keras.Model":
        inp = keras.Input(shape=(sequence_length, 1), name="input")
        x = inp

        for i in range(num_layers):
            return_seq = i < (num_layers - 1)
            units = lstm_units // (2 ** i)
            x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm{i+1}")(x)
            x = layers.BatchNormalization(name=f"bn{i+1}")(x)
            x = layers.Dropout(dropout_rate, name=f"drop{i+1}")(x)

        x = layers.Dense(256, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2, name="drop_final")(x)
        out = layers.Dense(n_vocab, activation="softmax", name="output")(x)

        model = keras.Model(inp, out, name="StackedLSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── D. Transformer (bonus) ───────────────────────────────────────

    @staticmethod
    def _transformer(
        n_vocab: int,
        sequence_length: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dff: int = 256,
        dropout_rate: float = 0.1,
        learning_rate: float = 5e-4,
        **_,
    ) -> "keras.Model":
        inp = keras.Input(shape=(sequence_length, 1), name="input")

        # Projection to d_model
        x = layers.Dense(d_model, name="proj")(inp)

        # Positional encoding (simple learnable)
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        pos_emb = layers.Embedding(sequence_length, d_model, name="pos_emb")(positions)
        x = x + pos_emb

        # Transformer blocks
        for i in range(num_layers):
            # Multi-head attention
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads,
                dropout=dropout_rate, name=f"mha_{i}"
            )(x, x)
            x = layers.LayerNormalization(name=f"ln1_{i}")(x + attn_out)

            # FFN
            ffn = layers.Dense(dff, activation="relu", name=f"ffn1_{i}")(x)
            ffn = layers.Dense(d_model, name=f"ffn2_{i}")(ffn)
            ffn = layers.Dropout(dropout_rate, name=f"drop_{i}")(ffn)
            x = layers.LayerNormalization(name=f"ln2_{i}")(x + ffn)

        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="dense1")(x)
        x = layers.Dropout(dropout_rate, name="drop_final")(x)
        out = layers.Dense(n_vocab, activation="softmax", name="output")(x)

        model = keras.Model(inp, out, name="TransformerMusicGen")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ──────────────────────────────────────────────────────────────────────────────
#  Callbacks Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_callbacks(
    model_dir: str = "models",
    patience: int = 10,
    monitor: str = "val_loss",
    use_tensorboard: bool = False,
    model_name: str = "best_model",
) -> list:
    """Return a list of Keras callbacks for training."""
    os.makedirs(model_dir, exist_ok=True)

    ckpt_path = os.path.join(model_dir, f"{model_name}.keras")

    cbs = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if use_tensorboard:
        cbs.append(TensorBoard(log_dir="logs/tensorboard", histogram_freq=1))

    return cbs


# ──────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if TF_AVAILABLE:
        for arch in MusicModelFactory.ARCHITECTURES:
            m = MusicModelFactory.build(arch, n_vocab=100, sequence_length=64)
            m.summary()
            print()
    else:
        print("TensorFlow not available.")
