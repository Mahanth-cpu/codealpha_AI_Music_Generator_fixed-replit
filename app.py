"""
AI Music Generator — Streamlit Web Application
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="🎵 AI Music Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import MusicPreprocessor
from generate import MusicGenerator, MIDIWriter
from utils import (
    ensure_dir, plot_training_history, plot_note_distribution,
    plot_song_lengths, plot_piano_roll, load_json, save_json, logger,
)

# ── Optional TF ──
try:
    from model import MusicModelFactory, build_callbacks
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  /* ── Gradient hero header ── */
  .hero-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(96,165,250,0.15);
  }
  .hero-header h1 {
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero-header p { color: #94a3b8; font-size: 1.1rem; margin-top: .5rem; }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform .2s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-card .val  { font-size: 2rem; font-weight: 700; color: #a78bfa; }
  .metric-card .lbl  { font-size: .85rem; color: #94a3b8; margin-top: 4px; }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #6366f1; padding-left: .75rem;
    margin: 1.5rem 0 1rem;
  }

  /* ── Success / info banners ── */
  .banner-success {
    background: linear-gradient(90deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius: 10px;
    padding: 1rem 1.25rem; color: #d1fae5; font-weight: 500;
  }
  .banner-info {
    background: linear-gradient(90deg, #1e3a5f, #1e40af);
    border: 1px solid #3b82f6; border-radius: 10px;
    padding: 1rem 1.25rem; color: #dbeafe; font-weight: 500;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1040 100%);
  }
  [data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: .65rem 1.5rem !important;
    transition: opacity .2s, transform .1s !important;
  }
  .stButton > button:hover {
    opacity: .9 !important; transform: translateY(-1px) !important;
  }

  /* ── Code blocks ── */
  code { font-family: 'JetBrains Mono', monospace; font-size: .85rem; }

  /* ── Tab labels ── */
  .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: .95rem; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #1e1b4b; }
  ::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Session-state helpers
# ──────────────────────────────────────────────────────────────────────────────

def ss_get(key: str, default=None):
    return st.session_state.get(key, default)


def ss_set(key: str, value) -> None:
    st.session_state[key] = value


def init_session() -> None:
    defaults = {
        "preprocessor":    None,
        "trainer":         None,
        "generator":       None,
        "training_done":   False,
        "generated_notes": None,
        "generated_midi":  None,
        "eda_paths":       {},
        "history":         {},
        "dataset_stats":   {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0'>"
            "<span style='font-size:2.5rem'>🎵</span>"
            "<h2 style='color:#a78bfa;margin:0'>AI Music Gen</h2>"
            "<p style='color:#64748b;font-size:.85rem'>LSTM-powered composition</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### ⚙️ Model Config")
        arch = st.selectbox(
            "Architecture",
            ["stacked_lstm", "basic_lstm", "bidirectional_lstm", "transformer"],
            help="Choose the neural network architecture",
        )
        seq_len    = st.slider("Sequence Length",      32, 128, 64, 16)
        lstm_units = st.slider("LSTM Units",           64, 512, 256, 64)
        dropout    = st.slider("Dropout Rate",         0.1, 0.5, 0.3, 0.05)
        lr         = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)

        st.divider()
        st.markdown("### 🏋️ Training")
        epochs     = st.slider("Epochs",     5, 200, 50, 5)
        batch_size = st.select_slider("Batch Size", [16, 32, 64, 128, 256], value=64)
        patience   = st.slider("Early Stop Patience", 3, 30, 10)
        val_split  = st.slider("Validation Split", 0.05, 0.30, 0.15, 0.05)

        st.divider()
        st.markdown("### 🎶 Generation")
        num_notes   = st.slider("Notes to Generate", 50, 500, 200, 25)
        temperature = st.slider(
            "Creativity (Temperature)",
            0.3, 2.0, 1.0, 0.1,
            help="Lower = conservative, Higher = experimental",
        )
        output_name = st.text_input("Output Filename", "ai_composition")
        tempo_bpm   = st.slider("Tempo (BPM)", 60, 200, 120, 5)

        st.divider()
        if TF_AVAILABLE:
            try:
                gpus = tf.config.list_physical_devices("GPU")
                device_label = f"🟢 GPU ({len(gpus)}x)" if gpus else "🔵 CPU"
            except Exception:
                device_label = "🔵 CPU"
        else:
            device_label = "❌ TF not installed"
        st.caption(f"Device: **{device_label}**")
        st.caption(f"TF: {'✅' if TF_AVAILABLE else '❌'}")

    return {
        "arch": arch, "seq_len": seq_len, "lstm_units": lstm_units,
        "dropout": dropout, "lr": lr, "epochs": epochs,
        "batch_size": batch_size, "patience": patience,
        "val_split": val_split, "num_notes": num_notes,
        "temperature": temperature, "output_name": output_name,
        "tempo_bpm": tempo_bpm,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Tab 1 — Data Upload & EDA
# ──────────────────────────────────────────────────────────────────────────────

def tab_data(cfg: Dict) -> None:
    st.markdown('<div class="section-header">📂 Upload MIDI Dataset</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload one or more MIDI files (.mid / .midi)",
            type=["mid", "midi"],
            accept_multiple_files=True,
            help="Upload a collection of MIDI files to train the model",
        )
    with col2:
        use_demo = st.checkbox("🎹 Use demo dataset", value=True,
                               help="Generate synthetic notes for demonstration")
        min_freq = st.number_input("Min token frequency", 1, 20, 1)

    if st.button("🔍 Analyse Dataset", use_container_width=True):
        with st.spinner("Parsing MIDI files…"):
            pre = MusicPreprocessor(
                sequence_length=cfg["seq_len"],
                min_note_frequency=int(min_freq),
            )

            if uploaded:
                tmp_dir = tempfile.mkdtemp()
                for uf in uploaded:
                    p = os.path.join(tmp_dir, uf.name)
                    with open(p, "wb") as fh:
                        fh.write(uf.read())
                pre.load_midi_directory(tmp_dir, force_reload=True)

            elif use_demo:
                notes = _generate_demo_notes(2000)
                pre.load_notes_list(notes)
            else:
                st.warning("Upload MIDI files or enable the demo dataset.")
                return

        ss_set("preprocessor", pre)
        stats = pre.get_stats().summary()
        ss_set("dataset_stats", stats)

        # EDA plots
        with st.spinner("Generating visualisations…"):
            ensure_dir("assets")
            eda_paths = pre.run_eda()
            ss_set("eda_paths", eda_paths)

        st.markdown(
            '<div class="banner-success">✅ Dataset loaded successfully!</div>',
            unsafe_allow_html=True,
        )

    # ── Show stats & plots ──
    stats = ss_get("dataset_stats", {})
    if stats:
        st.markdown('<div class="section-header">📊 Dataset Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        _metric_card(c1, stats.get("num_songs", 0), "Songs")
        _metric_card(c2, f"{stats.get('total_notes', 0):,}", "Total Notes")
        _metric_card(c3, stats.get("unique_tokens", 0), "Unique Tokens")
        _metric_card(c4, f"{stats.get('avg_song_length', 0):.0f}", "Avg Notes/Song")

        if stats.get("top_10_notes"):
            st.markdown('<div class="section-header">🎼 Top Notes & Chords</div>', unsafe_allow_html=True)
            tokens, counts = zip(*stats["top_10_notes"])
            df = pd.DataFrame({"Token": tokens, "Frequency": counts})
            st.bar_chart(df.set_index("Token"))

    eda = ss_get("eda_paths", {})
    if eda:
        st.markdown('<div class="section-header">📈 Visualisations</div>', unsafe_allow_html=True)
        tab_names = list(eda.keys())
        if tab_names:
            tabs = st.tabs([n.replace("_", " ").title() for n in tab_names])
            for tab, (name, path) in zip(tabs, eda.items()):
                with tab:
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.info(f"Plot not yet generated: {name}")


# ──────────────────────────────────────────────────────────────────────────────
#  Tab 2 — Model Training
# ──────────────────────────────────────────────────────────────────────────────

def tab_train(cfg: Dict) -> None:
    pre: Optional[MusicPreprocessor] = ss_get("preprocessor")

    if pre is None:
        st.info("👆 Load a dataset in the **Data** tab first.")
        return

    st.markdown('<div class="section-header">🏋️ Training Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Setting | Value |
        |---------|-------|
        | Architecture | `{cfg["arch"]}` |
        | Sequence Length | `{cfg["seq_len"]}` |
        | LSTM Units | `{cfg["lstm_units"]}` |
        | Dropout | `{cfg["dropout"]}` |
        | Learning Rate | `{cfg["lr"]}` |
        """)
    with col2:
        st.markdown(f"""
        | Setting | Value |
        |---------|-------|
        | Epochs | `{cfg["epochs"]}` |
        | Batch Size | `{cfg["batch_size"]}` |
        | Patience | `{cfg["patience"]}` |
        | Validation Split | `{cfg["val_split"]}` |
        | Vocab Size | `{pre.n_vocab}` |
        """)

    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not installed. Install it to train models.")
        return

    if st.button("🚀 Start Training", use_container_width=True):
        _run_training(pre, cfg)

    # ── Show history if available ──
    history = ss_get("history", {})
    if history:
        st.markdown('<div class="section-header">📈 Training History</div>', unsafe_allow_html=True)

        loss_df = pd.DataFrame({
            "Train Loss": history.get("loss", []),
            "Val Loss":   history.get("val_loss", history.get("loss", [])),
        })
        st.line_chart(loss_df)

        acc_key     = "accuracy" if "accuracy" in history else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
        if acc_key in history:
            acc_df = pd.DataFrame({
                "Train Accuracy": history.get(acc_key, []),
                "Val Accuracy":   history.get(val_acc_key, history.get(acc_key, [])),
            })
            st.line_chart(acc_df)

        c1, c2, c3 = st.columns(3)
        _metric_card(c1, f"{history['loss'][-1]:.4f}", "Final Train Loss")
        val_loss = history.get("val_loss", [history["loss"][-1]])
        _metric_card(c2, f"{val_loss[-1]:.4f}", "Final Val Loss")
        _metric_card(c3, len(history["loss"]), "Epochs Completed")


def _run_training(pre: MusicPreprocessor, cfg: Dict) -> None:
    """Run training and update session state with results."""
    progress = st.progress(0, text="Preparing training data…")
    log_box  = st.empty()

    try:
        from model import MusicModelFactory, build_callbacks
        import tensorflow as tf

        X, y = pre.get_training_data()
        progress.progress(15, text="Building model…")

        model = MusicModelFactory.build(
            architecture    = cfg["arch"],
            n_vocab         = pre.n_vocab,
            sequence_length = cfg["seq_len"],
            lstm_units      = cfg["lstm_units"],
            dropout_rate    = cfg["dropout"],
            learning_rate   = cfg["lr"],
        )

        callbacks = build_callbacks(
            model_dir  = "models",
            patience   = cfg["patience"],
            model_name = f"{cfg['arch']}_best",
        )

        progress.progress(25, text=f"Training '{cfg['arch']}'… (this may take a while)")

        history = model.fit(
            X, y,
            epochs           = cfg["epochs"],
            batch_size       = cfg["batch_size"],
            validation_split = cfg["val_split"],
            callbacks        = callbacks,
            verbose          = 0,
        )

        progress.progress(90, text="Saving model & history…")

        ensure_dir("models")
        model.save(f"models/{cfg['arch']}_best.keras")

        # Save config for generator
        save_json({
            "architecture":    cfg["arch"],
            "sequence_length": cfg["seq_len"],
            "n_vocab":         pre.n_vocab,
        }, "models/model_config.json")

        ss_set("history", history.history)
        ss_set("training_done", True)

        progress.progress(100, text="✅ Training complete!")
        st.markdown(
            '<div class="banner-success">🎉 Model trained and saved successfully!</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        progress.empty()
        st.error(f"Training failed: {exc}")
        logger.exception("Training error")


# ──────────────────────────────────────────────────────────────────────────────
#  Tab 3 — Generate Music
# ──────────────────────────────────────────────────────────────────────────────

def tab_generate(cfg: Dict) -> None:
    training_done = ss_get("training_done", False)
    pre: Optional[MusicPreprocessor] = ss_get("preprocessor")

    if not training_done:
        st.info("👆 Train a model in the **Train** tab first.")
        return

    st.markdown('<div class="section-header">🎶 Generate Music</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Notes to Generate", cfg["num_notes"])
    with col2:
        temp_label = (
            "🥶 Conservative" if cfg["temperature"] < 0.7
            else "😎 Balanced" if cfg["temperature"] < 1.3
            else "🔥 Experimental"
        )
        st.metric("Creativity", f"{cfg['temperature']} — {temp_label}")
    with col3:
        st.metric("Tempo", f"{cfg['tempo_bpm']} BPM")

    if st.button("🎵 Generate Music", use_container_width=True):
        _run_generation(pre, cfg)

    # ── Playback / download ──
    midi_bytes = ss_get("generated_midi")
    notes      = ss_get("generated_notes")

    if midi_bytes is not None:
        st.markdown('<div class="section-header">🎧 Your Composition</div>', unsafe_allow_html=True)

        st.download_button(
            label     = "⬇️ Download MIDI",
            data      = midi_bytes,
            file_name = f"{cfg['output_name']}.mid",
            mime      = "audio/midi",
            use_container_width=True,
        )

        if notes:
            stats = _note_stats(notes)
            c1, c2, c3, c4 = st.columns(4)
            _metric_card(c1, stats["total"],    "Total Notes")
            _metric_card(c2, stats["unique"],   "Unique Tokens")
            _metric_card(c3, f"{stats['div']}%", "Diversity")
            _metric_card(c4, stats["chords"],   "Chords")

            st.markdown("**Sample generated tokens:**")
            st.code(" → ".join(notes[:30]), language=None)

            # Frequency chart
            from collections import Counter
            top = Counter(notes).most_common(15)
            toks, cnts = zip(*top)
            df = pd.DataFrame({"Token": toks, "Count": cnts})
            st.bar_chart(df.set_index("Token"))


def _run_generation(pre: Optional[MusicPreprocessor], cfg: Dict) -> None:
    with st.spinner("🎹 Composing…"):
        try:
            if not TF_AVAILABLE:
                notes = _demo_generate(cfg["num_notes"])
            else:
                gen = MusicGenerator(
                    output_dir = "generated_music",
                    tempo_bpm  = cfg["tempo_bpm"],
                )
                gen.load(architecture=cfg["arch"])
                notes = gen.generate(
                    num_notes   = cfg["num_notes"],
                    temperature = cfg["temperature"],
                )

            ss_set("generated_notes", notes)

            # Write MIDI to buffer
            writer = MIDIWriter(tempo_bpm=cfg["tempo_bpm"], output_dir="generated_music")
            out_path = writer.write(notes, cfg["output_name"])

            with open(out_path, "rb") as fh:
                ss_set("generated_midi", fh.read())

            st.markdown(
                '<div class="banner-success">✅ Music generated!</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            logger.exception("Generation error")


# ──────────────────────────────────────────────────────────────────────────────
#  Tab 4 — Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def tab_evaluate() -> None:
    history = ss_get("history", {})

    if not history:
        st.info("👆 Train a model to see evaluation metrics.")
        return

    st.markdown('<div class="section-header">📉 Loss Curves</div>', unsafe_allow_html=True)
    loss_df = pd.DataFrame({
        "Train Loss": history.get("loss", []),
        "Val Loss":   history.get("val_loss", history.get("loss", [])),
    })
    st.line_chart(loss_df)

    acc_key     = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    if acc_key in history:
        st.markdown('<div class="section-header">📈 Accuracy Curves</div>', unsafe_allow_html=True)
        acc_df = pd.DataFrame({
            "Train Accuracy": history.get(acc_key, []),
            "Val Accuracy":   history.get(val_acc_key, history.get(acc_key, [])),
        })
        st.line_chart(acc_df)

    # Summary table
    st.markdown('<div class="section-header">📋 Training Summary</div>', unsafe_allow_html=True)
    summary = {
        "Total Epochs":         len(history.get("loss", [])),
        "Best Train Loss":      f"{min(history.get('loss', [0])):.4f}",
        "Best Val Loss":        f"{min(history.get('val_loss', history.get('loss', [0]))):.4f}",
        "Final Train Loss":     f"{history['loss'][-1]:.4f}" if history.get("loss") else "N/A",
    }
    if acc_key in history:
        summary["Best Train Accuracy"] = f"{max(history.get(acc_key, [0]))*100:.2f}%"
        val_acc = history.get(val_acc_key, history.get(acc_key, [0]))
        summary["Best Val Accuracy"]   = f"{max(val_acc)*100:.2f}%"

    df = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Tab 5 — About
# ──────────────────────────────────────────────────────────────────────────────

def tab_about() -> None:
    st.markdown(
        """
        <div class="section-header">🧠 About This Project</div>

        **AI Music Generator** uses deep learning (LSTM / Transformer) to learn
        musical patterns from MIDI datasets and compose original music.

        ### 🏗️ Architecture
        | Model | Description |
        |-------|-------------|
        | **Basic LSTM** | Single-layer LSTM with batch norm + dropout |
        | **Bidirectional LSTM** | Reads sequences forwards *and* backwards |
        | **Stacked LSTM** | 3-layer deep LSTM (default, recommended) |
        | **Transformer** | Multi-head attention for long-range patterns |

        ### 🔄 Pipeline
        ```
        MIDI Files → music21 Parser → Tokenisation → Sequence Windows
             → LSTM Training → Temperature Sampling → MIDI Output
        ```

        ### 📂 Project Structure
        ```
        AI_Music_Generator/
        ├── app.py            ← Streamlit web app (this file)
        ├── train.py          ← Training pipeline CLI
        ├── generate.py       ← Generation engine
        ├── preprocess.py     ← MIDI parsing & preprocessing
        ├── model.py          ← Neural network architectures
        ├── utils.py          ← Shared utilities & plotting
        ├── data/             ← MIDI files & processed cache
        ├── models/           ← Saved checkpoints
        ├── generated_music/  ← Output MIDI files
        └── assets/           ← EDA plots
        ```

        ### 🎛️ Temperature Guide
        | Value | Effect |
        |-------|--------|
        | 0.3 – 0.7 | Conservative, close to training data |
        | 0.8 – 1.2 | Balanced creativity (recommended) |
        | 1.3 – 2.0 | Experimental, more surprising output |

        ### 📦 Tech Stack
        `TensorFlow/Keras` · `music21` · `NumPy` · `Pandas` ·
        `Matplotlib` · `PrettyMIDI` · `Streamlit` · `MIDIUtil`

        ---
        Built with ❤️ as an internship-ready portfolio project.
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _metric_card(col, value, label: str) -> None:
    col.markdown(
        f'<div class="metric-card"><div class="val">{value}</div>'
        f'<div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _note_stats(notes: List[str]) -> Dict:
    from collections import Counter
    c = Counter(notes)
    chords = sum(1 for n in notes if "." in n)
    return {
        "total":  len(notes),
        "unique": len(c),
        "div":    round(len(c) / max(len(notes), 1) * 100, 1),
        "chords": chords,
    }


def _generate_demo_notes(n: int = 2000) -> List[str]:
    """Generate plausible-sounding demo tokens without music21."""
    rng   = np.random.default_rng(42)
    scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
             "C5", "D5", "E5", "F5", "G5", "A5",
             "C4.E4.G4", "F4.A4.C5", "G4.B4.D5",
             "A4.C5.E5", "D4.F4.A4"]
    probs = np.array([4, 3, 3, 2, 4, 3, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2], dtype=float)
    probs /= probs.sum()
    return rng.choice(scale, size=n, p=probs).tolist()


def _demo_generate(num_notes: int) -> List[str]:
    """Generate demo output when TF is unavailable."""
    rng   = np.random.default_rng(int(time.time()) % 9999)
    scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
             "C5", "D5", "E5", "G4.B4.D5", "C4.E4.G4"]
    return rng.choice(scale, size=num_notes).tolist()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session()
    inject_css()

    # Hero header
    st.markdown(
        """
        <div class="hero-header">
          <h1>🎵 AI Music Generator</h1>
          <p>Deep Learning · LSTM · Original Compositions from MIDI Datasets</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = render_sidebar()

    tabs = st.tabs(["📂 Data", "🏋️ Train", "🎶 Generate", "📊 Evaluate", "ℹ️ About"])

    with tabs[0]: tab_data(cfg)
    with tabs[1]: tab_train(cfg)
    with tabs[2]: tab_generate(cfg)
    with tabs[3]: tab_evaluate()
    with tabs[4]: tab_about()


if __name__ == "__main__":
    main()
