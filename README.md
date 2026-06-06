# 🎵 AI Music Generator

> Deep Learning–powered music composition using LSTM & Transformer networks trained on MIDI datasets.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

AI Music Generator learns musical patterns from MIDI datasets and composes
original music using sequence-to-sequence deep learning. The system:

1. **Parses** MIDI files into note/chord token sequences (via `music21`)
2. **Trains** an LSTM (or Transformer) on sliding-window sequences
3. **Generates** new music with temperature-controlled sampling
4. **Exports** compositions as `.mid` (MIDI) files

---

## ✨ Features

| Category | Feature |
|---|---|
| **Data** | Bulk MIDI ingestion, corrupted-file handling, vocab filtering |
| **EDA** | Note/chord frequency charts, song-length distribution, piano-roll |
| **Models** | Basic LSTM · Bidirectional LSTM · Stacked LSTM · Transformer |
| **Training** | Early stopping, LR scheduling, model checkpointing, GPU support |
| **Generation** | Temperature sampling, seed selection, multi-variation export |
| **Output** | `.mid` MIDI export, optional WAV via FluidSynth |
| **UI** | Streamlit web dashboard — upload, train, generate, download |

---

## 🏗️ Architecture

```
MIDI Files
    │
    ▼
┌─────────────────┐
│  music21 Parser │  Extract notes & chords → string tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenisation   │  Build note→int vocabulary
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Sequence Windows    │  Sliding windows of length 64
│  X: (N, 64, 1)       │
│  y: (N, vocab_size)  │
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            Neural Network               │
│                                         │
│  A) Basic LSTM                          │
│     LSTM(256) → BN → Dropout → Dense   │
│                                         │
│  B) Bidirectional LSTM                  │
│     BiLSTM(256) → BiLSTM(128) → Dense  │
│                                         │
│  C) Stacked LSTM  ← recommended        │
│     LSTM(512) → LSTM(256) → LSTM(128)  │
│     → Dense(256) → Softmax             │
│                                         │
│  D) Transformer (bonus)                 │
│     MultiHeadAttention × 2 → GAP       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Temperature     │  Sample next token
│ Sampling        │  (0.5 = safe, 1.5 = wild)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MIDI Writer    │  music21 / midiutil / pretty_midi
└────────┬────────┘
         │
         ▼
    output.mid
```

---

## 📂 Project Structure

```
AI_Music_Generator/
│
├── app.py               ← Streamlit web dashboard
├── train.py             ← Training pipeline (CLI)
├── generate.py          ← Music generation engine (CLI)
├── preprocess.py        ← MIDI parsing & preprocessing
├── model.py             ← Neural network architectures
├── sample_generator.py  ← Demo MIDI without training
├── utils.py             ← Shared utilities & plotting
│
├── data/                ← MIDI input files + processed cache
│   ├── processed_notes.pkl
│   ├── vocabulary.json
│   └── dataset_stats.json
│
├── models/              ← Saved Keras checkpoints
│   ├── stacked_lstm_best.keras
│   └── model_config.json
│
├── generated_music/     ← Output MIDI compositions
│   └── sample_output.mid
│
├── assets/              ← EDA plots (auto-saved PNGs)
├── logs/                ← Training logs
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/ai-music-generator.git
cd ai-music-generator
```

### 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support** — if you have a CUDA-compatible GPU, install
> `tensorflow-gpu` instead of `tensorflow`.

### 4 — (Optional) FluidSynth for WAV export

```bash
# macOS
brew install fluidsynth

# Ubuntu / Debian
sudo apt-get install fluidsynth

# Windows — download from https://www.fluidsynth.org/
```

---

## 🎹 Dataset

The model can be trained on any collection of MIDI files. Recommended sources:

| Dataset | Genre | Files | Link |
|---------|-------|-------|------|
| MAESTRO v3 | Classical piano | 1,276 | [magenta.tensorflow.org](https://magenta.tensorflow.org/datasets/maestro) |
| Lakh MIDI | Mixed | 176,581 | [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/) |
| Piano-midi.de | Classical | ~300 | [piano-midi.de](http://www.piano-midi.de/) |
| MusicNet | Classical | 330 | [homes.cs.washington.edu](https://homes.cs.washington.edu/~thickstn/musicnet.html) |
| JSB Chorales | Bach | 382 | packaged with music21 |

Place downloaded `.mid` / `.midi` files inside the `data/` directory.

---

## 🏋️ Training

### Quick start (CLI)

```bash
# Preprocess + train with defaults (Stacked LSTM, 50 epochs)
python train.py --data_dir data

# Choose architecture
python train.py --data_dir data --arch bidirectional_lstm --epochs 100

# Run EDA visualisations before training
python train.py --data_dir data --eda

# Full example
python train.py \
  --data_dir  data       \
  --arch      stacked_lstm \
  --epochs    100        \
  --batch_size 64        \
  --seq_len   64         \
  --lstm_units 512       \
  --dropout   0.3        \
  --lr        0.001      \
  --patience  15
```

### Available architectures

| Flag | Model |
|------|-------|
| `basic_lstm` | Single LSTM layer |
| `bidirectional_lstm` | Bidirectional LSTM |
| `stacked_lstm` | 3-layer stacked LSTM (**default**) |
| `transformer` | Multi-head attention |

---

## 🎵 Generating Music

### CLI

```bash
# Generate 200 notes with default creativity
python generate.py --num_notes 200 --output my_song

# Adjust temperature (creativity)
python generate.py --num_notes 300 --temperature 1.2 --output expressive_piece

# Generate 3 variations at different temperatures
python generate.py --variations --output theme

# Demo MIDI (no training required)
python sample_generator.py --key C --mode major --bars 16 --tempo 120
```

### Streamlit web app

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and use the interactive dashboard:

1. **Data tab** — upload MIDI files or use the built-in demo dataset
2. **Train tab** — configure and train the model
3. **Generate tab** — compose music and download the `.mid` file
4. **Evaluate tab** — view loss/accuracy curves

---

## 🎛️ Temperature Guide

| Temperature | Musical Effect |
|-------------|---------------|
| `0.3 – 0.6` | Very predictable; closely mimics training data |
| `0.7 – 1.0` | Natural variation; recommended for most genres |
| `1.1 – 1.4` | More creative; occasional unexpected leaps |
| `1.5 – 2.0` | Experimental / avant-garde output |

---

## 📊 Results

Typical results on ~500 MIDI files (classical piano), 50 epochs:

| Metric | Value |
|--------|-------|
| Train Loss | ~0.35 |
| Validation Loss | ~0.42 |
| Train Accuracy | ~88% |
| Validation Accuracy | ~83% |
| Training Time (GPU) | ~25 min |
| Training Time (CPU) | ~3 hrs |

Generated compositions at `temperature=1.0` demonstrate:
- Recognisable melodic phrases
- Recurring motifs and harmonic patterns
- Stylistic similarity to training corpus

---

## 🔮 Future Improvements

- [ ] **GAN-based generation** — adversarial training for higher quality
- [ ] **Genre conditioning** — separate models per genre
- [ ] **Polyphony support** — multi-instrument MIDI output
- [ ] **Real-time generation** — streaming note-by-note output
- [ ] **Music style transfer** — apply one artist's style to another's melody
- [ ] **Chord-aware tokenisation** — encode harmonic context explicitly
- [ ] **ONNX export** — lightweight deployment without TensorFlow
- [ ] **HuggingFace Spaces deployment** — public demo

---

## 🛠️ Troubleshooting

**`music21` parse errors** — corrupt files are silently skipped; check `logs/` for details.

**Out of memory during training** — reduce `--batch_size` or `--lstm_units`.

**No MIDI library found** — install at least one: `pip install midiutil pretty_midi`.

**Streamlit port conflict** — `streamlit run app.py --server.port 8502`.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [music21](https://web.mit.edu/music21/) — MIT toolkit for computer-aided musicology
- [Magenta](https://magenta.tensorflow.org/) — inspiration for architecture choices
- [Keras](https://keras.io/) — high-level deep learning API
- [Streamlit](https://streamlit.io/) — rapid ML web-app framework
