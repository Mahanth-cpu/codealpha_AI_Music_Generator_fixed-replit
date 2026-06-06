AI Music Generator — Dataset Directory
  ======================================

  Place MIDI files (.mid or .midi) here before training.

  Recommended datasets:
  - MAESTRO v3 (classical piano, 1,276 files)
    https://magenta.tensorflow.org/datasets/maestro

  - Lakh MIDI Dataset (mixed genres, 176,581 files)
    https://colinraffel.com/projects/lmd/

  - Piano-midi.de (classical piano, ~300 files)
    http://www.piano-midi.de/

  - JSB Chorales (Bach, 382 files — bundled with music21)
    Run: python -c "from music21 import corpus; corpus.getComposer('bach')"

  Usage:
  ------
  1. Download any dataset above
  2. Copy .mid / .midi files into this folder
  3. Run: python train.py --data_dir data

  The preprocessor will recursively find all MIDI files in subdirectories.
  Processed data is cached as data/processed_notes.pkl for faster reloads.
  