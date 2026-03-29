# Tablio

Tablio is an end-to-end pipeline for automatic guitar tablature transcription from audio. Given a raw audio file (MP3, WAV, etc.), it separates the guitar stem, runs a deep learning model to detect note onsets and fret positions on each string, and outputs both MIDI and ASCII tablature.

The core model is a CNN-RNN architecture (GuitarTabCRNN) built on top of the design from [trimplexx/music-transcription](https://github.com/trimplexx/music-transcription), retrained on a synthetic dataset of mixed GuitarSet recordings processed through [Demucs](https://github.com/adefossez/demucs) stem separation.

---

## Pipeline Overview

```
Audio file
    │
    ▼
[1] Stem Extraction       — Demucs htdemucs_6s separates the guitar stem
    │
    ▼
[2] Guitar Stem Cleaning  — Bandpass filter + spectral gate + Wiener subtraction
    │
    ▼
[3] Model Inference       — GuitarTabCRNN predicts per-frame onset + fret per string
    │
    ▼
[4] Output Generation     — ASCII tablature (.txt) + MIDI (.mid)
```

### Model Architecture

**GuitarTabCRNN** consists of two stages:

- **TabCNN** — 5-layer convolutional front-end applied to a Constant-Q Transform (CQT) spectrogram (168 bins, 22050 Hz, hop 512)
- **GRU** — 2-layer bidirectional GRU (hidden size 768, dropout 0.55) for sequential prediction
- **Output heads** — onset detector (6 strings) and fret classifier (6 strings × 21 fret classes including silence)

---

## Results

Evaluated on a held-out validation split of synthetic GuitarSet + MusDB18 mixes. The best model (`hyperparam_set_v2 / run_2`) was retrained on Demucs-extracted stems with data augmentation.

| Metric | Mean | Median |
|--------|------|--------|
| TDR Precision | 0.791 | 0.845 |
| TDR Recall | 0.736 | 0.775 |
| **TDR F1** | **0.758** | **0.813** |
| Onset Precision | 0.893 | 0.923 |
| Onset Recall | 0.682 | 0.707 |
| **Onset F1** | **0.756** | **0.796** |

**TDR (Tablature Detection Rate):** note-level F1 — a predicted note is correct if its onset is within 50 ms of the ground truth and the string and fret match exactly.

---

## Requirements

- Python 3.12
- PyTorch 2.x
- See `requirements.txt` for full dependency list

Key dependencies:

| Package | Purpose |
|---------|---------|
| `torch` / `torchaudio` | Model inference |
| `demucs` | Guitar stem separation |
| `librosa` | CQT feature extraction |
| `pretty_midi` | MIDI export |
| `mir_eval` | Onset evaluation metrics |
| `mirdata` / `jams` | GuitarSet dataset loading |

---

## Installation

```bash
git clone https://github.com/JCoji/Tablio.git
cd Tablio
python -m venv venv
```

Activate the virtual environment:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```cmd
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### Download model and datasets

```bash
python scripts/download_assets.py
```

This downloads the pretrained model weights and evaluation datasets from [Hugging Face](https://huggingface.co/JCoji) into the expected local directories. Pass `--model-only` or `--data-only` to download selectively.

---

## Usage

Supported input formats: MP3, WAV, FLAC, OGG, M4A (any format readable by torchaudio).

```bash
# Basic run (auto-selects best available device)
python src/run_pipeline.py --input input/song.mp3 --device mps

# Skip DSP cleaning (provides best results)
python src/run_pipeline.py --input input/song.mp3 --device cpu --no-cleaning

# Use a specific model run
python src/run_pipeline.py \
  --input input/song.mp3 \
  --device mps \
  --artifacts hyperparam_set_v2 \
  --run run_2_Test_Higher_Dropout_0.55_augEnabled
```

Output is written to `output/<song_name>/`:
- `predicted_tab.txt` — ASCII guitar tablature
- `predicted.mid` — MIDI file

Supported devices: `cpu`, `cuda`, `mps` (Apple Silicon).

---

## Repository Structure

```
src/                        Core pipeline code
  run_pipeline.py           Entry point and CLI
  nodes.py                  Pipeline stage implementations
  stem_extraction.py        Demucs wrapper
  config.py                 Global parameters
model/
  architecture.py           GuitarTabCRNN and TabCNN definitions
  utils.py                  Model loading utilities
training/
  note_conversion_utils.py  Frame-to-note conversion
evaluation/
  metrics.py                TDR, onset, and MPE metrics
  tablature_export.py       ASCII tab and MIDI export
scripts/
  download_assets.py        HuggingFace asset downloader
hyperparam_set_v2/          Trained model checkpoints
guitarset_demucs/           Synthetic training/evaluation datasets
```

---

## Reproducing Experiments

The `experiment*.ipynb` notebooks document the full evaluation history:

| Notebook | Model | Dataset | TDR F1 (mean) |
|----------|-------|---------|---------------|
| experiment1 | set_v1 / run_7 | Clean mixes | 0.722 |
| experiment2 | set_v1 / run_7 | Augmented mixes | 0.685 |
| experiment3 | set_v1 / run_7 | Augmented + DSP cleaning | 0.643 |
| experiment4 | set_v2 / run_2 | Augmented mixes | **0.758** |

To regenerate the synthetic datasets from scratch (requires GuitarSet via `mirdata` and MusDB18-HQ):

```bash
# Clean mixes
python guitarset_demucs/create_demucs_finetuning_dataset.py \
  --musdb musdb18hq \
  --output guitarset_demucs/guitarset_mixed_dataset_clean \
  --no-augment

# Augmented mixes
python guitarset_demucs/create_demucs_finetuning_dataset.py \
  --musdb musdb18hq \
  --output guitarset_demucs/guitarset_mixed_dataset_augmented
```

---

## Citations

**Model architecture** — original CRNN design this work builds on:

```bibtex
@mastersthesis{krawczyk2025gtt,
  title={Przekształcanie nagrań dźwiękowych na zapis nutowy
         [Converting audio recordings to musical notation]},
  author={Krawczyk, Łukasz},
  year={2025},
  school={Silesian University of Technology},
  type={Master's thesis},
  note={Faculty of Automatic Control, Electronics and Computer Science.
        Supervisor: Dr inż. Paweł Benecki.
        Code: https://github.com/trimplexx/music-transcription}
}
```

**Stem separation** — HTDemucs (6-stem model used for guitar extraction):

```bibtex
@inproceedings{rouard2023hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={ISMIR Workshop on Music Source Separation},
  year={2021}
}
```

**GuitarSet** — guitar transcription dataset:

```bibtex
@inproceedings{xi2018guitarset,
  title={GuitarSet: A Dataset for Guitar Transcription},
  author={Xi, Qingyang and Bittner, Rachel M. and Ye, Xuzhou and Pauwels, Johan and Bello, Juan Pablo},
  booktitle={Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)},
  pages={453--460},
  address={Paris, France},
  year={2018}
}
```

**MUSDB18-HQ** — music source separation dataset:

```bibtex
@misc{rafii2017musdb18,
  title={MUSDB18: A Corpus for Music Separation},
  author={Rafii, Zafar and Liutkus, Antoine and St\"{o}ter, Fabian-Robert and Mimilakis, Stylianos Ioannis and Bittner, Rachel},
  year={2017},
  doi={10.5281/zenodo.1117372},
  url={https://doi.org/10.5281/zenodo.1117372}
}
```
