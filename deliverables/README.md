# DeepCue

**Real-time multimodal emotion recognition for Hebrew-speaking job interview candidates.**

DeepCue analyzes three parallel signal streams during a live mock interview — facial micro-expressions, paralinguistic audio, and speech semantics — and fuses them into a unified 8-class emotion prediction (`neutral`, `confident`, `anxious`, `happy`, `sad`, `angry`, `surprised`, `uncertain`), streamed to the browser in real time. At session end, the system generates a structured PDF report with full Hebrew RTL support.

> Solo final project by **Rachel Brodsky** — Deep Learning course, Computer Science degree.

---

## Architecture Overview

DeepCue follows a **Mixture-of-Experts** design: three independently trained deep models ("experts"), one per modality, whose outputs are fused by a Cross-Modal Transformer.

| Expert | Model | Task | Output |
|---|---|---|---|
| **Video** | EfficientNet-B0 + LSTM | 8-class emotion classification from facial frame sequences | 8 logits |
| **Audio** | wav2vec 2.0 (fine-tuned) | 8-class emotion classification from 3-second speech chunks | 8 logits |
| **Text** | XLM-RoBERTa (fine-tuned) | Hebrew sentiment regression from transcribed speech | 1 scalar |
| **Fusion** | Cross-Modal Transformer + MLP head | Unify the 17-dim concatenated feature vector into one emotion | 8-class prediction |

```
Video frames ──► EfficientNet-B0 + LSTM ──► 8 logits ─┐
Audio chunks ──► wav2vec 2.0            ──► 8 logits ─┼──► Cross-Modal Transformer ──► emotion_result
Transcript   ──► XLM-RoBERTa            ──► 1 score  ─┘        (17-dim input)
```

All models are trained on GPU (Kaggle) and exported to **ONNX** for CPU-only inference, keeping end-to-end latency under 10 seconds on modest hardware with no GPU at inference time.

The surrounding system (Django Channels WebSocket backend, Celery task queues, Redis, MongoDB, plain-JS frontend) is documented in the repository's architecture docs; this README focuses on the deep learning pipeline.

---

## Repository Structure

> **Note:** This repository uses **multiple branches**. Check out the relevant branch and inspect the folder structure for the exact `.py` script paths before running anything — paths below reflect the main layout.

```
DeepCue/
├── kaggle_scripts/          # GPU training & export scripts (run on Kaggle/Colab, never locally)
│   ├── train_video_model.py       # EfficientNet-B0 + LSTM on RAVDESS
│   ├── train_audio_model.py       # wav2vec 2.0 fine-tuning on RAVDESS audio
│   ├── finetune_xlm_roberta.py    # XLM-RoBERTa on Hebrew sentiment data
│   ├── train_fusion_model.py      # Cross-Modal Transformer fusion training
│   ├── evaluate_models.py         # Unified evaluation (F1 / MAE gate checks)
│   └── export_and_quantize.py     # PyTorch → ONNX export
├── backend/                 # Django + Channels inference server (loads .onnx only)
├── frontend/                # Plain HTML/JS SPA (MediaPipe landmarks + audio capture)
├── models/                  # Exported .onnx weights
├── requirements.txt         # Local (backend) dependencies
└── requirements_kaggle.txt  # Training-environment dependencies
```

---

## Installation

**Requirements:** Python 3.12 (recommended — newer versions have incompatible C-extension wheels for several dependencies).

```bash
git clone https://github.com/RachelBSade/DeepCue.git
cd DeepCue

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

For the **training environment** (Kaggle/Colab notebooks), use `requirements_kaggle.txt` instead. Training scripts are written as importable `run()`-style functions for notebook execution — they are not meant to run locally.

---

## Running the Pipeline

### 1. Training (GPU — Kaggle/Colab)

Each script in `kaggle_scripts/` is self-contained: upload it to a notebook cell (or import it), point it at the dataset, and call its `run()` function. Order:

1. `train_video_model.py` — trains on RAVDESS video frames
2. `train_audio_model.py` — trains on RAVDESS audio
3. `finetune_xlm_roberta.py` — fine-tunes on Hebrew sentiment data
4. `train_fusion_model.py` — trains the Cross-Modal Transformer on the fusion feature space
5. `evaluate_models.py` — runs the F1/MAE gate checks
6. `export_and_quantize.py` — exports all checkpoints to ONNX

Download the exported `.onnx` files and place them under `models/`.

### 2. Inference (local CPU)

Set the model paths in `.env` (`VIDEO_MODEL_PATH`, `AUDIO_MODEL_PATH`, `TEXT_MODEL_PATH`, `FUSION_MODEL_PATH`), then follow the backend/frontend startup instructions in the repository docs. The backend loads ONNX weights only — it never trains.

> **Branch note (again):** script names and paths can differ between branches. Always verify with `git branch -a` and browse the branch's `kaggle_scripts/` folder before running.

---

## Evaluation Results & How to Interpret Them

| Model | Metric | Result | Threshold | Status |
|---|---|---|---|---|
| Video (EfficientNet-B0 + LSTM) | Macro F1 | **0.80** | ≥ 0.50 | ✅ Pass |
| Audio (wav2vec 2.0) | Macro F1 | **0.45** | ≥ 0.50 | ⚠️ Near threshold |
| Text (XLM-RoBERTa) | MAE | **0.046** | lower = better | ✅ Excellent |
| Fusion (Cross-Modal Transformer) | Macro F1 | **0.98** (simulation) | ≥ 0.50 | ✅ Architecture validated |

### Macro F1 (Video, Audio, Fusion — classification)

Macro F1 averages the F1 score (harmonic mean of precision and recall) **equally across all 8 emotion classes**, regardless of class frequency. This is deliberately stricter than accuracy for emotion recognition, where classes are imbalanced — a model that only predicts `neutral` scores high accuracy but near-zero macro F1. A macro F1 of 0.80 means the video model performs strongly and *consistently* across emotions, not just on the common ones.

The audio score (0.45) sits just under the 0.50 gate; analysis attributes part of the gap to a train/eval source mismatch (trained on audio-only RAVDESS recordings, evaluated on audio extracted from the audio-visual MP4 set). It remains an open fine-tuning item.

### MAE (Text — regression)

The text expert predicts a *continuous* sentiment score, so it is evaluated with **Mean Absolute Error**: the average absolute distance between predicted and true sentiment. An MAE of ~0.046 on a normalized scale means predictions land within ~5% of ground truth on average — an excellent result for Hebrew, a comparatively low-resource language in sentiment analysis.

### The Fusion result — Architectural Viability Simulation

Real *paired* multimodal data (the same interview moment labeled across video, audio, and text simultaneously) does not yet exist for this domain. The fusion model was therefore trained and evaluated on a **carefully constructed synthetic dataset** that simulates the empirical output distributions of the three experts, with intensity-modulated class means.

This is a deliberate **architectural proof-of-concept**: the 0.98 macro F1 demonstrates that the cross-modal attention mechanism can learn the 17-dimensional fused feature space and recover the underlying emotion signal near-perfectly when that signal is present. It validates the architecture and the end-to-end pipeline; it is **not** a claim about real-world fusion performance, which awaits real paired interview data collected through the deployed system.

---

## Datasets

- **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song (video + audio experts). Actor-disjoint train/validation splits are used to prevent identity leakage.
- **Hebrew Sentiment** (`omilab/hebrew_sentiment`) — Hebrew-language sentiment corpus for the text expert (adopted after CMU-MOSI availability issues).
- **Synthetic fusion set** — simulated 17-dim expert-output vectors for the fusion architecture study (see above).

---

## License & Contact

Academic project — Rachel Brodsky. See repository for license details.
