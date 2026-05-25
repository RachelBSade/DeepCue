# DeepCue

Real-time multimodal emotion recognition system for Hebrew-speaking job interview candidates.

Analyzes facial micro-expressions, paralinguistic audio features, and speech semantics simultaneously, fusing all three streams into a unified 8-class emotion output. Generates a structured PDF report at the end of each session.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ MediaPipe JS │  │ Web Audio API│  │    UI Controller      │  │
│  │ Face Mesh    │  │ Mic capture  │  │ Emotion panel, live   │  │
│  │ 468 landmarks│  │ 3s chunks    │  │ transcript, controls  │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
│         │   landmark JSON │   base64 audio                       │
└─────────┼─────────────────┼───────────────────────────────────────┘
          │                 │  WebSocket (ws/interview/<session_id>/)
┌─────────▼─────────────────▼───────────────────────────────────────┐
│                    Django + Django Channels                        │
│                    InterviewConsumer (async)                       │
│                    motor → MongoDB Atlas (async writes)            │
└──────────┬──────────────────┬─────────────────────────────────────┘
           │ Celery tasks      │
    ┌──────▼──────┐    ┌───────▼──────┐    ┌──────────────┐
    │ video_queue │    │ audio_queue  │    │ fusion_queue │
    │             │    │              │    │              │
    │ Video       │    │ Audio        │    │ Fusion       │
    │ Pipeline    │    │ Pipeline     │    │ Pipeline     │
    │ EfficientNet│    │ wav2vec 2.0  │    │ Cross-modal  │
    │ -B0 + LSTM  │    │ + librosa    │    │ Transformer  │
    │ (ONNX/CPU)  │    │ (ONNX/CPU)  │    │ + MLP head   │
    └──────┬──────┘    └──────┬───────┘    └──────┬───────┘
           │                  │   Text Pipeline    │
           │            ┌─────▼──────┐             │
           │            │  Whisper   │             │
           │            │ (Hebrew    │             │
           │            │  STT) +    │             │
           │            │ XLM-RoBERTa│             │
           │            │ (ONNX/CPU) │             │
           │            └─────┬──────┘             │
           │                  │                    │
           └──────────────────┴────────────────────┘
                         Redis cache
                    (per-session modality scores)
                              │
                    ┌─────────▼──────────┐
                    │   MongoDB Atlas    │
                    │  Session document  │
                    │  Emotion frames    │
                    │  Transcript segs   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  ReportLab PDF     │
                    │  5-section report  │
                    │  Hebrew RTL text   │
                    │  GridFS storage    │
                    └────────────────────┘
```

**8 emotion classes:** `neutral` · `confident` · `anxious` · `happy` · `sad` · `angry` · `surprised` · `uncertain`

---

## Environment Split

This project is split across two compute environments:

| Concern | Environment | Location |
|---|---|---|
| Model training & fine-tuning | Kaggle GPU (T4/P100/A100) | `kaggle_scripts/` |
| ONNX export & INT8 quantization | Kaggle GPU | `kaggle_scripts/` |
| Live inference (all 4 pipelines) | Local Windows CPU | `backend/apps/inference/` |
| WebSocket server + task queue | Local / any server | `backend/` |
| Frontend | Browser | `frontend/` |

Training never runs locally. The Django backend only loads pre-quantized `.onnx` weight files produced by the Kaggle scripts and dropped into `models/`.

---

## Directory Structure

```
DeepCue/
├── backend/
│   ├── deepcue_backend/          # Django project package
│   │   └── settings/             # base.py · local.py · production.py
│   ├── apps/
│   │   ├── sessions_app/         # Session lifecycle, MongoDB writes
│   │   ├── inference/            # VideoEmotionPipeline, AudioEmotionPipeline,
│   │   │                         # TextEmotionPipeline, FusionPipeline
│   │   └── reporting/            # ReportLab PDF generator, GridFS storage
│   ├── tasks/                    # Celery tasks (video, audio, text, fusion, report)
│   └── db/                       # MongoDB client, TypedDict document schemas
├── frontend/                     # HTML/JS SPA — no build step required
├── kaggle_scripts/               # GPU training + ONNX export (run on Kaggle only)
├── models/
│   ├── video/                    # efficientnet_lstm.onnx
│   ├── audio/                    # wav2vec2_classifier.onnx
│   ├── text/                     # xlm_roberta_sentiment.onnx + whisper_cache/
│   └── fusion/                   # cross_modal_transformer.onnx
├── reports/                      # Generated PDFs (gitignored)
├── scripts/                      # Dev utility scripts
├── .env.example                  # All required environment variables
├── docker-compose.yml            # Redis + MongoDB (local dev infrastructure)
├── requirements.txt              # Django backend — CPU inference
└── requirements_kaggle.txt       # Kaggle training — GPU
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for Redis + MongoDB)
- A Kaggle account (for training; not required to run the inference server)

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd DeepCue
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set DJANGO_SECRET_KEY and local MongoDB URI:
# MONGODB_URI=mongodb://deepcue:deepcue_local@localhost:27017/deepcue?authSource=admin
```

### 4. Start infrastructure services

```bash
docker compose up -d
# Redis → localhost:6379
# MongoDB → localhost:27017
# Mongo Express UI → http://localhost:8081
```

### 5. Run Django migrations and start the server

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

### 6. Start Celery workers (separate terminals)

```bash
# Video + Audio workers
celery -A deepcue_backend worker -Q video_queue,audio_queue -c 2 -l info

# Fusion + Report workers
celery -A deepcue_backend worker -Q fusion_queue -c 1 -l info
```

### 7. Open the frontend

Open `frontend/index.html` in your browser. Allow camera and microphone access when prompted.

---

## Model Artifacts

After running the Kaggle training scripts, download the exported `.onnx` files and place them here:

```
models/video/efficientnet_lstm.onnx
models/audio/wav2vec2_classifier.onnx
models/text/xlm_roberta_sentiment.onnx
models/fusion/cross_modal_transformer.onnx
```

Paths are configurable via `.env` (`VIDEO_MODEL_PATH`, etc.) without touching code.

---

## Performance Targets

| Metric | Target |
|---|---|
| End-to-end inference latency | < 10 seconds (weak Windows CPU) |
| Macro F1-score (RAVDESS) | ≥ 0.50 |
| Macro F1-score (CMU-MOSI) | ≥ 0.50 |
| WebSocket reconnect | Exponential backoff, max 5 retries |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 · Vanilla JS · MediaPipe Face Mesh JS · Web Audio API |
| WebSocket server | Django Channels 4 · Daphne (ASGI) |
| Task queue | Celery 5 · Redis 7 |
| Database | MongoDB Atlas · motor (async) · pymongo (sync) |
| Video inference | EfficientNet-B0 + LSTM → ONNX (onnxruntime) |
| Audio inference | wav2vec 2.0 + librosa features → ONNX |
| Text inference | OpenAI Whisper (STT) + XLM-RoBERTa → ONNX |
| Fusion | Cross-modal Transformer + MLP head → ONNX |
| Reporting | ReportLab (PDF · RTL Hebrew · charts) |
| Training | PyTorch · HuggingFace Transformers · timm · optimum |
