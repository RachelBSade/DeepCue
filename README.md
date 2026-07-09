# DeepCue

**Real-time multimodal emotion recognition for Hebrew-speaking job interview candidates.**

DeepCue analyzes three signal streams simultaneously — facial micro-expressions (video), paralinguistic features (audio), and speech semantics (Hebrew text via ASR) — and fuses them into a unified 8-class emotion prediction, streamed live over WebSocket during an interview session. At session end, it generates a structured 5-section PDF report with full Hebrew RTL support.

All inference runs on **CPU-only consumer hardware** under a **< 10 second end-to-end latency budget**, using quantized/optimized ONNX models trained offline on GPU.

> **Status: active development.** The core pipeline (three modality models, fusion, live streaming, PDF reporting) is functional and under continuous improvement. Planned next: a bidirectional AI interviewer (live spoken questions driven by the candidate's emotional state), an upgraded 8-class text model, and fusion retraining on real paired interview data. See [Roadmap](#roadmap).

**8 emotion classes:** `neutral` · `confident` · `anxious` · `happy` · `sad` · `angry` · `surprised` · `uncertain`

---

## System Architecture

```
 Browser ──► MediaPipe Face Mesh (468 landmarks) ─┐
         ──► Web Audio API (3s chunks)            ├─► WebSocket ws/interview/<session_id>/
                                                  ┘
 Django Channels (InterviewConsumer)
   └─ validate → rate-limit → dispatch to Celery (never infers inline)
        ├─ video_queue  → EfficientNet-B0 + LSTM        (ONNX, CPU)
        ├─ audio_queue  → wav2vec 2.0 + librosa         (ONNX, CPU)
        │                 Whisper ASR → XLM-RoBERTa     (ONNX, CPU)
        └─ per-modality scores → Redis (keyed by session)
             └─ fusion_queue → Cross-Modal Transformer (17-dim input)
                  └─ emotion_result → browser (Channels group) + MongoDB
 session_end → Celery report task → ReportLab PDF (Hebrew RTL) → GridFS
```

Full design rationale — latency budget, ONNX decision, queue topology, failure handling — in **[docs/architecture.md](docs/architecture.md)**.

### Design constraints

| Constraint | Target |
|---|---|
| End-to-end latency (all 3 modalities → fused result) | < 10 s on consumer CPU, no GPU |
| Per-model quality gate | Macro F1 ≥ 0.50 (RAVDESS / sentiment benchmarks) |
| Live-session resilience | A pipeline failure degrades to neutral — never crashes the session |

### Tech stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 · vanilla JS (ES modules, no build step) · MediaPipe Face Mesh · Web Audio API |
| Realtime server | Django 4.2 · Channels 4 · Daphne (ASGI) |
| Task queue | Celery 5 · Redis (broker, result backend, channel layer, score cache) |
| Persistence | MongoDB (motor async + pymongo) · GridFS for PDFs |
| Inference runtime | ONNX Runtime (CPU) — models exported & optimized on GPU, consumed here |
| Models | EfficientNet-B0+LSTM (video) · wav2vec 2.0 (audio) · Whisper + XLM-RoBERTa (text) · Cross-Modal Transformer (fusion) |
| Reporting | ReportLab — 5-section PDF, RTL Hebrew, charts |
| Training (offline, GPU) | PyTorch · HuggingFace Transformers · timm · optimum |

---

## Repository Map

```
DeepCue/
├── backend/            # Django + Channels inference server (CPU-only runtime)
│   ├── apps/           # sessions_app (WS protocol) · inference (4 pipelines) · reporting
│   ├── tasks/          # Celery tasks per queue: video, audio, text, fusion, report
│   ├── db/             # MongoDB client + document schemas
│   └── tests/          # pytest suite — runs with zero live infrastructure
├── frontend/           # HTML/JS SPA — served by serve.py, no build step
├── training/           # Clean training/export scripts (run on Kaggle GPU, never locally)
├── notebooks/          # Research notebooks with preserved outputs (EDA → training → eval)
├── models/             # ONNX weights land here (gitignored — see Model Artifacts)
├── docs/               # architecture.md · RESULTS.md · DEPLOYMENT.md
├── requirements.txt        # Production/inference dependencies (CPU)
└── requirements-train.txt  # Training dependencies (Kaggle GPU)
```

Training and inference are strictly separated: `training/` produces `.onnx` artifacts on Kaggle GPUs; `backend/` only ever loads them. Nothing trains locally.

---

## Quickstart

### Prerequisites

- **Python 3.12** (3.14 has broken C-extension wheels for several dependencies)
- **Redis 6/7** — on Windows use [Memurai](https://www.memurai.com/) as a native service (Docker Desktop's TCP proxy breaks the asyncio streams Channels/Daphne rely on)
- **MongoDB** — Atlas free tier or a local instance

### 1. Clone and install

```bash
git clone https://github.com/RachelBSade/DeepCue.git
cd DeepCue
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env       # cp on macOS/Linux
```

Set `DJANGO_SECRET_KEY` and `MONGODB_URI`. Redis URLs should use `localhost` (not `127.0.0.1`). Model paths (`VIDEO_MODEL_PATH` etc.) default to `models/<modality>/` — see [Model Artifacts](#model-artifacts).

### 3. Migrate and start the ASGI server

```bash
cd backend
python manage.py migrate
python run_daphne.py -b 0.0.0.0 -p 8000 deepcue_backend.asgi:application
```

### 4. Start Celery workers (one terminal each)

```bash
celery -A deepcue_backend worker -Q video_queue  -c 1 -l info
celery -A deepcue_backend worker -Q audio_queue  -c 1 -l info
celery -A deepcue_backend worker -Q fusion_queue -c 1 -l info
```

> On Windows, if you run a single combined worker instead, add `--pool=solo` — Celery's default prefork pool requires `os.fork()`.

### 5. Serve the frontend

```bash
cd frontend
python serve.py
```

Open `http://localhost:5500` and allow camera + microphone access. (Don't open `index.html` via `file://` or `python -m http.server` — `serve.py` exists to fix Windows MIME types for ES modules.)

---

## Model Artifacts

The `.onnx` weights are not committed. Download them from the [latest release](https://github.com/RachelBSade/DeepCue/releases) and place them at:

```
models/video/efficientnet_lstm.onnx
models/audio/wav2vec2_classifier.onnx
models/text/xlm_roberta_sentiment.onnx
models/fusion/cross_modal_transformer.onnx
```

Paths are configurable via `.env` without touching code. Whisper weights download automatically on first run (`WHISPER_MODEL_SIZE=base`, cached under `models/text/whisper_cache/`).

---

## Testing

```bash
cd backend
python -m pytest -q
```

The suite runs under `deepcue_backend.settings.test` with an in-memory channel layer and `CELERY_TASK_ALWAYS_EAGER=True` — **no live Redis, MongoDB, or `.onnx` files required**, which also makes it CI-friendly.

---

## Roadmap

- **Bidirectional AI interviewer** — live spoken questions adapting to the candidate's emotional state (WebSocket protocol slot already reserved)
- **Text model v2** — upgrade the sentiment regressor to an 8-class emotion classifier matching the video/audio heads
- **Fusion retraining on real data** — replace synthetic fusion training data with paired multimodal recordings from real sessions
- **Audio model fine-tuning** — close the remaining gap to the F1 quality gate on cross-source evaluation

## License

MIT — see [LICENSE](LICENSE).
