# DeepCue — Build Progress Log

This file is updated after every completed phase. It records what was built, what files were created or modified, and where to find each piece of work.

---

## Phase 0 — Project Scaffold & Environment
**Status: Complete**

### What was done
Set up the full project skeleton: directory structure, dependency files, environment config, local dev infrastructure, and root documentation. No application code yet — this phase establishes the foundation every later phase builds on.

### Files created

| File | Purpose |
|---|---|
| `deepcue_workflow_checklist.md` | Master checklist — single source of truth for project progress |
| `requirements.txt` | Django backend pip dependencies (CPU-only PyTorch, pinned versions) |
| `requirements_kaggle.txt` | Kaggle training pip dependencies (CUDA PyTorch, optimum, timm, pinned versions) |
| `.env.example` | Template for all environment variables — copy to `.env` and fill in real values |
| `docker-compose.yml` | Local dev infrastructure: Redis (port 6379) + MongoDB (port 27017) + Mongo Express UI (port 8081) |
| `README.md` | Architecture overview, ASCII data-flow diagram, environment split table, setup instructions, performance targets |

### Directories created

| Directory | Purpose |
|---|---|
| `backend/deepcue_backend/settings/` | Django project package + split settings (base / local / production) |
| `backend/apps/sessions_app/` | Django app: session lifecycle and MongoDB writes |
| `backend/apps/inference/` | Django app: all 4 ML inference pipeline classes |
| `backend/apps/reporting/` | Django app: ReportLab PDF generation and GridFS storage |
| `backend/tasks/` | Celery task modules (cross-cutting concern, lives at backend root) |
| `backend/db/` | MongoDB async/sync client and TypedDict document schemas |
| `frontend/` | HTML/JS single-page application (no build step) |
| `kaggle_scripts/` | GPU training + ONNX export scripts (run on Kaggle, not locally) |
| `models/video/` | Drop `efficientnet_lstm.onnx` here after Kaggle training |
| `models/audio/` | Drop `wav2vec2_classifier.onnx` here after Kaggle training |
| `models/text/` | Drop `xlm_roberta_sentiment.onnx` + Whisper cache here |
| `models/fusion/` | Drop `cross_modal_transformer.onnx` here after Kaggle training |
| `reports/` | Generated PDF session reports (gitignored) |
| `scripts/` | Dev utility scripts |

---

## Phase 1 — Django Project Bootstrap
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `backend/manage.py` | Django CLI entry point; defaults to `settings.local` |
| `backend/deepcue_backend/__init__.py` | Exposes `celery_app` so `-A deepcue_backend` resolves |
| `backend/deepcue_backend/celery.py` | Celery app; autodiscovers tasks from all INSTALLED_APPS |
| `backend/deepcue_backend/settings/base.py` | All shared settings: apps, middleware, Channels, Celery, MongoDB, inference paths |
| `backend/deepcue_backend/settings/local.py` | Dev overrides: DEBUG=True, CORS open |
| `backend/deepcue_backend/settings/production.py` | Production overrides: HTTPS, strict CORS, HSTS |
| `backend/deepcue_backend/asgi.py` | ProtocolTypeRouter: HTTP → Django, WebSocket → Channels |
| `backend/deepcue_backend/urls.py` | Root URL router |
| `backend/apps/sessions_app/apps.py` | Session lifecycle app config |
| `backend/apps/inference/apps.py` | Inference pipeline app config |
| `backend/apps/reporting/apps.py` | PDF reporting app config |
| `backend/db/mongo_client.py` | `sync_db` (pymongo) and `async_db` (motor) singletons |
| `backend/db/schemas.py` | TypedDicts: `InterviewSession`, `EmotionFrame`, `TranscriptSegment` |
| `backend/apps/sessions_app/views.py` | `GET /api/health/` — probes Django, Redis, MongoDB |
| `backend/apps/sessions_app/urls.py` | URL patterns for sessions_app |

---

## Phase 2 — WebSocket Protocol & Django Channels Consumer
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `backend/apps/sessions_app/protocol.py` | Full WS message protocol: type constants + TypedDict schemas for all inbound/outbound messages |
| `backend/apps/sessions_app/consumers.py` | `InterviewConsumer`: connect/disconnect/receive, all 5 inbound handlers, 4 outbound group handlers, `interviewer_audio` stub |
| `backend/apps/sessions_app/routing.py` | WebSocket URL pattern `ws/interview/<session_id>/` (UUID4 regex) |

---

## Phase 3 — JS Web Frontend
**Status: Complete (3.8 deferred — requires full backend running)**

### Files created

| File | Purpose |
|---|---|
| `frontend/index.html` | SPA shell: webcam feed, emotion panel (8 bars), Hebrew transcript, controls |
| `frontend/style.css` | Dark theme, per-emotion colour palette, responsive grid, RTL transcript support |
| `frontend/mediapipe_handler.js` | `MediaPipeHandler`: loads Face Mesh from CDN, emits 468 normalised {x,y,z} landmarks per frame |
| `frontend/audio_handler.js` | `AudioHandler`: buffers 3 s PCM chunks, encodes 16-bit mono WAV, emits base64 via callback |
| `frontend/websocket_client.js` | `WebSocketClient`: typed send methods, handler registry, exponential backoff reconnect (5 retries, cap 30 s) |
| `frontend/ui_controller.js` | `UIController`: all DOM mutations — state transitions, emotion bars, transcript feed, timer, errors |
| `frontend/main.js` | Orchestrator: generates `session_id`, wires all modules, drives idle→connecting→active→ended state machine |

---

## Phase 4 — Celery Task Queue & Inference Orchestration
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `backend/tasks/video_tasks.py` | `process_video_frame`: runs VideoEmotionPipeline, caches score, triggers fusion |
| `backend/tasks/audio_tasks.py` | `process_audio_chunk`: decodes WAV, runs AudioEmotionPipeline, caches score |
| `backend/tasks/text_tasks.py` | `process_transcript_segment`: Whisper transcription + XLM-RoBERTa score, pushes transcript_update |
| `backend/tasks/fusion_tasks.py` | `run_fusion`: reads 3 Redis scores, runs FusionPipeline, writes EmotionFrame to MongoDB, pushes emotion_result |
| `backend/tasks/report_tasks.py` | `generate_report`: pulls MongoDB data, generates PDF, stores in GridFS, pushes final session_ended |
| `backend/deepcue_backend/celery.py` | Celery app + Beat schedule stub (frame-driven fusion; Beat heartbeat commented out) |

### Also restored (missing from dev branch)
All Phase 1 files: `manage.py`, `settings/`, `asgi.py`, `urls.py`, `db/`, `apps/*/apps.py`, `__init__.py` files.

---

## Phase 5 — Inference Pipelines (CPU-Optimized)
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `backend/apps/inference/video_pipeline.py` | `VideoEmotionPipeline`: landmarks→224×224 image, per-session LSTM buffer, ONNX inference |
| `backend/apps/inference/audio_pipeline.py` | `AudioEmotionPipeline`: librosa feature extraction (pitch, RMS, ZCR, MFCCs) + wav2vec ONNX |
| `backend/apps/inference/text_pipeline.py` | `TextEmotionPipeline`: Whisper Hebrew STT + XLM-RoBERTa ONNX sentiment scoring |
| `backend/apps/inference/fusion_pipeline.py` | `FusionPipeline`: 3-score input → Cross-modal Transformer ONNX → 8-class softmax output |

### Also updated
`backend/tasks/video_tasks.py` — passes `session_id` to `predict()` for LSTM buffer keying.

---

## Phase 6 — Kaggle Training Scripts
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `kaggle_scripts/train_video_model.py` | `RAVDESSVideoDataset` + `EfficientNetLSTM` (timm B0 + BiLSTM); trains on RAVDESS mp4s; exports `efficientnet_lstm.onnx` |
| `kaggle_scripts/train_audio_model.py` | `RAVDESSAudioDataset` + `Wav2Vec2EmotionClassifier`; 2-stage fine-tuning (freeze → full); exports `wav2vec2_classifier.onnx` with dual-input (waveform + features) |
| `kaggle_scripts/finetune_xlm_roberta.py` | `XLMRobertaRegressor`; trained on CMU-MOSI + optional Hebrew CSV; exports `xlm_roberta_sentiment.onnx` |
| `kaggle_scripts/train_fusion_model.py` | `CrossModalTransformer` (3-token self-attention + MLP head); supports synthetic or real modality scores; exports `cross_modal_transformer.onnx` |
| `kaggle_scripts/evaluate_models.py` | CLI evaluator for all 4 models; reports Macro F1 per model; asserts >= 0.50 threshold |
| `kaggle_scripts/export_and_quantize.py` | Unified export + INT8 dynamic quantization for all models; prints copy instructions for `models/` dirs |
| `kaggle_scripts/README.md` | Dataset setup, execution order, artifact destinations, performance targets |

---

## Phase 7 — PDF Reporting (ReportLab)
**Status: Complete**

### Files created

| File | Purpose |
|---|---|
| `backend/apps/reporting/report_generator.py` | `InterviewReportGenerator.generate()` — 5-section A4 PDF: executive summary, emotion timeline chart, text insights (Hebrew RTL), model performance table, rule-based recommendations |
| `backend/apps/reporting/pdf_storage.py` | `store_report()` / `retrieve_report()` — GridFS read/write; idempotent (deletes old file before re-storing) |
| `backend/apps/reporting/views.py` | `download_report()` — `GET /api/report/<session_id>/` — streams PDF bytes; 404 if not found |
| `backend/apps/reporting/urls.py` | URL pattern for the download endpoint |

### Also updated
`backend/deepcue_backend/urls.py` — added `path("api/", include("apps.reporting.urls"))`

---

## Phase 8 — Integration Testing & Performance Validation
**Status: Not started**

---

## Phase 9 — Hardening, Cleanup & Documentation
**Status: Not started**
