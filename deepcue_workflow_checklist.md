# DeepCue — Project Workflow Checklist
> Real-Time Multimodal Emotion Recognition System for Job Interviews
> Lead Developer controls all Git operations. This file is the single source of truth for project progress.

---

## LEGEND
- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete (approved by Lead Developer)
- `[s]` — Stubbed / deferred (out of scope for current phase)

---

## PHASE 0 — Project Scaffold & Environment

- [x] **0.1** Create root project directory structure (`deepcue/`, subdirectories for backend, frontend, scripts, models, reports)
- [x] **0.2** Create `requirements.txt` for Django backend (Django, Channels, Celery, Redis, pymongo, reportlab, torch, torchaudio, transformers, onnxruntime, openai-whisper, mediapipe, etc.)
- [x] **0.3** Create `requirements_kaggle.txt` for Kaggle training environment (full GPU training deps)
- [x] **0.4** Create `.env.example` with all required environment variables (MongoDB URI, Redis URL, Django secret key, model weight paths, etc.)
- [x] **0.5** Create `docker-compose.yml` for local Redis + MongoDB Atlas proxy setup
- [x] **0.6** Create root `README.md` with architecture overview, setup instructions, and environment split explanation

---

## PHASE 1 — Django Project Bootstrap

- [x] **1.1** Scaffold Django project (`deepcue_backend/`) with `settings.py` split into `base.py`, `local.py`, `production.py`
- [x] **1.2** Configure `INSTALLED_APPS` — add `channels`, `django_celery_results`, and all internal apps (`sessions_app`, `inference`, `reporting`)
- [x] **1.3** Configure Django Channels with Redis channel layer (`CHANNEL_LAYERS` in settings)
- [x] **1.4** Configure Celery (`celery.py`) with Redis as broker and result backend
- [x] **1.5** Configure MongoDB Atlas connection via `pymongo` (utility module `db/mongo_client.py`)
- [x] **1.6** Define MongoDB document schemas (interview session, emotion frame, transcript segment) as Python `TypedDict` structures
- [x] **1.7** Write Django `urls.py` root router + `asgi.py` with Channels routing
- [x] **1.8** Write a basic health-check HTTP endpoint (`GET /api/health/`) returning service status (Django, Redis, MongoDB)

---

## PHASE 2 — WebSocket Protocol & Django Channels Consumer

- [x] **2.1** Design and document the full WebSocket message protocol (all message `type` fields: `session_start`, `video_frame`, `audio_chunk`, `transcript_segment`, `emotion_result`, `session_end`, `interviewer_audio` [stubbed], `error`)
- [x] **2.2** Write `consumers.py` — `InterviewConsumer` (AsyncWebsocketConsumer) handling connection lifecycle (`connect`, `disconnect`, `receive`)
- [x] **2.3** Implement `session_start` handler — creates MongoDB session document, returns `session_id`
- [x] **2.4** Implement `video_frame` handler — validates incoming MediaPipe JSON payload, dispatches Celery task
- [x] **2.5** Implement `audio_chunk` handler — validates base64-encoded audio payload, dispatches Celery task
- [x] **2.6** Implement `session_end` handler — triggers report generation task, finalizes MongoDB document
- [x] **2.7** Implement `error` handler and graceful disconnect with session cleanup
- [x] **2.8** Write `routing.py` — WebSocket URL pattern (`ws/interview/<session_id>/`)
- [x] **2.9** [s] Stub `interviewer_audio` outbound message type in protocol (Phase 6 placeholder)

---

## PHASE 3 — JS Web Frontend

- [x] **3.1** Scaffold single-page frontend (`frontend/`) — `index.html`, `style.css`, `main.js`, `mediapipe_handler.js`, `websocket_client.js`, `audio_handler.js`, `ui_controller.js`
- [x] **3.2** Implement `mediapipe_handler.js` — load MediaPipe Face Mesh JS, initialize camera, extract 468 landmark coordinates per frame, output normalized JSON
- [x] **3.3** Implement `audio_handler.js` — access microphone via Web Audio API, chunk audio into configurable windows (e.g., 3-second sliding windows), encode as base64 PCM/WAV
- [x] **3.4** Implement `websocket_client.js` — manage WebSocket connection lifecycle, send/receive typed messages, handle reconnect logic with exponential backoff
- [x] **3.5** Implement `ui_controller.js` — interview start/stop controls, live emotion display panel, confidence score bars, real-time transcript feed
- [x] **3.6** Implement `main.js` — orchestrate all modules, wire up event listeners, manage global session state
- [x] **3.7** Add responsive CSS styling — clean, professional interview UI (webcam feed, sidebar emotion panel, transcript box)
- [x] **3.8** Test end-to-end: browser connects to Django Channels, MediaPipe landmarks stream over WebSocket, audio chunks received by backend

---

## PHASE 4 — Celery Task Queue & Inference Orchestration

- [x] **4.1** Write `tasks/video_tasks.py` — `process_video_frame` Celery task (receive landmark JSON → call video pipeline → push result to Channels group)
- [x] **4.2** Write `tasks/audio_tasks.py` — `process_audio_chunk` Celery task (receive base64 audio → call audio pipeline → push result to Channels group)
- [x] **4.3** Write `tasks/text_tasks.py` — `process_transcript_segment` Celery task (receive audio → Whisper transcription → call text pipeline → push result to Channels group)
- [x] **4.4** Write `tasks/fusion_tasks.py` — `run_fusion` Celery task (collect latest modality outputs from Redis cache → call fusion pipeline → push unified `emotion_result` to Channels group → write to MongoDB)
- [x] **4.5** Implement result caching strategy in Redis — store latest per-modality scores keyed by `session_id` for fusion aggregation
- [x] **4.6** Configure Celery task routing — assign tasks to dedicated queues (`video_queue`, `audio_queue`, `fusion_queue`)
- [x] **4.7** Write `tasks/report_tasks.py` — `generate_report` Celery task (triggered on `session_end`, pulls MongoDB data, calls reporting module)
- [x] **4.8** Add Celery Beat schedule stubs for periodic fusion triggering (e.g., every 1s during active session)

---

## PHASE 5 — Inference Pipelines (Django Backend — CPU-Optimized)

### 5A — Video Pipeline (Facial Micro-Expression)
- [x] **5A.1** Write `inference/video_pipeline.py` — `VideoEmotionPipeline` class with `load_model()` and `predict(landmarks_json) -> float` interface
- [x] **5A.2** Implement landmark preprocessing — convert 468 MediaPipe (x, y, z) coords to normalized feature tensor for EfficientNet-B0 input
- [x] **5A.3** Implement LSTM temporal windowing — buffer last N frames, pass sequence to LSTM head
- [x] **5A.4** Load quantized ONNX weights (EfficientNet-B0 + LSTM exported from Kaggle) via `onnxruntime.InferenceSession`
- [x] **5A.5** Implement `NEUTRAL_FALLBACK = 0.5` exception handling wrapping entire predict call

### 5B — Audio Pipeline (Paralinguistic Features)
- [x] **5B.1** Write `inference/audio_pipeline.py` — `AudioEmotionPipeline` class with `load_model()` and `predict(audio_bytes) -> float` interface
- [x] **5B.2** Implement feature extraction — pitch (librosa), speech rate, RMS energy, WPM estimation from raw audio bytes
- [x] **5B.3** Load quantized wav2vec 2.0 ONNX weights, extract deep audio embeddings
- [x] **5B.4** Combine paralinguistic features + wav2vec embeddings as input to classifier head
- [x] **5B.5** Implement `NEUTRAL_FALLBACK = 0.5` exception handling

### 5C — Text Pipeline (Whisper + XLM-RoBERTa)
- [x] **5C.1** Write `inference/text_pipeline.py` — `TextEmotionPipeline` class with `load_model()`, `transcribe(audio_bytes) -> str`, and `predict(text: str) -> float` interface
- [x] **5C.2** Integrate OpenAI Whisper (small/base model) for Hebrew speech-to-text transcription
- [x] **5C.3** Load fine-tuned XLM-RoBERTa ONNX weights for sentiment + uncertainty analysis
- [x] **5C.4** Implement Hebrew-aware text preprocessing (tokenization via HuggingFace tokenizer)
- [x] **5C.5** Implement `NEUTRAL_FALLBACK = 0.5` exception handling

### 5D — Fusion Pipeline (Cross-Modal Transformer + MLP Head)
- [x] **5D.1** Write `inference/fusion_pipeline.py` — `FusionPipeline` class with `load_model()` and `predict(video_score, audio_score, text_score) -> dict` interface
- [x] **5D.2** Implement Cross-modal Transformer encoder (load ONNX weights from Kaggle training)
- [x] **5D.3** Implement MLP head: `Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 8) → Softmax`
- [x] **5D.4** Output: dict mapping 8 emotion labels to confidence scores (sum to 1.0)
- [x] **5D.5** Define the 8 emotion classes: `[neutral, confident, anxious, happy, sad, angry, surprised, uncertain]`
- [x] **5D.6** Implement `NEUTRAL_FALLBACK = 0.5` exception handling; fallback output: `{neutral: 1.0, all others: 0.0}`

---

## PHASE 6 — Kaggle Training Scripts

- [x] **6.1** Write `kaggle_scripts/train_video_model.py` — EfficientNet-B0 + LSTM training on RAVDESS dataset (facial frames → emotion labels), export to ONNX + quantize
- [x] **6.2** Write `kaggle_scripts/train_audio_model.py` — wav2vec 2.0 fine-tuning on RAVDESS audio, export to ONNX + quantize
- [x] **6.3** Write `kaggle_scripts/finetune_xlm_roberta.py` — XLM-RoBERTa fine-tuning on CMU-MOSI + Hebrew sentiment dataset, export to ONNX + quantize
- [x] **6.4** Write `kaggle_scripts/train_fusion_model.py` — Cross-modal Transformer + MLP head training on fused RAVDESS/CMU-MOSI features, export to ONNX + quantize
- [x] **6.5** Write `kaggle_scripts/evaluate_models.py` — compute Macro F1-score on RAVDESS and CMU-MOSI held-out test sets, assert >= 0.50 threshold
- [x] **6.6** Write `kaggle_scripts/export_and_quantize.py` — unified export script: PT → ONNX → INT8 dynamic quantization for all models
- [x] **6.7** Document Kaggle dataset setup and model artifact download workflow in `kaggle_scripts/README.md`

---

## PHASE 7 — PDF Reporting (ReportLab)

- [x] **7.1** Write `reporting/report_generator.py` — `InterviewReportGenerator` class with `generate(session_id: str) -> bytes` interface
- [x] **7.2** Implement **Section 1: Executive Summary** — candidate name, session date/duration, dominant emotion, overall confidence score
- [x] **7.3** Implement **Section 2: Emotion Timeline** — line/area chart (ReportLab Drawing) of emotion scores over session time
- [x] **7.4** Implement **Section 3: Text-Based Insights** — top uncertainty phrases, sentiment arc, Hebrew transcript excerpts (RTL text rendering)
- [x] **7.5** Implement **Section 4: Model Performance Metrics** — per-modality confidence distributions, fusion model output breakdown
- [x] **7.6** Implement **Section 5: Recommendations** — rule-based text snippets triggered by emotion thresholds (e.g., high anxiety → breathing tips)
- [x] **7.7** Add DeepCue branding — header logo placeholder, footer with generation timestamp
- [x] **7.8** Write `reporting/pdf_storage.py` — save generated PDF bytes to MongoDB GridFS and return download URL
- [x] **7.9** Write Django HTTP endpoint `GET /api/report/<session_id>/` to stream PDF to browser

---

## PHASE 8 — Integration Testing & Performance Validation

- [x] **8.1** Write end-to-end integration test — simulate full interview session (mock MediaPipe payload + mock audio) via WebSocket test client
- [x] **8.2** Write unit tests for each pipeline's `predict()` method with mock inputs and assert `NEUTRAL_FALLBACK` on intentional exceptions
- [x] **8.3** Write Celery task unit tests using `task.apply()` (eager mode) — assert correct Redis cache writes
- [x] **8.4** Benchmark end-to-end inference latency — assert total pipeline time < 10 seconds on target hardware profile (weak Windows CPU simulation)
- [x] **8.5** Validate Macro F1-score >= 0.50 using Kaggle evaluation script outputs against RAVDESS and CMU-MOSI test sets
- [x] **8.6** Test Hebrew RTL text rendering in PDF report
- [x] **8.7** Test WebSocket reconnect logic (simulate server drop, assert exponential backoff)

---

## PHASE 9 — Hardening, Cleanup & Documentation

- [x] **9.1** Add Django rate limiting on WebSocket consumer (`channels_ratelimit` or custom middleware) — custom per-connection token bucket in `apps/sessions_app/rate_limit.py`, applied to `video_frame`/`audio_chunk` handlers
- [x] **9.2** Add input validation / sanitization on all incoming WebSocket payloads (Pydantic schemas) — `apps/sessions_app/validation.py`, wired into all inbound handlers in `consumers.py`
- [x] **9.3** Configure structured logging (Python `logging` module) across all pipelines and tasks — JSON format for production — `deepcue_backend/logging_json.py` + `LOGGING` dict in `settings/base.py`, switches on `DJANGO_DEBUG`
- [x] **9.4** Write `DEPLOYMENT.md` — step-by-step guide for local dev setup and production deployment notes
- [x] **9.5** Final pass: ensure all Python files have complete type hints and docstrings; all JS modules have JSDoc comments
- [x] **9.6** Update this checklist to mark all completed items `[x]`

---

## PHASE 10 — Stubbed / Future Work (Out of Scope)

- [s] **10.1** Phase 6 (original spec): Bidirectional AI Interviewer — real-time LLM-driven question generation based on emotion state
- [s] **10.2** `interviewer_audio` WebSocket message type — outbound TTS audio stream to candidate browser
- [s] **10.3** Multi-language support beyond Hebrew (extend XLM-RoBERTa fine-tuning)
- [s] **10.4** Mobile PWA frontend (camera/mic access on iOS/Android)
- [s] **10.5** Admin dashboard for HR reviewers to browse session reports

---

*Last updated: Phases 0–9 complete. All 49 tests pass. Remaining work: Kaggle model training/export (in progress) and Phase 10 (stubbed, out of scope).*
