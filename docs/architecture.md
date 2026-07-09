# DeepCue — System Architecture

This document explains **why** DeepCue is built the way it is. Every major decision below traces back to one of three hard constraints. For a component-level tour, see the [repository map in the README](../README.md#repository-map).

## 1. The Constraints That Drove the Design

| # | Constraint | Consequence |
|---|---|---|
| C1 | **< 10 s end-to-end latency** — from captured signal to fused emotion on screen — on **consumer CPU-only hardware** (weak Windows laptop, no GPU at inference time) | ONNX Runtime instead of PyTorch at inference; asynchronous per-modality processing; client-side feature extraction |
| C2 | **Macro F1 ≥ 0.50 quality gate** per model on held-out benchmarks (RAVDESS for video/audio; sentiment corpora for text) | Every export is gate-checked before it ships; optimizations that fail the gate are rejected (see §4) |
| C3 | **A live interview session must never crash** — inference failures are invisible to the candidate | Universal `NEUTRAL_FALLBACK` degradation strategy (see §6) |

The single most important structural decision follows from C1: **training and inference are different programs on different machines.** Training, export, and quantization run on Kaggle GPUs (`training/`); the Django backend (`backend/`) never trains anything and never imports training code — it only loads pre-exported `.onnx` files whose paths come from `.env` (`VIDEO_MODEL_PATH`, `AUDIO_MODEL_PATH`, `TEXT_MODEL_PATH`, `FUSION_MODEL_PATH`). The ONNX files are the *entire contract* between the two worlds.

## 2. Meeting the Latency Budget (C1)

Three decisions, in order of impact:

**ONNX Runtime as the inference engine.** Exporting from PyTorch to ONNX buys CPU-optimized execution (graph fusion, MLAS kernels) and removes the training-framework overhead from the hot path. It also decouples the serving environment from the training stack — the backend installs `onnxruntime`, not the GPU training toolchain.

**Asynchronous, parallel modality processing via Celery.** The WebSocket consumer (`InterviewConsumer`) does **no inference inline** — it validates and rate-limits each inbound message, then dispatches to Celery and returns immediately. Video, audio, and text/fusion work run on **dedicated queues** (`video_queue`, `audio_queue`, `fusion_queue`) with separate worker processes, so the three modalities are processed in parallel rather than sequentially: total latency ≈ max(modality latencies) + fusion, not their sum. The dominant term is the text path (Whisper ASR → XLM-RoBERTa), which the budget was sized around. Dedicated queues also isolate backpressure — a slow Whisper transcription can't starve video inference.

**Feature extraction pushed to the client.** The browser runs MediaPipe Face Mesh and streams **468 facial landmarks per frame**, not raw video. This removes face detection/mesh fitting from the server's CPU budget, shrinks the WebSocket payload by orders of magnitude, and keeps raw video off the wire entirely (a privacy benefit as well). Audio is streamed in 3-second chunks — the unit of work for both the audio and text pipelines.

**Redis as the coordination fabric.** One Redis instance (separate DB indexes) serves as the Celery broker, the result backend, the Channels layer, and the **per-session modality score cache**. Per-modality scores are written to Redis keyed by `session_id`; fusion reads across modalities from that cache. Everything on the hot path is a Redis round-trip, not a database query.

## 3. Data Flow, End to End

```
Browser
  ├─ MediaPipe Face Mesh → 468-landmark frames ─┐
  └─ Web Audio API → 3 s audio chunks ──────────┤
                                                ▼
              WebSocket  ws/interview/<session_id>/
              (message shapes: backend/apps/sessions_app/protocol.py —
               the single source of truth for the wire protocol)
                                                ▼
  InterviewConsumer:  validation.py → rate_limit.py → Celery dispatch
        │
        ├─ video_queue  → video_pipeline  → EfficientNet-B0 + LSTM → 8 logits ─┐
        ├─ audio_queue  → audio_pipeline  → wav2vec 2.0 + librosa  → 8 logits ─┤→ Redis
        │                 text_pipeline   → Whisper → XLM-RoBERTa  → 1 score  ─┘  (per session)
        │
        └─ fusion_queue → fusion_pipeline reads Redis across modalities
                            → 17-dim vector → Cross-Modal Transformer
                            → emotion_result (8 classes)
                                 ├─→ browser, via the session's Channels group
                                 └─→ MongoDB (session document)

session_end → report task → report_generator.py → 5-section ReportLab PDF
              (Hebrew RTL) → GridFS
```

**The 17-dimensional fusion vector.** Fusion input concatenates the full output distributions of each modality: **8 video logits + 8 audio logits + 1 text sentiment score = 17 dimensions**, consumed by a Cross-Modal Transformer with an MLP classification head. Passing full logit vectors rather than argmax labels preserves each modality's uncertainty — the fusion model learns, for example, that a confident video "happy" should outweigh an ambiguous audio split, which a hard-label scheme would discard. (The scalar text input reflects the current sentiment-regression model; upgrading text to a matching 8-class head is on the roadmap and widens the vector to 24.)

## 4. Quality Gates (C2)

Every trained model must clear **macro F1 ≥ 0.50** on a held-out, **actor-disjoint** split before its export is accepted (`training/evaluate_models.py` is the gate-check; results are logged run-over-run in [RESULTS.md](RESULTS.md)). Two consequences of taking the gate seriously:

- **Actor-disjoint evaluation is mandatory.** An early video run scored ~0.99 — actor leakage across train/val splits, not real performance. The honest post-fix score (0.59 val / 0.80 eval) is what ships and what the log records.
- **Optimizations are subordinate to the gate.** INT8 dynamic quantization was applied per-model and *kept only where it survived the gate*. The video model shipped **full-precision** after quantization cost too much accuracy — a 74% size saving is worthless below the quality bar.

Current status: video and text pass (video macro F1 = 0.80; text MAE = 0.046 on its regression objective); the audio model sits just under the gate (0.45) on a cross-source evaluation and is scheduled for fine-tuning; fusion currently trains on synthetic data pending real paired session recordings — both tracked openly in [RESULTS.md](RESULTS.md).

## 5. Graceful Degradation: `NEUTRAL_FALLBACK` (C3)

A live interview cannot show a stack trace. The failure policy on the inference path is uniform:

**Every operation that can fail is wrapped, and on failure the pipeline emits `NEUTRAL_FALLBACK = 0.5` — a neutral score — instead of raising.**

Concretely:

- Each modality pipeline (`backend/apps/inference/`) wraps model loading, preprocessing, and inference in `try/except`; any failure yields the neutral fallback score for that chunk, logged server-side but invisible to the candidate.
- Fusion reads whatever scores exist in Redis; a modality that failed (or hasn't produced a score yet) contributes its fallback value, so a dead video pipeline degrades the session to effectively audio+text rather than ending it.
- The consumer's validation and rate-limiting layers reject malformed or flooding input *before* it reaches Celery, so bad client data cannot poison the pipelines.

The trade-off is deliberate: a temporarily wrong-but-neutral reading is strictly better than a dropped session, and persistent failures still surface in logs and in the final report's data density.

## 6. Operational Notes

- **Settings** are split across `deepcue_backend/settings/{base,local,production,test}.py`. Production logging switches to structured JSON on stdout keyed off the `DJANGO_DEBUG` env var (not Django's `DEBUG` setting), so logging is correct before settings overrides apply.
- **Testing without infrastructure:** the pytest suite runs under `settings.test` with an in-memory channel layer and `CELERY_TASK_ALWAYS_EAGER=True` — no Redis, MongoDB, or `.onnx` files needed. This is what makes the suite runnable in any CI container.
- **Windows specifics:** Redis is provided by Memurai (native service) because Docker Desktop's TCP proxy breaks the asyncio streams Channels/Daphne depend on; `redis==4.6.0` is pinned because 5.x sends `CLIENT SETINFO`, which breaks `channels-redis`; Celery workers run one per queue (a combined worker needs `--pool=solo`, as prefork requires `os.fork()`).
- **Extension point:** the WebSocket protocol reserves an `interviewer_audio` message type for the planned bidirectional AI interviewer; it is intentionally stubbed until that phase begins.
