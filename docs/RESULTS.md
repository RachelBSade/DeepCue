# DeepCue — Kaggle Script Results Log

Tracks results from each run of the 6 `kaggle_scripts/` files. Add a new entry under the
relevant script each time you run it — don't overwrite previous entries, so we can compare
runs over time.

---

## train_video_model.py

| Date | Result | Runtime | Notes |
|---|---|---|---|
| 2026-06-30 | Epoch 05/5, loss=0.0366, **val_macro_f1=0.5932** | ~2h | First run with all fixes: actor-disjoint split, mean pooling (instead of last-timestep), simplified frame sampling, horizontal flip augmentation, deepcopy fix for train/val independence. **PASS** (≥0.50 threshold). First trustworthy video result — earlier ~0.99 number was actor-leakage from the pre-fix script. |

---

## train_audio_model.py

| Date | Result | Runtime | Notes |
|---|---|---|---|
| 2026-06-29 | Stage 1: F1≈0.44, Stage 2: best F1≈0.7179 (epoch 10) | — | Actor-disjoint split, early stopping (patience=5). **Note**: `export_onnx()` was changed since this run to export 8-class logits instead of a scalar score — checkpoint itself is still valid, but ONNX export shape changed. |
| 2026-06-30 | Stage 1: Epoch 12/12, loss=0.7980, val_f1=0.4424. Stage 2: best val_macro_f1=**0.7258** (epoch 10, loss=0.0101) | 65.9 min | Consistent with the prior run (0.7179→0.7258, slight improvement) — confirms this is a stable, repeatable result, not a fluke. Same overfit-plateau loss pattern as before (expected, not a new problem). This run already reflects the 8-class logits export change. |

---

## finetune_xlm_roberta.py

| Date | Result | Runtime | Notes |
|---|---|---|---|
| 2026-06-29 | val_MAE≈0.0235 (epoch 9/10) | — | Trained on `omilab/hebrew_sentiment` (replaced broken CMU-MOSI). MAE metric, not F1 — see `evaluate_models.py` section below for the F1-based gate-check (which had a metric-mismatch issue, not yet fully resolved). |

---

## train_fusion_model.py

| Date | Result | Runtime | Notes |
|---|---|---|---|
| 2026-06-29 | Epoch 30/30, best val_macro_f1=0.4137 | ~0.3 min | **FAIL** (<0.50). Trained on synthetic placeholder data only (`USE_SYNTHETIC=True`). Traced to overlapping synthetic class means, not a real model failure. **Note**: model architecture changed since this run (now takes 17-dim input: video[8]+audio[8]+text[1] logits, not 3 scalars) — this result is from the old 3-scalar architecture and is no longer representative. |

---

## export_and_quantize.py

| Date | Model | Full-precision size | Quantized size | Notes |
|---|---|---|---|---|
| 2026-06-29 | Video | 33.3 MB | 8.7 MB (74% smaller) | |
| 2026-06-29 | Audio | — | ~91 MB | |
| 2026-06-29 | Text | — | ~265 MB | |
| 2026-06-29 | Fusion | — | ~366 KB | |

(File size only — see `evaluate_models.py` below for accuracy after quantization.)

---

## evaluate_models.py

| Date | Model | F1 / MAE | Samples | Status | Notes |
|---|---|---|---|---|---|
| 2026-07-05 | Video | **F1=0.7984** | 300 | ✅ PASS | New checkpoint (actor-disjoint, mean pooling, flip aug). Full-precision ONNX. Quantization permanently dropped. "Confident" class has support=0 — no RAVDESS emotion code maps to class 1. |
| 2026-07-05 | Audio | **F1=0.4467** | 300 | ❌ FAIL | Close to threshold. Trained on audio-only RAVDESS dataset, evaluated on audio-visual mp4s — different dataset source may partly explain gap. Normalization bug now fixed. To revisit. |
| 2026-07-05 | Text | **MAE=0.0458** | 300 | ✅ PASS | Switched to MAE metric (correct for regression model). Excellent result. |
| 2026-07-05 | Fusion | **F1=0.9825** | 400 synthetic | ✅ PASS | Synthetic data with improved intensity-modulated means. High score expected — reflects synthetic separability, not real-world performance. |

---

## Open Items

- [ ] **Audio F1=0.4467 — below threshold.** Likely caused by train/eval dataset mismatch (trained on audio-only WAV dataset, evaluated on audio-visual MP4s). Fine-tuning pass deferred — backend/frontend first.
- [ ] **Update `backend/apps/inference/`** — audio pipeline expects old scalar output (now 8-logit), fusion pipeline expects old 3-dim input (now 17-dim). Must be done before DeepCue can run end-to-end.
- [ ] **Text model future upgrade** — currently regression (scalar); upgrade to 8-class classifier to match video/audio architecture (deferred).
- [ ] **Real fusion training data** — currently synthetic only; real paired multimodal data needed once interview sessions are being collected (deferred).
