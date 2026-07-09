# DeepCue — Kaggle Training Scripts

All model training and export scripts run on **Kaggle GPUs** (T4/P100).
The Django backend uses only the final quantized ONNX files; no GPU required at inference time.

---

## Prerequisites

### Kaggle datasets to attach to each notebook

| Script | Datasets to add |
|---|---|
| `train_video_model.py` | `uwrfkaggler/ravdess-emotional-speech-video` |
| `train_audio_model.py` | `uwrfkaggler/ravdess-emotional-speech-video` |
| `finetune_xlm_roberta.py` | `uwrfkaggler/cmu-mosi` · (optional) Hebrew sentiment CSV |
| `train_fusion_model.py` | None (uses pre-computed scores or synthetic data) |
| `evaluate_models.py` | Same as training scripts for the model(s) being tested |
| `export_and_quantize.py` | None (reads from `/kaggle/working/`) |

### Accelerator
Set **GPU → T4 x2** or **P100** in Kaggle notebook settings.

---

## Execution order

Run the scripts in this order. Each depends on the artifacts from the previous step.

```
1. train_video_model.py       → efficientnet_lstm.pt + .onnx
2. train_audio_model.py       → wav2vec2_classifier.pt + .onnx
3. finetune_xlm_roberta.py    → xlm_roberta_sentiment.pt + .onnx
4. train_fusion_model.py      → cross_modal_transformer.pt + .onnx
5. export_and_quantize.py     → *_quant.onnx  (all four models)
6. evaluate_models.py         → Macro F1 report (must be >= 0.50 per model)
```

---

## Running on Kaggle

Each script can be run as a standalone Kaggle notebook cell:

```python
# In a Kaggle notebook, run any script with:
%run /kaggle/working/train_video_model.py
```

Or copy the script contents directly into notebook cells.

---

## Output artifacts

All artifacts are saved to `/kaggle/working/`. Download them from the Kaggle notebook
output panel after each run.

| File | Size (approx.) | Destination |
|---|---|---|
| `efficientnet_lstm_quant.onnx` | ~15 MB | `models/video/efficientnet_lstm.onnx` |
| `wav2vec2_classifier_quant.onnx` | ~45 MB | `models/audio/wav2vec2_classifier.onnx` |
| `xlm_roberta_sentiment_quant.onnx` | ~80 MB | `models/text/xlm_roberta_sentiment.onnx` |
| `cross_modal_transformer_quant.onnx` | ~1 MB | `models/fusion/cross_modal_transformer.onnx` |

---

## Environment variables to update after downloading

Edit your `.env` file:

```env
VIDEO_MODEL_PATH=models/video/efficientnet_lstm.onnx
AUDIO_MODEL_PATH=models/audio/wav2vec2_classifier.onnx
TEXT_MODEL_PATH=models/text/xlm_roberta_sentiment.onnx
FUSION_MODEL_PATH=models/fusion/cross_modal_transformer.onnx
```

---

## Dataset notes

### RAVDESS
- 24 actors, 1440 audio files + 2452 video files
- 8 emotion classes (mapped to DeepCue's 8 classes — see script headers)
- Kaggle dataset slug: `uwrfkaggler/ravdess-emotional-speech-video`

### CMU-MOSI
- 2199 video clips with sentiment annotations in [-3, 3]
- Normalised to [0, 1] for regression training
- Find on Kaggle by searching "CMU-MOSI"

### Hebrew sentiment (optional)
- Upload your own CSV to Kaggle as a private dataset
- Required columns: `text` (str), `sentiment` (float in [-1, 1])
- Increases model accuracy on Hebrew interview audio

---

## Fusion model — synthetic vs real data

`train_fusion_model.py` has a `USE_SYNTHETIC = True` flag at the top.

- **`True`** (default): trains on Gaussian synthetic data. Fast, always works, lower accuracy.
- **`False`**: trains on real modality scores. Requires running all three modality models
  on RAVDESS/CMU-MOSI first and saving predictions to `/kaggle/working/modality_scores.csv`
  with columns `video_score, audio_score, text_score, label`.

For the best results, set `USE_SYNTHETIC = False` after completing Steps 1–3.

---

## Performance targets

| Model | Target Macro F1 | Dataset |
|---|---|---|
| Video (EfficientNet-B0 + LSTM) | ≥ 0.50 | RAVDESS video |
| Audio (wav2vec2) | ≥ 0.50 | RAVDESS audio |
| Text (XLM-RoBERTa) | ≥ 0.50 | CMU-MOSI |
| Fusion (Transformer) | ≥ 0.50 | Combined |

These are evaluated by `evaluate_models.py` and correspond to checklist item 8.5.
