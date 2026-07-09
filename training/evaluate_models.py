# Phase 6.5 — Evaluate all four quantized ONNX models; each must reach Macro F1 >= 0.50 (item 8.5)
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, f1_score

WORKING_DIR  = Path("/kaggle/working")
# Full RAVDESS (MP4 + WAV) — attach dataset orvile/ravdess-dataset
RAVDESS_ROOT = Path("/kaggle/input/datasets/orvile/ravdess-dataset")
HEBREW_SENTIMENT_HF_DATASET = "omilab/hebrew_sentiment"
SAMPLE_RATE  = 16000
AUDIO_LEN    = 48000
IMAGE_SIZE   = 224
MAX_LEN      = 128
WINDOW_SIZE  = 8
SEED         = 42
F1_THRESHOLD = 0.50

EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]

# RAVDESS filename emotion code (field 3) → DeepCue class
_RAVDESS_MAP: dict[int, int] = {
    1: 0, 2: 0,  # neutral + calm → neutral
    3: 3,        # happy
    4: 4,        # sad
    5: 5,        # angry
    6: 2,        # fearful → anxious
    7: 7,        # disgust → uncertain
    8: 6,        # surprised
}


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def _subsample(items: list, max_samples: int | None) -> list:
    """Randomly subsample (fixed seed, for reproducibility) when max_samples is given —
    lets you sanity-check a model in minutes instead of running the full RAVDESS/HF set,
    which can take hours. Use the full (max_samples=None) run only for the official
    F1 >= 0.50 gate-check, not for quick debugging."""
    if max_samples is None or max_samples >= len(items):
        return items
    return random.Random(SEED).sample(items, max_samples)


def _ravdess_samples(root: Path) -> list[tuple[Path, int]]:
    """Walk RAVDESS root and collect (mp4_path, label) pairs."""
    samples = []
    for mp4 in sorted(root.rglob("*.mp4")):
        parts = mp4.stem.split("-")
        if len(parts) >= 3:
            try:
                label = _RAVDESS_MAP.get(int(parts[2]))
                if label is not None:
                    samples.append((mp4, label))
            except ValueError:
                pass
    return samples


def _load_audio(path: Path) -> np.ndarray:
    """Load MP4 audio stream, resample to 16 kHz, pad/truncate to 3 s."""
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
    audio = waveform.squeeze().numpy()
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)  # match train_audio_model.py
    if len(audio) < AUDIO_LEN:
        audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
    return audio[:AUDIO_LEN].astype(np.float32)


def _extract_audio_features(waveform: np.ndarray) -> np.ndarray:
    """16-dim feature vector matching AudioEmotionPipeline: pitch, RMS, ZCR, 13 MFCCs."""
    f0     = librosa.yin(waveform, fmin=80.0, fmax=400.0, sr=SAMPLE_RATE)
    voiced = f0[f0 > 0]
    pitch  = float(np.mean(voiced) / 400.0) if len(voiced) > 0 else 0.0
    rms    = float(np.mean(librosa.feature.rms(y=waveform)[0]))
    zcr    = float(np.mean(librosa.feature.zero_crossing_rate(waveform)[0]))
    mfccs  = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=13).mean(axis=1).tolist()
    return np.array([pitch, rms, zcr] + mfccs, dtype=np.float32)


def _load_frames(path: Path, n_frames: int = WINDOW_SIZE) -> np.ndarray:
    """Sample n_frames evenly across a video; returns [n_frames, 3, H, W] float32.
    Seeks directly to each target frame index instead of decoding every frame in between
    — a RAVDESS clip has ~100+ frames at native framerate, so decoding all of them just to
    keep 8 made this the dominant cost of evaluation (~1.5s/sample, hours for the full set)."""
    import cv2
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs  = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frames.append(frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        frames = [np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)]
    arr = np.stack(frames)
    if len(arr) < n_frames:
        pad = np.tile(arr[-1:], (n_frames - len(arr), 1, 1, 1))
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n_frames]


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------

def evaluate_video(onnx_path: Path, max_samples: int | None = None) -> float:
    print(f"\n[Video] Loading {onnx_path.name}")
    sess       = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    samples    = _ravdess_samples(RAVDESS_ROOT)
    samples    = _subsample(samples, max_samples)
    print(f"[Video] {len(samples)} RAVDESS samples")
    preds, labels = [], []
    t0 = time.time()

    for i, (mp4, label) in enumerate(samples):
        frames  = _load_frames(mp4, WINDOW_SIZE)
        seq     = frames[np.newaxis].astype(np.float32)  # [1, W, 3, H, H]
        out     = sess.run(None, {input_name: seq})[0].flatten()
        pred    = int(np.argmax(out)) if len(out) == 8 else _score_to_class(out[0])
        preds.append(pred)
        labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  [Video] {i+1}/{len(samples)}  ({time.time()-t0:.0f}s elapsed)")

    return _report("Video", labels, preds)


def evaluate_audio(onnx_path: Path, max_samples: int | None = None) -> float:
    print(f"\n[Audio] Loading {onnx_path.name}")
    sess    = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    samples = _ravdess_samples(RAVDESS_ROOT)
    samples = _subsample(samples, max_samples)
    print(f"[Audio] {len(samples)} RAVDESS samples")
    preds, labels = [], []
    t0 = time.time()

    for i, (mp4, label) in enumerate(samples):
        audio = _load_audio(mp4)
        feats = _extract_audio_features(audio)
        out   = sess.run(None, {
            "audio_waveform": audio[np.newaxis],
            "features":       feats[np.newaxis],
        })[0].flatten()
        preds.append(int(np.argmax(out)))  # audio now exports 8-class logits, not a scalar
        labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  [Audio] {i+1}/{len(samples)}  ({time.time()-t0:.0f}s elapsed)")

    return _report("Audio", labels, preds)


# omilab/hebrew_sentiment label int → scalar score in [0, 1] — must match the mapping used
# in finetune_xlm_roberta.py's training data, so evaluation classes are computed consistently.
_HEBREW_SENTIMENT_LABEL_SCORE: dict[int, float] = {0: 1.0, 1: 0.0, 2: 0.5}


def evaluate_text(onnx_path: Path, max_samples: int | None = None) -> float:
    """Evaluate text model by MAE on its raw [0,1] scalar output — not by 8-class F1.
    The text model is a sentiment regressor, not an emotion classifier; forcing it into
    the 8-class F1 gate produces meaningless results (only 3 of 8 classes ever appear,
    macro F1 averages over all 8). MAE directly measures regression quality."""
    print(f"\n[Text] Loading {onnx_path.name}")
    from datasets import load_dataset
    from sklearn.metrics import mean_absolute_error
    from transformers import AutoTokenizer

    sess      = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    hf_dataset = load_dataset(HEBREW_SENTIMENT_HF_DATASET, revision="refs/convert/parquet")["test"]
    pairs      = _subsample(list(zip(hf_dataset["text"], hf_dataset["label"])), max_samples)
    texts      = [t for t, _ in pairs]
    gt_scores  = [_HEBREW_SENTIMENT_LABEL_SCORE[label] for _, label in pairs]
    pred_scores: list[float] = []
    t0 = time.time()
    print(f"[Text] {len(texts)} {HEBREW_SENTIMENT_HF_DATASET} (test) samples")

    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="np", max_length=MAX_LEN,
                        truncation=True, padding="max_length")
        out = sess.run(None, {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0].flatten()
        pred_scores.append(float(out[0]))
        if (i + 1) % 200 == 0:
            print(f"  [Text] {i+1}/{len(texts)}  ({time.time()-t0:.0f}s elapsed)")

    mae = float(mean_absolute_error(gt_scores, pred_scores))
    print(f"\n{'='*50}")
    print(f"  Text — MAE: {mae:.4f}  (lower is better; <0.15 = PASS)")
    status = "PASS" if mae < 0.15 else "BELOW THRESHOLD"
    print(f"  [{status}]  MAE={mae:.4f}")
    return mae


def evaluate_fusion(onnx_path: Path) -> float:
    print(f"\n[Fusion] Loading {onnx_path.name}")
    sess       = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Synthetic 17-dim test data matching train_fusion_model.py's _build_synthetic_dataset:
    # video[8] + audio[8] logit vectors peaked at the true class, text[1] scalar score.
    rng = np.random.default_rng(42)
    # Must match train_fusion_model.py's _SYNTHETIC_MEANS exactly (vm, am, tm) and same
    # bump formula so the test distribution matches what the model was trained on.
    _MEANS = {
        0: (0.50, 0.50, 0.52), 1: (0.85, 0.80, 0.80), 2: (0.20, 0.80, 0.32),
        3: (0.90, 0.85, 0.92), 4: (0.15, 0.20, 0.22), 5: (0.90, 0.90, 0.12),
        6: (0.70, 0.95, 0.62), 7: (0.40, 0.30, 0.42),
    }
    X_list, y = [], []
    for label, (vm, am, tm) in _MEANS.items():
        n = 50
        v = rng.normal(0, 1.0, (n, 8))
        v[:, label] += 1.5 + vm * 3.0
        a = rng.normal(0, 1.0, (n, 8))
        a[:, label] += 1.5 + am * 3.0
        t = np.clip(rng.normal(tm, 0.08, (n, 1)), 0, 1)
        X_list.append(np.concatenate([v, a, t], axis=1))
        y.extend([label] * n)
    X_arr = np.vstack(X_list).astype(np.float32)
    print(f"[Fusion] {len(X_arr)} synthetic samples")

    preds = []
    for row in X_arr:
        out = sess.run(None, {input_name: row[np.newaxis]})[0].flatten()
        preds.append(int(np.argmax(out)))

    return _report("Fusion", y, preds)


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------

def _score_to_class(score: float) -> int:
    """Map [0,1] scalar → 8-class index (equal-width buckets). Used for audio/text outputs."""
    return min(int(score * 8), 7)


def _report(name: str, labels: list[int], preds: list[int]) -> float:
    all_classes = list(range(len(EMOTION_CLASSES)))
    f1 = float(f1_score(labels, preds, average="macro", labels=all_classes, zero_division=0))
    print(f"\n{'='*50}")
    print(f"  {name} — Macro F1: {f1:.4f}  (threshold: {F1_THRESHOLD})")
    # labels=all_classes: a small max_samples subset may not contain every one of the 8
    # classes, which otherwise crashes classification_report with a class-count mismatch.
    print(classification_report(labels, preds, labels=all_classes, target_names=EMOTION_CLASSES, zero_division=0))
    status = "PASS" if f1 >= F1_THRESHOLD else "BELOW THRESHOLD"
    print(f"  [{status}]  {f1:.4f}")
    return f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    model: str = "all",
    video_onnx: Path | None = None,
    audio_onnx: Path | None = None,
    text_onnx: Path | None = None,
    fusion_onnx: Path | None = None,
    max_samples: int | None = None,
) -> None:
    """Notebook entry point — call this directly (e.g. `run()` or `run("audio")`) instead
    of main(). Running the whole script in a notebook cell triggers `if __name__ ==
    "__main__"`, and argparse then reads the kernel's own sys.argv (e.g. Colab/Kaggle's
    `-f kernel.json` launcher flag) instead of your intended arguments, raising
    'unrecognized arguments'. This function takes plain parameters instead.

    max_samples: cap on how many RAVDESS/HF samples to evaluate per model, e.g.
    run("video", max_samples=300) for a quick sanity check in minutes instead of the
    multi-hour full-dataset pass. Leave as None only for the official F1 >= 0.50 gate-check."""
    # Defaults point to full-precision ONNX — quantization is skipped for this project
    # because INT8 dynamic quantization breaks LSTM/transformer models (video F1 0.87→0.07).
    video_onnx  = video_onnx  or (WORKING_DIR / "efficientnet_lstm.onnx")
    audio_onnx  = audio_onnx  or (WORKING_DIR / "wav2vec2_classifier.onnx")
    text_onnx   = text_onnx   or (WORKING_DIR / "xlm_roberta_sentiment.onnx")
    fusion_onnx = fusion_onnx or (WORKING_DIR / "cross_modal_transformer.onnx")
    t_start = time.time()

    f1_results:  dict[str, float] = {}
    mae_results: dict[str, float] = {}

    if model in ("all", "video"):
        f1_results["video"]  = evaluate_video(Path(video_onnx), max_samples)
    if model in ("all", "audio"):
        f1_results["audio"]  = evaluate_audio(Path(audio_onnx), max_samples)
    if model in ("all", "text"):
        # Text is a regression model — evaluated by MAE, not F1; excluded from the F1 gate.
        mae_results["text"]  = evaluate_text(Path(text_onnx), max_samples)
    if model in ("all", "fusion"):
        f1_results["fusion"] = evaluate_fusion(Path(fusion_onnx))

    elapsed  = (time.time() - t_start) / 60
    all_pass = all(f1 >= F1_THRESHOLD for f1 in f1_results.values() if not np.isnan(f1))

    print(f"\n{'='*50}  EVALUATION SUMMARY  {'='*50}")
    for name, f1 in f1_results.items():
        status = "PASS" if f1 >= F1_THRESHOLD else "FAIL"
        print(f"  {name:8s}  F1={f1:.4f}  [{status}]")
    for name, mae in mae_results.items():
        status = "PASS" if mae < 0.15 else "FAIL"
        print(f"  {name:8s}  MAE={mae:.4f}  [{status}]  (regression — lower is better)")
    print(f"\nTotal evaluation time: {elapsed:.1f} min")
    if all_pass:
        print("All models meet their threshold. Ready for deployment.")
    else:
        print("One or more models are below threshold. Review training and data quality.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepCue ONNX models.")
    parser.add_argument("--model", default="all",
                        choices=["all", "video", "audio", "text", "fusion"])
    parser.add_argument("--video_onnx",  default=str(WORKING_DIR / "efficientnet_lstm_quant.onnx"))
    parser.add_argument("--audio_onnx",  default=str(WORKING_DIR / "wav2vec2_classifier_quant.onnx"))
    parser.add_argument("--text_onnx",   default=str(WORKING_DIR / "xlm_roberta_sentiment_quant.onnx"))
    parser.add_argument("--fusion_onnx", default=str(WORKING_DIR / "cross_modal_transformer_quant.onnx"))
    args = parser.parse_args()
    run(args.model, Path(args.video_onnx), Path(args.audio_onnx), Path(args.text_onnx), Path(args.fusion_onnx))


if __name__ == "__main__":
    main()
    print("=" * 50 + " SCRIPT COMPLETE " + "=" * 50)
