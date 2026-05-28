"""
Phase 6.2 — Audio Model Training (Kaggle GPU)

wav2vec 2.0 fine-tuning on RAVDESS audio for emotion classification.

Dataset expected: same RAVDESS mp4 files as train_video_model.py.
Audio is extracted on-the-fly from mp4 using torchaudio.

Architecture:
    - facebook/wav2vec2-base loaded via HuggingFace transformers
    - Frozen feature extractor (CNN layers), fine-tune transformer blocks
    - Mean-pool hidden states → Linear(768, 64) → ReLU → Linear(64, 8 classes)
    - Second head: Linear(64 + 16_features, 1) → Sigmoid for scalar score
      (the scalar output [0,1] is what AudioEmotionPipeline expects)

Two-stage training:
    Stage 1 (5 epochs): freeze wav2vec2.feature_extractor, train transformer + head
    Stage 2 (10 epochs): unfreeze all, low LR

Export:
    Combined model (wav2vec encoder + classifier) → single ONNX with two inputs:
        "audio_waveform" : [1, 48000]
        "features"       : [1, 16]
    Output: [1] scalar score

Output artifacts:
    /kaggle/working/wav2vec2_classifier.pt
    /kaggle/working/wav2vec2_classifier.onnx
    /kaggle/working/wav2vec2_classifier_quant.onnx  ← copy to models/audio/
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAVDESS_ROOT  = Path("/kaggle/input/ravdess-emotional-speech-video")
OUTPUT_DIR    = Path("/kaggle/working")
SAMPLE_RATE   = 16000
AUDIO_LEN     = 48000          # 3 seconds at 16 kHz
BATCH_SIZE    = 8
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 10
LR_STAGE1     = 3e-4
LR_STAGE2     = 1e-5
NUM_CLASSES   = 8
N_MFCC        = 13
FEATURE_DIM   = 3 + N_MFCC    # pitch/RMS/ZCR + 13 MFCCs = 16
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42

_RAVDESS_MAP: dict[int, int] = {
    1: 0, 2: 0, 3: 3, 4: 4, 5: 5, 6: 2, 7: 7, 8: 6,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RAVDESSAudioDataset(Dataset):
    """Extract audio from RAVDESS mp4s, returning waveform + features + label."""

    def __init__(self, root: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        for mp4 in sorted(root.rglob("*.mp4")):
            parts = mp4.stem.split("-")
            if len(parts) >= 3:
                try:
                    code = int(parts[2])
                    label = _RAVDESS_MAP.get(code)
                    if label is not None:
                        self.samples.append((mp4, label))
                except ValueError:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        waveform = _load_audio(path)           # [48000,] float32
        features = _extract_features(waveform) # [16,]    float32
        return (
            torch.from_numpy(waveform).float(),
            torch.from_numpy(features).float(),
            label,
        )


def _load_audio(path: Path) -> np.ndarray:
    """Load first audio stream from mp4, resample to 16 kHz, pad/truncate to AUDIO_LEN."""
    waveform, sr = torchaudio.load(str(path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    audio = waveform.squeeze().numpy()
    if len(audio) < AUDIO_LEN:
        audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
    else:
        audio = audio[:AUDIO_LEN]
    return audio.astype(np.float32)


def _extract_features(waveform: np.ndarray) -> np.ndarray:
    """16-dim paralinguistic features (must match AudioEmotionPipeline._extract_features)."""
    import librosa

    f0 = librosa.yin(waveform, fmin=80.0, fmax=400.0, sr=SAMPLE_RATE)
    voiced = f0[f0 > 0]
    mean_pitch = float(np.mean(voiced) / 400.0) if len(voiced) > 0 else 0.0

    rms = librosa.feature.rms(y=waveform)[0]
    mean_rms = float(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    mean_zcr = float(np.mean(zcr))

    mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_means = mfccs.mean(axis=1).tolist()

    return np.array([mean_pitch, mean_rms, mean_zcr] + mfcc_means, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Wav2Vec2EmotionClassifier(nn.Module):
    """
    wav2vec2-base encoder with a dual-head classifier.

    Classification head : [8 classes] — used for training with cross-entropy
    Regression head     : [1 scalar]  — used for ONNX export / inference
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_dim = self.wav2vec2.config.hidden_size  # 768

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(64, num_classes)
        self.score_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def freeze_feature_extractor(self) -> None:
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        audio_waveform: torch.Tensor,  # [B, 48000]
        features: torch.Tensor,        # [B, 16]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class_logits [B,8], score [B,1])."""
        hidden = self.wav2vec2(audio_waveform).last_hidden_state  # [B, T, 768]
        pooled = hidden.mean(dim=1)                                # [B, 768]

        combined = torch.cat([pooled, features], dim=1)            # [B, 784]
        shared = self.shared(combined)                             # [B, 64]

        logits = self.class_head(shared)
        score  = self.score_head(shared)
        return logits, score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RAVDESSAudioDataset(RAVDESS_ROOT)
    print(f"Dataset size: {len(dataset)} samples")

    val_size = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = Wav2Vec2EmotionClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0

    # Stage 1: frozen feature extractor.
    print("--- Stage 1: fine-tuning transformer layers ---")
    model.freeze_feature_extractor()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
    )
    _run_epochs(model, train_loader, val_loader, criterion, optimizer, STAGE1_EPOCHS)

    # Stage 2: unfreeze all.
    print("--- Stage 2: full fine-tune ---")
    model.unfreeze_all()
    optimizer = optim.AdamW(model.parameters(), lr=LR_STAGE2)
    best_f1 = _run_epochs(model, train_loader, val_loader, criterion, optimizer, STAGE2_EPOCHS,
                          ckpt_path=OUTPUT_DIR / "wav2vec2_classifier.pt")

    print(f"\nTraining complete. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("WARNING: F1 below 0.50 threshold.")


def _run_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int,
    ckpt_path: Path | None = None,
) -> float:
    best_f1 = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for wav, feat, labels in train_loader:
            wav, feat, labels = wav.to(DEVICE), feat.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(wav, feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        f1 = _evaluate(model, val_loader)
        print(f"  Epoch {epoch:02d}  loss={total_loss / len(train_loader):.4f}  val_f1={f1:.4f}")

        if ckpt_path and f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
    return best_f1


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for wav, feat, labels in loader:
            wav, feat = wav.to(DEVICE), feat.to(DEVICE)
            logits, _ = model(wav, feat)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(ckpt_path: Path | None = None) -> Path:
    """Export the regression (scalar score) path to ONNX."""
    ckpt_path = ckpt_path or (OUTPUT_DIR / "wav2vec2_classifier.pt")

    model = Wav2Vec2EmotionClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    class _ScoreOnlyWrapper(nn.Module):
        """Wraps the classifier to return only the scalar score output."""
        def __init__(self, inner: Wav2Vec2EmotionClassifier) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, audio_waveform: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            _, score = self.inner(audio_waveform, features)
            return score  # [B, 1]

    wrapper = _ScoreOnlyWrapper(model)
    dummy_wav = torch.zeros(1, AUDIO_LEN)
    dummy_feat = torch.zeros(1, FEATURE_DIM)

    onnx_path = OUTPUT_DIR / "wav2vec2_classifier.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_wav, dummy_feat),
        str(onnx_path),
        input_names=["audio_waveform", "features"],
        output_names=["score"],
        dynamic_axes={
            "audio_waveform": {0: "batch"},
            "features":       {0: "batch"},
            "score":          {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    export_onnx()
