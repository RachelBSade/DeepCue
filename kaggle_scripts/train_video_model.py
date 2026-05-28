"""
Phase 6.1 — Video Model Training (Kaggle GPU)

EfficientNet-B0 backbone + LSTM head trained on RAVDESS facial frames.

Dataset expected layout (Kaggle dataset: "uwrfkaggler/ravdess-emotional-speech-video"):
    /kaggle/input/ravdess-emotional-speech-video/
        Actor_01/
            01-01-01-01-01-01-01.mp4   (filename encodes 8 fields, field 3 = emotion)
            ...
        Actor_02/
            ...

RAVDESS emotion codes → DeepCue 8 classes mapping:
    01 neutral   → neutral
    02 calm      → neutral
    03 happy     → happy
    04 sad        → sad
    05 angry      → angry
    06 fearful   → anxious
    07 disgust   → uncertain
    08 surprised → surprised

Export: EfficientNet feature extractor + LSTM head → single ONNX model,
        then INT8 dynamic quantization.

Output artifacts:
    /kaggle/working/efficientnet_lstm.pt         (PyTorch checkpoint)
    /kaggle/working/efficientnet_lstm.onnx       (full precision ONNX)
    /kaggle/working/efficientnet_lstm_quant.onnx (INT8 quantized — copy to models/video/)
"""
from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score
import timm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAVDESS_ROOT   = Path("/kaggle/input/ravdess-emotional-speech-video")
OUTPUT_DIR     = Path("/kaggle/working")
BATCH_SIZE     = 16
EPOCHS         = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
WINDOW_SIZE    = 8        # frames per LSTM sequence
IMAGE_SIZE     = 224
NUM_CLASSES    = 8
LSTM_HIDDEN    = 256
LSTM_LAYERS    = 2
DROPOUT        = 0.3
FRAMES_PER_VID = 32       # sample this many frames uniformly from each video
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42

EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]

# RAVDESS emotion code (1-indexed) → DeepCue class index
_RAVDESS_MAP: dict[int, int] = {
    1: 0,  # neutral   → neutral
    2: 0,  # calm      → neutral
    3: 3,  # happy     → happy
    4: 4,  # sad       → sad
    5: 5,  # angry     → angry
    6: 2,  # fearful   → anxious
    7: 7,  # disgust   → uncertain
    8: 6,  # surprised → surprised
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RAVDESSVideoDataset(Dataset):
    """Loads short frame sequences from RAVDESS mp4 files."""

    def __init__(self, root: Path, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.samples: list[tuple[Path, int]] = []
        for mp4 in sorted(root.rglob("*.mp4")):
            label = _ravdess_label(mp4.stem)
            if label is not None:
                self.samples.append((mp4, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        frames = _load_frames(path, FRAMES_PER_VID)  # [N, 3, H, W]

        # Build one window by sampling evenly from extracted frames.
        if len(frames) >= self.window_size:
            indices = np.linspace(0, len(frames) - 1, self.window_size, dtype=int)
            window = frames[indices]
        else:
            # Pad by repeating last frame.
            pad = [frames[-1]] * (self.window_size - len(frames))
            window = np.concatenate([frames, np.stack(pad)], axis=0)

        tensor = torch.from_numpy(window).float()  # [window, 3, H, W]
        return tensor, label


def _ravdess_label(stem: str) -> int | None:
    """Extract DeepCue class index from RAVDESS filename stem (field 3 = emotion)."""
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        code = int(parts[2])
        return _RAVDESS_MAP.get(code)
    except ValueError:
        return None


def _load_frames(path: Path, n_frames: int) -> np.ndarray:
    """Sample n_frames uniformly from video, return [n, 3, H, W] float32 in [0,1]."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = set(np.linspace(0, total - 1, min(n_frames, total), dtype=int).tolist())

    frames: list[np.ndarray] = []
    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fno in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
        fno += 1
    cap.release()

    if not frames:
        frames = [np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EfficientNetLSTM(nn.Module):
    """EfficientNet-B0 per-frame feature extractor + bidirectional LSTM head."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        feat_dim: int = self.backbone.num_features  # 1280 for B0

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),  # *2 for bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, 3, H, W]
        Returns logits [B, num_classes].
        """
        B, T, C, H, W = x.shape
        # Extract per-frame features.
        feats = self.backbone(x.view(B * T, C, H, W))   # [B*T, feat_dim]
        feats = feats.view(B, T, -1)                     # [B, T, feat_dim]

        out, _ = self.lstm(feats)                        # [B, T, hidden*2]
        logits = self.classifier(out[:, -1, :])          # last timestep
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RAVDESSVideoDataset(RAVDESS_ROOT, window_size=WINDOW_SIZE)
    print(f"Dataset size: {len(dataset)} samples")

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = EfficientNetLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation.
        f1 = _evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            ckpt_path = OUTPUT_DIR / "efficientnet_lstm.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint (f1={best_f1:.4f})")

    print(f"\nTraining complete. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("WARNING: F1 below 0.50 threshold. Consider more data or tuning.")


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            preds = model(batch_x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# ONNX export (run after training — see export_and_quantize.py for INT8 step)
# ---------------------------------------------------------------------------

def export_onnx(ckpt_path: Path | None = None) -> Path:
    """Load best checkpoint and export to full-precision ONNX."""
    ckpt_path = ckpt_path or (OUTPUT_DIR / "efficientnet_lstm.pt")
    model = EfficientNetLSTM()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, WINDOW_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    onnx_path = OUTPUT_DIR / "efficientnet_lstm.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    export_onnx()
