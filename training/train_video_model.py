"""Phase 6.1 — EfficientNet-B0 + LSTM trained on RAVDESS facial frames → ONNX.
Filename fields: 3 = emotion code, 7 = actor ID. See _RAVDESS_MAP for the code mapping."""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score
import timm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAVDESS_ROOT   = Path("/kaggle/input/datasets/orvile/ravdess-dataset")
OUTPUT_DIR     = Path("/kaggle/working")
BATCH_SIZE     = 16
EPOCHS         = 5
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
WINDOW_SIZE    = 8        # frames per LSTM sequence
IMAGE_SIZE     = 224
NUM_CLASSES    = 8
LSTM_HIDDEN    = 256
LSTM_LAYERS    = 2
DROPOUT        = 0.3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42
VAL_ACTOR_FRACTION = 0.2  # fraction of RAVDESS actors held out for validation

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

    def __init__(self, root: Path, window_size: int = WINDOW_SIZE, is_train: bool = False) -> None:
        self.window_size = window_size
        self.is_train = is_train
        self.samples: list[tuple[Path, int]] = []
        self.actors: list[int] = []  # actor ID per sample, parallel to self.samples
        for mp4 in sorted(root.rglob("*.mp4")):
            label = _ravdess_label(mp4.stem)
            actor = _ravdess_actor(mp4.stem)
            if label is not None and actor is not None:
                self.samples.append((mp4, label))
                self.actors.append(actor)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        frames = _load_frames(path, self.window_size)  # [window_size, 3, H, W]
        tensor = torch.from_numpy(frames).float()

        # Synchronized horizontal flip — train only, applied identically to every frame
        # in the sequence so the temporal motion stays consistent.
        if self.is_train and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[3])

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


def _ravdess_actor(stem: str) -> int | None:
    """Extract actor ID from RAVDESS filename stem (field 7)."""
    parts = stem.split("-")
    if len(parts) < 7:
        return None
    try:
        return int(parts[6])
    except ValueError:
        return None


def _actor_disjoint_split(
    dataset: RAVDESSVideoDataset,
    val_fraction: float = VAL_ACTOR_FRACTION,
) -> tuple[Subset, Subset]:
    """Split by actor ID so no actor appears in both train and val — a per-sample random
    split would let the model learn faces instead of emotions, inflating F1.

    Returns Subsets wrapping two separate deepcopy'd dataset instances (not the same
    shared instance) so train.dataset.is_train can be flipped on without also turning on
    augmentation for val — Subset.dataset would otherwise be the same object for both."""
    rng = np.random.default_rng(SEED)
    unique_actors = sorted(set(dataset.actors))
    rng.shuffle(unique_actors)
    n_val = max(1, round(val_fraction * len(unique_actors)))
    val_actors = set(unique_actors[:n_val])

    train_idx = [i for i, a in enumerate(dataset.actors) if a not in val_actors]
    val_idx   = [i for i, a in enumerate(dataset.actors) if a in val_actors]
    train_dataset = copy.deepcopy(dataset)
    val_dataset   = copy.deepcopy(dataset)
    return Subset(train_dataset, train_idx), Subset(val_dataset, val_idx)


def _load_frames(path: Path, n_frames: int) -> np.ndarray:
    """Sample n_frames uniformly from video, return [n, 3, H, W] float32 in [0,1]."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = set(np.linspace(0, total - 1, min(n_frames, total), dtype=int).tolist())

    frames: list[np.ndarray] = []
    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= n_frames:
            break
        if fno in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
        fno += 1
    cap.release()

    if not frames:
        frames = [np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)]
    while len(frames) < n_frames:
        frames.append(frames[-1])
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
        # Mean pool over time instead of taking the last timestep: out[:, -1, :] gives the
        # forward direction full context (good) but the backward direction only 1 frame of
        # context (it's barely started), since the backward LSTM runs in reverse.
        pooled = out.mean(dim=1)                          # [B, hidden*2]
        logits = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RAVDESSVideoDataset(RAVDESS_ROOT, window_size=WINDOW_SIZE)
    print(f"Dataset size: {len(dataset)} samples, {len(set(dataset.actors))} actors")

    train_ds, val_ds = _actor_disjoint_split(dataset)
    train_ds.dataset.is_train = True  # safe: _actor_disjoint_split deepcopies per split
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples (actor-disjoint split)")

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
        dynamo=False,  # force legacy TorchScript exporter; avoids needing onnxscript
    )
    print(f"ONNX model exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    # export_onnx() runs separately, after you Save Version, in its own cell:
    #   from train_video_model import export_onnx
    #   export_onnx()
    print("=" * 50 + " TRAINING COMPLETE — SAVE VERSION NOW " + "=" * 50)
