# Phase 6.4 — Cross-modal Transformer trained on (video, audio, text) score triplets → 8-class ONNX
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("/kaggle/working")
BATCH_SIZE      = 64
EPOCHS          = 30
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
D_MODEL         = 128   # transformer token dimension
NHEAD           = 4
DIM_FEEDFORWARD = 256
NUM_LAYERS      = 2
DROPOUT         = 0.3
NUM_CLASSES     = 8
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 42
USE_SYNTHETIC   = True  # set False once all three modality ONNX models are ready

EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]

# Per-emotion synthetic signal profile: (vm, am, tm) where:
#   vm = how clearly the emotion shows in video  [0=subtle, 1=very obvious]
#   am = how clearly the emotion shows in audio  [0=subtle, 1=very obvious]
#   tm = expected text sentiment score           [0=very negative, 1=very positive]
# vm/am are used to scale the class-peak bump on the logit vectors — stronger emotion
# expression gives a larger peak → more confident logit vector → easier to classify.
# tm values are spaced with ≥0.10 gap between neighbours (noise std=0.08) to avoid overlap.
_SYNTHETIC_MEANS: dict[int, tuple[float, float, float]] = {
    0: (0.50, 0.50, 0.52),  # neutral   — moderate in all channels
    1: (0.85, 0.80, 0.80),  # confident — strong and clear in all channels
    2: (0.20, 0.80, 0.32),  # anxious   — subtle video but high audio (trembling/fast speech)
    3: (0.90, 0.85, 0.92),  # happy     — high energy and positive across all channels
    4: (0.15, 0.20, 0.22),  # sad       — low energy everywhere
    5: (0.90, 0.90, 0.12),  # angry     — explosive video/audio, very negative text
    6: (0.70, 0.95, 0.62),  # surprised — audio spike, moderate video/text
    7: (0.40, 0.30, 0.42),  # uncertain — subdued, slightly below neutral
}
_SAMPLES_PER_CLASS = 200  # synthetic samples per class


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_synthetic_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic 17-dim (video[8], audio[8], text[1]) triplets.
    vm/am modulate the class-peak bump strength so each emotion's signal is as strong/subtle
    in each modality as it would be in reality. tm sets the text sentiment centre per class."""
    rng = np.random.default_rng(SEED)
    X_list, y_list = [], []
    for label, (vm, am, tm) in _SYNTHETIC_MEANS.items():
        n = _SAMPLES_PER_CLASS
        video = rng.normal(0, 1.0, (n, NUM_CLASSES))
        video[:, label] += 1.5 + vm * 3.0   # bump ∈ [1.5, 4.5] — scales with visual clarity
        audio = rng.normal(0, 1.0, (n, NUM_CLASSES))
        audio[:, label] += 1.5 + am * 3.0   # bump ∈ [1.5, 4.5] — scales with audio clarity
        text  = np.clip(rng.normal(tm, 0.08, (n, 1)), 0.0, 1.0)
        X_list.append(np.concatenate([video, audio, text], axis=1))
        y_list.extend([label] * n)
    X = torch.from_numpy(np.vstack(X_list).astype(np.float32))
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


def _build_real_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load pre-computed modality scores from /kaggle/working/modality_scores.csv.
    Required columns: video_logit_0..7, audio_logit_0..7, text_score, label.
    Generate this file by running inference with all three modality ONNX models first.
    """
    import pandas as pd
    csv_path = OUTPUT_DIR / "modality_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Set USE_SYNTHETIC=True or generate scores first.")
    df = pd.read_csv(csv_path)
    v_cols = [f"video_logit_{i}" for i in range(NUM_CLASSES)]
    a_cols = [f"audio_logit_{i}" for i in range(NUM_CLASSES)]
    X  = torch.from_numpy(df[v_cols + a_cols + ["text_score"]].values.astype(np.float32))
    y  = torch.from_numpy(df["label"].values.astype(np.int64))
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModalityProjection(nn.Module):
    """Project a variable-sized modality input to a D_MODEL-dim token: [B,in_features] → [B,1,D]."""

    def __init__(self, in_features: int, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).unsqueeze(1)  # [B, 1, D_MODEL]


class CrossModalTransformer(nn.Module):
    """
    Fuse mixed-dimension modality inputs via Transformer self-attention.
    Input [B,17] = video[8] logits + audio[8] logits + text[1] score →
    three projected tokens → encoder → mean-pool → MLP → [B,8] logits.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        # One projection per modality so the model learns modality-specific embeddings.
        # Video/audio give 8-class logits (more resolution); text gives a single scalar.
        self.video_proj = ModalityProjection(in_features=num_classes, d_model=d_model)
        self.audio_proj = ModalityProjection(in_features=num_classes, d_model=d_model)
        self.text_proj  = ModalityProjection(in_features=1, d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 17] — (video_logits[8], audio_logits[8], text_score[1])."""
        v       = self.video_proj(x[:, 0:8])       # [B, 1, D]
        a       = self.audio_proj(x[:, 8:16])      # [B, 1, D]
        t       = self.text_proj(x[:, 16:17])      # [B, 1, D]
        tokens  = torch.cat([v, a, t], dim=1)     # [B, 3, D]
        encoded = self.encoder(tokens)             # [B, 3, D]
        pooled  = encoded.mean(dim=1)             # [B, D] — average across modality tokens
        return self.head(pooled)                  # [B, num_classes]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[Fusion] Device: {DEVICE}")
    print(f"[Fusion] USE_SYNTHETIC={USE_SYNTHETIC}")

    X, y = _build_synthetic_dataset() if USE_SYNTHETIC else _build_real_dataset()
    print(f"[Fusion] Dataset: {len(X)} samples across {NUM_CLASSES} classes")

    dataset  = TensorDataset(X, y)
    val_size = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = CrossModalTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1   = 0.0
    ckpt_path = OUTPUT_DIR / "cross_modal_transformer.pt"

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        f1      = _evaluate(model, val_loader)
        elapsed = time.time() - t0
        print(f"[Fusion] Epoch {epoch:02d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  "
              f"val_f1={f1:.4f}  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved (f1={best_f1:.4f})")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[Fusion] Training complete in {elapsed:.1f} min. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("[Fusion] WARNING: F1 below 0.50 threshold.")


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(batch_y.numpy())
    return float(f1_score(labels_all, preds_all, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(ckpt_path: Path | None = None) -> Path:
    print("\n[Fusion] Exporting to ONNX ...")
    ckpt_path = ckpt_path or (OUTPUT_DIR / "cross_modal_transformer.pt")

    model = CrossModalTransformer()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy     = torch.zeros(1, 17)  # video_logits[8] + audio_logits[8] + text_score[1]
    onnx_path = OUTPUT_DIR / "cross_modal_transformer.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # force legacy TorchScript exporter; avoids needing onnxscript
    )
    print(f"[Fusion] ONNX exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    # export_onnx() runs separately, after you Save Version, in its own cell:
    #   from train_fusion_model import export_onnx
    #   export_onnx()
    print("=" * 50 + " TRAINING COMPLETE — SAVE VERSION NOW " + "=" * 50)
