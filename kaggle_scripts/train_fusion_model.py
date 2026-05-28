"""
Phase 6.4 — Fusion Model Training (Kaggle GPU)

Cross-modal Transformer + MLP head trained on feature vectors
derived from the three trained modality models.

This script assumes the three modality ONNX models have already been exported
(see 6.1–6.3). It re-runs those models on the RAVDESS/CMU-MOSI validation splits
to generate (video_score, audio_score, text_score, label) triplets, then trains
the fusion head on those triplets.

For simplicity when real modality scores aren't available yet, a synthetic
dataset is generated from the RAVDESS emotion labels with small Gaussian noise.
Set USE_SYNTHETIC=False after exporting all three modality models.

Architecture (matches FusionPipeline in Django backend):
    Input  : [B, 3]  — [video_score, audio_score, text_score]
    Encoder: 2-layer Transformer (d_model=128, nhead=4, dim_ff=256) applied
             to 3 learned token embeddings projected from the 3 scalars
    Head   : Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 8) → softmax
    Output : [B, 8] — class probabilities

Output artifacts:
    /kaggle/working/cross_modal_transformer.pt
    /kaggle/working/cross_modal_transformer.onnx
    /kaggle/working/cross_modal_transformer_quant.onnx  ← copy to models/fusion/
"""
from __future__ import annotations

import math
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
D_MODEL         = 128
NHEAD           = 4
DIM_FEEDFORWARD = 256
NUM_LAYERS      = 2
DROPOUT         = 0.3
NUM_CLASSES     = 8
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 42
USE_SYNTHETIC   = True   # set False once all three modality models are ready

EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]

# Mean score per emotion class used for synthetic data generation.
# Approximates what each modality model should output on RAVDESS.
_SYNTHETIC_MEANS: dict[int, tuple[float, float, float]] = {
    0: (0.5, 0.5, 0.5),   # neutral
    1: (0.7, 0.6, 0.65),  # confident
    2: (0.3, 0.4, 0.35),  # anxious
    3: (0.75, 0.7, 0.72), # happy
    4: (0.25, 0.3, 0.28), # sad
    5: (0.35, 0.45, 0.38),# angry
    6: (0.8, 0.65, 0.7),  # surprised
    7: (0.4, 0.5, 0.45),  # uncertain
}
_SAMPLES_PER_CLASS = 200


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_synthetic_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (video, audio, text) score triplets with Gaussian noise per class."""
    rng = np.random.default_rng(SEED)
    X_list, y_list = [], []
    for label, (vm, am, tm) in _SYNTHETIC_MEANS.items():
        n = _SAMPLES_PER_CLASS
        video = np.clip(rng.normal(vm, 0.10, n), 0.0, 1.0)
        audio = np.clip(rng.normal(am, 0.12, n), 0.0, 1.0)
        text  = np.clip(rng.normal(tm, 0.11, n), 0.0, 1.0)
        X_list.append(np.stack([video, audio, text], axis=1))
        y_list.extend([label] * n)
    X = torch.from_numpy(np.vstack(X_list).astype(np.float32))
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


def _build_real_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load pre-computed modality scores from CSV.

    Expected file: /kaggle/working/modality_scores.csv
    Columns: video_score, audio_score, text_score, label
    Generate this file by running inference with all three modality ONNX
    models on the RAVDESS and CMU-MOSI datasets.
    """
    import pandas as pd
    csv_path = OUTPUT_DIR / "modality_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run modality models first or set USE_SYNTHETIC=True."
        )
    df = pd.read_csv(csv_path)
    X = torch.from_numpy(
        df[["video_score", "audio_score", "text_score"]].values.astype(np.float32)
    )
    y = torch.from_numpy(df["label"].values.astype(np.int64))
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModalityProjection(nn.Module):
    """Project a single scalar score to a D_MODEL-dim token embedding."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1] → [B, 1, D_MODEL]"""
        return self.proj(x.unsqueeze(-1))


class CrossModalTransformer(nn.Module):
    """
    Fuse 3 scalar modality scores via Transformer self-attention.

    Input  : [B, 3] — (video_score, audio_score, text_score)
    Output : [B, NUM_CLASSES] — logits
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
        self.video_proj = ModalityProjection(d_model)
        self.audio_proj = ModalityProjection(d_model)
        self.text_proj  = ModalityProjection(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3] — three modality scores."""
        v = self.video_proj(x[:, 0:1])  # [B, 1, D]
        a = self.audio_proj(x[:, 1:2])  # [B, 1, D]
        t = self.text_proj(x[:, 2:3])   # [B, 1, D]

        tokens = torch.cat([v, a, t], dim=1)   # [B, 3, D]
        encoded = self.encoder(tokens)          # [B, 3, D]
        pooled  = encoded.mean(dim=1)           # [B, D]
        return self.head(pooled)                # [B, num_classes]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = _build_synthetic_dataset() if USE_SYNTHETIC else _build_real_dataset()
    print(f"Dataset: {len(X)} samples, USE_SYNTHETIC={USE_SYNTHETIC}")

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
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        f1 = _evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}  val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining complete. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("WARNING: F1 below 0.50 threshold.")


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
    ckpt_path = ckpt_path or (OUTPUT_DIR / "cross_modal_transformer.pt")

    model = CrossModalTransformer()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3)
    onnx_path = OUTPUT_DIR / "cross_modal_transformer.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    export_onnx()
