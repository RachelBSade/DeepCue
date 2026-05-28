"""
Phase 6.3 — XLM-RoBERTa Fine-tuning (Kaggle GPU)

Fine-tunes xlm-roberta-base on CMU-MOSI sentiment + Hebrew sentiment data
for emotion intensity regression (scalar output [0,1]).

Datasets expected:
    CMU-MOSI (via SDK or manual CSV):
        /kaggle/input/cmu-mosi/mosi_sentiment.csv
        Columns: text (str), sentiment (float in [-3, 3])

    Hebrew sentiment (optional, self-collected or public):
        /kaggle/input/hebrew-sentiment/hebrew_sentiment.csv
        Columns: text (str), sentiment (float in [-1, 1])

Both are normalised to [0,1]:
    score = (sentiment - min) / (max - min)

Architecture:
    xlm-roberta-base [CLS] token → Linear(768, 1) → Sigmoid

Output artifacts:
    /kaggle/working/xlm_roberta_sentiment.pt
    /kaggle/working/xlm_roberta_sentiment.onnx
    /kaggle/working/xlm_roberta_sentiment_quant.onnx  ← copy to models/text/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MOSI_CSV         = Path("/kaggle/input/cmu-mosi/mosi_sentiment.csv")
HEBREW_CSV       = Path("/kaggle/input/hebrew-sentiment/hebrew_sentiment.csv")
OUTPUT_DIR       = Path("/kaggle/working")
MODEL_NAME       = "xlm-roberta-base"
MAX_LEN          = 128
BATCH_SIZE       = 32
EPOCHS           = 10
LR               = 2e-5
WEIGHT_DECAY     = 1e-2
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED             = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Text regression dataset normalised to [0,1]."""

    def __init__(
        self,
        texts: list[str],
        scores: list[float],
        tokenizer: AutoTokenizer,
    ) -> None:
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.scores[idx], dtype=torch.float32),
        }


def _load_mosi(path: Path, tokenizer: AutoTokenizer) -> Dataset | None:
    if not path.exists():
        print(f"CMU-MOSI CSV not found at {path} — skipping.")
        return None
    df = pd.read_csv(path)
    texts  = df["text"].fillna("").tolist()
    raw    = df["sentiment"].astype(float).tolist()
    # Normalise from [-3, 3] to [0, 1]
    scores = [(v + 3.0) / 6.0 for v in raw]
    return SentimentDataset(texts, scores, tokenizer)


def _load_hebrew(path: Path, tokenizer: AutoTokenizer) -> Dataset | None:
    if not path.exists():
        print(f"Hebrew CSV not found at {path} — skipping.")
        return None
    df = pd.read_csv(path)
    texts  = df["text"].fillna("").tolist()
    raw    = df["sentiment"].astype(float).tolist()
    # Normalise from [-1, 1] to [0, 1]
    scores = [(v + 1.0) / 2.0 for v in raw]
    return SentimentDataset(texts, scores, tokenizer)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class XLMRobertaRegressor(nn.Module):
    """XLM-RoBERTa [CLS] → scalar sentiment score."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size  # 768
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, 1] score in [0, 1]."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]  # [B, 768]
        return self.regressor(cls_token)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    datasets = []
    ds_mosi   = _load_mosi(MOSI_CSV, tokenizer)
    ds_hebrew = _load_hebrew(HEBREW_CSV, tokenizer)
    if ds_mosi:
        datasets.append(ds_mosi)
    if ds_hebrew:
        datasets.append(ds_hebrew)

    if not datasets:
        raise RuntimeError("No training data found. Check CSV paths.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Combined dataset size: {len(combined)} samples")

    val_size   = max(1, int(0.15 * len(combined)))
    train_size = len(combined) - val_size
    train_ds, val_ds = random_split(
        combined, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = XLMRobertaRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                            total_iters=EPOCHS)

    best_mae  = float("inf")
    ckpt_path = OUTPUT_DIR / "xlm_roberta_sentiment.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(ids, mask)
            loss = criterion(pred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        mae = _evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_MAE={mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint (MAE={best_mae:.4f})")

    print(f"\nTraining complete. Best val MAE: {best_mae:.4f}")


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            pred = model(ids, mask).cpu().squeeze(1).numpy()
            preds_all.extend(pred)
            labels_all.extend(batch["label"].numpy())
    return float(mean_absolute_error(labels_all, preds_all))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(ckpt_path: Path | None = None) -> Path:
    ckpt_path = ckpt_path or (OUTPUT_DIR / "xlm_roberta_sentiment.pt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = XLMRobertaRegressor()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy_ids  = torch.zeros(1, MAX_LEN, dtype=torch.int64)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.int64)

    onnx_path = OUTPUT_DIR / "xlm_roberta_sentiment.onnx"
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["score"],
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "score":          {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX model exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    export_onnx()
