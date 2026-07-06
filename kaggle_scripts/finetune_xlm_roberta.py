# Phase 6.3 — XLM-RoBERTa fine-tuning on omilab/hebrew_sentiment (+optional local Hebrew CSV) → scalar score ONNX
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Config — update paths to match your Kaggle dataset mounts
# ---------------------------------------------------------------------------

HEBREW_SENTIMENT_HF_DATASET = "omilab/hebrew_sentiment"
HEBREW_CSV   = Path("/kaggle/input/hebrew-sentiment/hebrew_sentiment.csv")  # optional, local supplement
OUTPUT_DIR   = Path("/kaggle/working")
MODEL_NAME   = "xlm-roberta-base"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 2e-5
WEIGHT_DECAY = 1e-2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Text regression dataset with scores normalised to [0, 1]."""

    def __init__(self, texts: list[str], scores: list[float], tokenizer) -> None:
        self.texts     = texts
        self.scores    = scores
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


# omilab/hebrew_sentiment label int → scalar score in [0, 1]. 2 ("off-topic") is treated as
# the neutral midpoint since it carries no positive/negative signal, not a third class.
_HEBREW_SENTIMENT_LABEL_SCORE: dict[int, float] = {0: 1.0, 1: 0.0, 2: 0.5}


def _load_hebrew_sentiment_hf(tokenizer) -> Dataset:
    """Load the omilab/hebrew_sentiment HF dataset (train split) — 12.8k labeled Hebrew
    Facebook comments. Replaces the original CMU-MOSI source, whose Kaggle mirror ships
    raw transcripts/audio/video with no sentiment labels at all."""
    # revision="refs/convert/parquet" — the repo uses a custom loading script, which newer
    # `datasets` versions refuse to execute (security restriction); HF auto-mirrors such
    # repos to Parquet on this branch, which loads without running any repo code. The
    # Parquet mirror collapses the original token/morph configs into a single "default"
    # config, so no config name is passed here.
    hf_dataset = load_dataset(HEBREW_SENTIMENT_HF_DATASET, revision="refs/convert/parquet")["train"]
    texts  = list(hf_dataset["text"])
    scores = [_HEBREW_SENTIMENT_LABEL_SCORE[label] for label in hf_dataset["label"]]
    print(f"[Text] Loaded {HEBREW_SENTIMENT_HF_DATASET}: {len(texts)} samples")
    return SentimentDataset(texts, scores, tokenizer)


def _load_hebrew(path: Path, tokenizer) -> Dataset | None:
    """Load Hebrew CSV if available; normalise sentiment [-1, 1] → [0, 1]."""
    if not path.exists():
        print(f"[Text] Hebrew CSV not found at {path} — skipping (optional).")
        return None
    df     = pd.read_csv(path)
    texts  = df["text"].fillna("").tolist()
    scores = [(v + 1.0) / 2.0 for v in df["sentiment"].astype(float).tolist()]
    print(f"[Text] Loaded Hebrew: {len(texts)} samples")
    return SentimentDataset(texts, scores, tokenizer)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class XLMRobertaRegressor(nn.Module):
    """XLM-RoBERTa [CLS] token → Dropout → Linear(768,1) → Sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(MODEL_NAME)
        hidden         = self.encoder.config.hidden_size  # 768
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]  # [B, 768] — CLS representation
        return self.regressor(cls_token)               # [B, 1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[Text] Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Combine available datasets — HF Hebrew sentiment is required, local CSV is an optional supplement
    datasets  = [d for d in [
        _load_hebrew_sentiment_hf(tokenizer),
        _load_hebrew(HEBREW_CSV, tokenizer),
    ] if d is not None]

    combined   = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[Text] Combined dataset size: {len(combined)} samples")

    val_size   = max(1, int(0.15 * len(combined)))
    train_ds, val_ds = random_split(
        combined, [len(combined) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = XLMRobertaRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # LR decays linearly from LR to LR/10 over EPOCHS
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                            total_iters=EPOCHS)
    best_mae  = float("inf")
    ckpt_path = OUTPUT_DIR / "xlm_roberta_sentiment.pt"

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(ids, mask)
            loss = criterion(pred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx+1}/{len(train_loader)}  loss={loss.item():.4f}")

        scheduler.step()
        mae     = _evaluate(model, val_loader)
        elapsed = time.time() - t0
        print(f"[Text] Epoch {epoch:02d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  "
              f"val_MAE={mae:.4f}  ({elapsed:.0f}s)")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved (MAE={best_mae:.4f})")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[Text] Training complete in {elapsed:.1f} min. Best val MAE: {best_mae:.4f}")


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
    print("\n[Text] Exporting to ONNX ...")
    ckpt_path = ckpt_path or (OUTPUT_DIR / "xlm_roberta_sentiment.pt")

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
        dynamo=False,  # force legacy TorchScript exporter; avoids needing onnxscript
    )
    print(f"[Text] ONNX exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    # export_onnx() runs separately, after you Save Version, in its own cell:
    #   from finetune_xlm_roberta import export_onnx
    #   export_onnx()
    print("=" * 50 + " TRAINING COMPLETE — SAVE VERSION NOW " + "=" * 50)
