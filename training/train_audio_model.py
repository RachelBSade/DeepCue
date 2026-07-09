# Phase 6.2 — wav2vec2-base fine-tuning on RAVDESS audio → scalar emotion score ONNX
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Wav2Vec2Model, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config — edit RAVDESS_ROOT to match your Kaggle dataset mount path
# ---------------------------------------------------------------------------

RAVDESS_ROOT  = Path("/kaggle/input/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
OUTPUT_DIR    = Path("/kaggle/working")
SAMPLE_RATE   = 16000
AUDIO_LEN     = 48000       # 3 seconds at 16 kHz
BATCH_SIZE    = 8
STAGE1_EPOCHS = 12          # linear probe — wav2vec2 fully frozen, head only
STAGE2_EPOCHS = 20          # unfreeze top transformer layers, low-LR fine-tune
LR_STAGE1     = 1e-3        # head is small + randomly initialised — can move fast
LR_STAGE2     = 2e-5
STAGE2_UNFREEZE_LAYERS = 4  # of wav2vec2-base's 12 encoder layers — reduced from 6: val F1 was
                            # oscillating while train loss collapsed to ~0 by epoch 5-8, a sign
                            # Stage 2 had more capacity to memorize than the ~280-batch train set warrants
EARLY_STOP_PATIENCE = 5     # stop Stage 2 if val F1 hasn't improved in this many epochs
VAL_ACTOR_FRACTION = 0.2    # fraction of RAVDESS actors held out for validation
NUM_CLASSES   = 8
N_MFCC        = 13
FEATURE_DIM   = 3 + N_MFCC  # pitch + RMS + ZCR + 13 MFCCs = 16
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42

# RAVDESS emotion code (field 3 in filename) → DeepCue class index
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
# Dataset
# ---------------------------------------------------------------------------

class RAVDESSAudioDataset(Dataset):
    """Load RAVDESS wavs, extract audio on-the-fly, return waveform + features + label + valid length."""

    def __init__(self, root: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.actors: list[int] = []  # parallel to self.samples — actor ID per sample (filename field 7)
        for wav in sorted(root.rglob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) >= 7:
                try:
                    code  = int(parts[2])
                    actor = int(parts[6])
                    label = _RAVDESS_MAP.get(code)
                    if label is not None:
                        self.samples.append((wav, label))
                        self.actors.append(actor)
                except ValueError:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        path, label = self.samples[idx]
        waveform, valid_length = _load_audio(path)  # [48000,] float32, real-sample count
        features = _extract_features(waveform)       # [16,]    float32
        return (
            torch.from_numpy(waveform).float(),
            torch.from_numpy(features).float(),
            label,
            valid_length,
        )


def _actor_disjoint_split(
    dataset: RAVDESSAudioDataset,
    val_fraction: float = VAL_ACTOR_FRACTION,
) -> tuple[Subset, Subset]:
    """Split by actor ID, not by sample, so no speaker appears in both train and val.

    RAVDESS has only 24 actors repeating the same two sentences across all emotions —
    a per-sample random split leaks voice identity between train/val and inflates the
    reported F1 relative to true speaker-independent generalisation."""
    rng = np.random.default_rng(SEED)
    unique_actors = sorted(set(dataset.actors))
    rng.shuffle(unique_actors)
    n_val = max(1, round(val_fraction * len(unique_actors)))
    val_actors = set(unique_actors[:n_val])

    train_idx = [i for i, a in enumerate(dataset.actors) if a not in val_actors]
    val_idx   = [i for i, a in enumerate(dataset.actors) if a in val_actors]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load first audio stream, resample to 16 kHz, zero-mean/unit-variance normalize
    (matches Wav2Vec2FeatureExtractor's default preprocessing — wav2vec2 was pretrained
    on normalized input, so skipping this starves the model of in-distribution input),
    then pad/truncate to AUDIO_LEN samples. Returns the padded waveform plus the count
    of real (non-padded) samples, used to build an attention mask for pooling."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    audio = waveform.squeeze().numpy()
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    valid_length = min(len(audio), AUDIO_LEN)
    if len(audio) < AUDIO_LEN:
        audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
    return audio[:AUDIO_LEN].astype(np.float32), valid_length


def _compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights so the doubled-up 'neutral' class (RAVDESS codes
    1+2 both map to class 0) doesn't dominate the loss. Computed from the train split
    only, so validation-set class balance can't leak into training."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = counts.sum() / np.maximum(counts, 1)
    return torch.tensor(weights / weights.sum() * NUM_CLASSES, dtype=torch.float32)


def _extract_features(waveform: np.ndarray) -> np.ndarray:
    """16-dim paralinguistic feature vector — must match AudioEmotionPipeline._extract_features."""
    import librosa
    f0     = librosa.yin(waveform, fmin=80.0, fmax=400.0, sr=SAMPLE_RATE)
    voiced = f0[f0 > 0]
    pitch  = float(np.mean(voiced) / 400.0) if len(voiced) > 0 else 0.0
    rms    = float(np.mean(librosa.feature.rms(y=waveform)[0]))
    zcr    = float(np.mean(librosa.feature.zero_crossing_rate(waveform)[0]))
    mfccs  = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC).mean(axis=1).tolist()
    return np.array([pitch, rms, zcr] + mfccs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Wav2Vec2EmotionClassifier(nn.Module):
    """wav2vec2-base + shared MLP with two heads: class logits and scalar score."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # Disable gradient checkpointing — combined with SpecAugment's random time/feature
        # masking (also pretraining-only, not wanted here) it makes the backward recomputation
        # take a different random path than the forward pass and raises a CheckpointError.
        # gradient_checkpointing_disable() alone isn't reliable across transformers versions
        # (it doesn't always propagate down to the encoder submodule's own flag), so force it
        # directly on every submodule and neutralise the checkpoint function as a backstop.
        self.wav2vec2.gradient_checkpointing_disable()
        self.wav2vec2.config.apply_spec_augment = False
        for module in self.wav2vec2.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            if hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = lambda func, *args, **kwargs: func(*args, **kwargs)
        hidden_dim    = self.wav2vec2.config.hidden_size  # 768

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(64, num_classes)
        self.score_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def freeze_backbone(self) -> None:
        """Freeze the entire wav2vec2 backbone (CNN + feature projection + all transformer
        layers). Stage 1 is then a true linear probe — only the head trains, so the
        classifier stabilises before any pretrained representations are allowed to drift."""
        for p in self.wav2vec2.parameters():
            p.requires_grad_(False)

    def unfreeze_top_layers(self, n_layers: int) -> None:
        """Unfreeze the top n_layers transformer encoder layers + feature_projection for
        Stage 2 fine-tuning. The CNN feature_extractor stays frozen — its low-level acoustic
        filters don't need task-specific adaptation and are the most overfit-prone to retrain
        on ~1,200 samples."""
        for p in self.wav2vec2.feature_projection.parameters():
            p.requires_grad_(True)
        for layer in self.wav2vec2.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)

    def forward(
        self,
        audio_waveform: torch.Tensor,            # [B, 48000]
        features: torch.Tensor,                  # [B, 16]
        attention_mask: torch.Tensor | None = None,  # [B, 48000], 1 = real sample, 0 = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(audio_waveform, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state  # [B, T, 768]

        if attention_mask is not None:
            # Downsample the raw-waveform mask to the encoder's output frame rate (HF helper),
            # then mean-pool only over real frames — padded silence no longer dilutes the signal.
            feat_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            feat_mask = feat_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1e-6)
        else:
            # No mask given (e.g. ONNX export with full-length dummy input) — plain mean.
            pooled = hidden.mean(dim=1)

        combined = torch.cat([pooled, features], dim=1)             # [B, 784]
        shared   = self.shared(combined)                            # [B, 64]
        return self.class_head(shared), self.score_head(shared)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[Audio] Device: {DEVICE}")
    train_loader, val_loader, dataset, train_ds = _build_dataloaders()

    train_labels  = [dataset.samples[i][1] for i in train_ds.indices]
    class_weights = _compute_class_weights(train_labels).to(DEVICE)
    # Plain hard-target loss for both stages — label_smoothing combined with these
    # per-class weights caused Stage 2 loss to diverge (>30 from epoch 1, val F1 falling),
    # so it's been dropped in favour of just the capacity/early-stopping changes below.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = Wav2Vec2EmotionClassifier().to(DEVICE)

    print(f"\n[Audio] Stage 1: linear probe, head only ({STAGE1_EPOCHS} epochs)")
    _run_stage1(model, train_loader, val_loader, criterion)

    print(f"\n[Audio] Stage 2: fine-tuning top {STAGE2_UNFREEZE_LAYERS} layers ({STAGE2_EPOCHS} epochs)")
    best_f1 = _run_stage2(model, train_loader, val_loader, criterion)

    elapsed = (time.time() - t_start) / 60
    print(f"\n[Audio] Training complete in {elapsed:.1f} min. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("[Audio] WARNING: F1 below 0.50 threshold.")


def _run_stage1(
    model: Wav2Vec2EmotionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
) -> None:
    """Freeze the backbone, train the head as a linear probe, then save a Stage 1
    checkpoint so Stage 2 can be re-run later via train_stage2_only() without repeating
    this (slower, ~12-epoch) stage."""
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1)
    scheduler = _make_warmup_scheduler(optimizer, STAGE1_EPOCHS, len(train_loader))
    _run_epochs(model, train_loader, val_loader, criterion, optimizer, STAGE1_EPOCHS, scheduler=scheduler)

    ckpt_path = OUTPUT_DIR / "wav2vec2_stage1.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Audio] Stage 1 checkpoint saved: {ckpt_path}")


def _run_stage2(
    model: Wav2Vec2EmotionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Unfreeze the top transformer layers and fine-tune at a low LR, saving the
    best-val-F1 checkpoint to wav2vec2_classifier.pt."""
    model.unfreeze_top_layers(STAGE2_UNFREEZE_LAYERS)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2)
    scheduler = _make_warmup_scheduler(optimizer, STAGE2_EPOCHS, len(train_loader))
    return _run_epochs(
        model, train_loader, val_loader, criterion, optimizer, STAGE2_EPOCHS,
        ckpt_path=OUTPUT_DIR / "wav2vec2_classifier.pt", scheduler=scheduler,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )


def _build_dataloaders() -> tuple[DataLoader, DataLoader, RAVDESSAudioDataset, Subset]:
    """Load the dataset and build the actor-disjoint train/val loaders — shared by both
    train() and train_stage2_only() so the split is identical across runs (same SEED)."""
    print(f"[Audio] Loading dataset from {RAVDESS_ROOT} ...")
    dataset = RAVDESSAudioDataset(RAVDESS_ROOT)
    print(f"[Audio] Dataset size: {len(dataset)} samples, {len(set(dataset.actors))} actors")

    train_ds, val_ds = _actor_disjoint_split(dataset)
    print(f"[Audio] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples (actor-disjoint split)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, dataset, train_ds


def train_stage2_only(stage1_ckpt_path: Path | None = None) -> None:
    """Resume from a saved Stage 1 checkpoint and run only Stage 2 — for iterating on
    Stage 2 hyperparameters (unfreeze depth, LR, etc.) without repeating the Stage 1
    linear-probe run each time."""
    torch.manual_seed(SEED)
    stage1_ckpt_path = stage1_ckpt_path or (OUTPUT_DIR / "wav2vec2_stage1.pt")
    t_start = time.time()

    print(f"[Audio] Device: {DEVICE}")
    train_loader, val_loader, dataset, train_ds = _build_dataloaders()

    train_labels  = [dataset.samples[i][1] for i in train_ds.indices]
    class_weights = _compute_class_weights(train_labels).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    model = Wav2Vec2EmotionClassifier().to(DEVICE)
    model.load_state_dict(torch.load(stage1_ckpt_path, map_location=DEVICE))
    print(f"[Audio] Loaded Stage 1 checkpoint: {stage1_ckpt_path}")

    print(f"\n[Audio] Stage 2: fine-tuning top {STAGE2_UNFREEZE_LAYERS} layers ({STAGE2_EPOCHS} epochs)")
    best_f1 = _run_stage2(model, train_loader, val_loader, criterion)

    elapsed = (time.time() - t_start) / 60
    print(f"\n[Audio] Stage 2 complete in {elapsed:.1f} min. Best val Macro F1: {best_f1:.4f}")
    if best_f1 < 0.50:
        print("[Audio] WARNING: F1 below 0.50 threshold.")


def _make_warmup_scheduler(optimizer: optim.Optimizer, n_epochs: int, steps_per_epoch: int):
    """Linear warmup (10% of steps) then linear decay — avoids the large, unstable early
    gradient updates that constant-LR fine-tuning applies to pretrained transformer weights."""
    total_steps = max(1, n_epochs * steps_per_epoch)
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )


def _build_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """[B, max_len] mask of 1s for real samples, 0s for the padded tail, from per-sample lengths."""
    return (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()


def _run_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int,
    ckpt_path: Path | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    early_stop_patience: int | None = None,
) -> float:
    """Run training epochs, saving the best-val-F1 checkpoint. If early_stop_patience is
    set, stop once val F1 hasn't improved for that many consecutive epochs — avoids burning
    remaining epochs once the model has plateaued/overfit (observed: Stage 2 train loss
    collapses to ~0 by epoch 5-8 while val F1 stops climbing)."""
    best_f1 = 0.0
    epochs_without_improvement = 0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (wav, feat, labels, lengths) in enumerate(train_loader):
            wav, feat, labels = wav.to(DEVICE), feat.to(DEVICE), labels.to(DEVICE)
            attn_mask = _build_attention_mask(lengths, wav.shape[1]).to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(wav, feat, attention_mask=attn_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

            # Print batch progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx+1}/{len(train_loader)}  loss={loss.item():.4f}")

        f1      = _evaluate(model, val_loader)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{n_epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"val_f1={f1:.4f}  ({elapsed:.0f}s)")

        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
                print(f"  → Checkpoint saved (f1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                print(f"  Early stopping: no val F1 improvement in {early_stop_patience} epochs.")
                break

    return best_f1


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for wav, feat, labels, lengths in loader:
            attn_mask = _build_attention_mask(lengths, wav.shape[1]).to(DEVICE)
            logits, _ = model(wav.to(DEVICE), feat.to(DEVICE), attention_mask=attn_mask)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# ONNX export — exports the 8-class logits (fusion now takes per-modality logits,
# not a single scalar score, to preserve more emotional resolution per modality)
# ---------------------------------------------------------------------------

def export_onnx(ckpt_path: Path | None = None) -> Path:
    print("\n[Audio] Exporting to ONNX ...")
    ckpt_path = ckpt_path or (OUTPUT_DIR / "wav2vec2_classifier.pt")

    model = Wav2Vec2EmotionClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    class _LogitsOnlyWrapper(nn.Module):
        """Export the 8-class classification logits — scalar score head not used at runtime."""
        def __init__(self, inner: Wav2Vec2EmotionClassifier) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, audio_waveform: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            logits, _ = self.inner(audio_waveform, features)
            return logits  # [B, 8]

    wrapper   = _LogitsOnlyWrapper(model)
    dummy_wav = torch.zeros(1, AUDIO_LEN)
    dummy_feat = torch.zeros(1, FEATURE_DIM)

    onnx_path = OUTPUT_DIR / "wav2vec2_classifier.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_wav, dummy_feat),
        str(onnx_path),
        input_names=["audio_waveform", "features"],
        output_names=["logits"],
        dynamic_axes={
            "audio_waveform": {0: "batch"},
            "features":       {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # force legacy TorchScript exporter; avoids needing onnxscript
    )
    print(f"[Audio] ONNX exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    train()
    # export_onnx() runs separately, after you Save Version, in its own cell:
    #   from train_audio_model import export_onnx
    #   export_onnx()
    #
    # To re-run only Stage 2 against an existing Stage 1 checkpoint (skips the
    # linear-probe stage), run this in its own cell instead of train():
    #   from train_audio_model import train_stage2_only
    #   train_stage2_only()
    print("=" * 50 + " TRAINING COMPLETE — SAVE VERSION NOW " + "=" * 50)
