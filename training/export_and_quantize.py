# Phase 6.6 — Export all four trained PyTorch checkpoints to full-precision ONNX
# Quantization is intentionally skipped: INT8 dynamic quantization breaks LSTM/transformer
# accuracy (video F1 dropped from 0.87 to 0.07). Full-precision ONNX is fast enough for
# the project's <10s CPU inference budget.
from __future__ import annotations

import argparse
import time
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")

_CKPT_NAMES = {
    "video":  "efficientnet_lstm.pt",
    "audio":  "wav2vec2_classifier.pt",
    "text":   "xlm_roberta_sentiment.pt",
    "fusion": "cross_modal_transformer.pt",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def export_run(model: str = "all", skip_export: bool = False) -> None:
    """Notebook entry point — call this directly (e.g. `export_run()` or
    `export_run("audio")`) instead of main(). Running the whole script in a notebook cell
    triggers `if __name__ == "__main__"`, and argparse then reads the kernel's own
    sys.argv (e.g. Colab/Kaggle's `-f kernel.json` launcher flag) instead of your intended
    arguments, raising 'unrecognized arguments'. This function takes plain parameters
    instead, so it has nothing to do with sys.argv. One call now handles all four models
    in a single run and skips any model whose checkpoint isn't available yet, rather than
    stopping partway. Named export_run (not run) because evaluate_models.py also defines
    a notebook entry point called run() — both pasted into the same notebook would
    silently overwrite each other under the same name."""
    t_total = time.time()
    print(f"[Export] Target: {model}")

    if model in ("all", "video"):
        _handle("video", skip_export)
    if model in ("all", "audio"):
        _handle("audio", skip_export)
    if model in ("all", "text"):
        _handle("text", skip_export)
    if model in ("all", "fusion"):
        _handle("fusion", skip_export)

    elapsed = (time.time() - t_total) / 60
    print(f"\n[Export] All done in {elapsed:.1f} min.")
    _print_copy_instructions()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export + quantize DeepCue models.")
    parser.add_argument("--model", default="all",
                        choices=["all", "video", "audio", "text", "fusion"])
    parser.add_argument("--skip_export", action="store_true",
                        help="Skip PyTorch → ONNX step; only quantize existing .onnx files.")
    args = parser.parse_args()
    export_run(args.model, args.skip_export)


def _handle(model_name: str, skip_export: bool) -> None:
    onnx_path = _onnx_path(model_name)

    if not skip_export:
        ckpt_path = WORKING_DIR / _CKPT_NAMES[model_name]
        if not ckpt_path.exists():
            print(f"[Export] {model_name}: checkpoint {ckpt_path.name} not found — skipping entirely.")
            return
        t0 = time.time()
        print(f"\n[Export] {model_name}: PyTorch → ONNX ...")
        _export(model_name)
        print(f"[Export] {model_name}: ONNX written in {time.time()-t0:.1f}s")

    if not onnx_path.exists():
        print(f"[Export] {model_name}: {onnx_path} not found.")
        return

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"[Export] {model_name}: {onnx_path.name}  ({size_mb:.1f} MB)")


def _export(model_name: str) -> None:
    """Call the training script's export_onnx() to re-use its model definition."""
    if model_name == "video":
        from train_video_model import export_onnx
        export_onnx()
    elif model_name == "audio":
        from train_audio_model import export_onnx
        export_onnx()
    elif model_name == "text":
        from finetune_xlm_roberta import export_onnx
        export_onnx()
    elif model_name == "fusion":
        from train_fusion_model import export_onnx
        export_onnx()


def _onnx_path(model_name: str) -> Path:
    names = {
        "video":  "efficientnet_lstm.onnx",
        "audio":  "wav2vec2_classifier.onnx",
        "text":   "xlm_roberta_sentiment.onnx",
        "fusion": "cross_modal_transformer.onnx",
    }
    return WORKING_DIR / names[model_name]


def _print_copy_instructions() -> None:
    print("\n" + "=" * 60)
    print("Download from Kaggle output panel, then copy to backend:")
    print()
    print("  efficientnet_lstm.onnx        → models/video/efficientnet_lstm.onnx")
    print("  wav2vec2_classifier.onnx      → models/audio/wav2vec2_classifier.onnx")
    print("  xlm_roberta_sentiment.onnx    → models/text/xlm_roberta_sentiment.onnx")
    print("  cross_modal_transformer.onnx  → models/fusion/cross_modal_transformer.onnx")
    print()
    print("Then update VIDEO/AUDIO/TEXT/FUSION_MODEL_PATH in your .env file.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    print("=" * 50 + " SCRIPT COMPLETE " + "=" * 50)
