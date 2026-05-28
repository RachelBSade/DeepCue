"""
Phase 6.6 — Unified ONNX Export & INT8 Dynamic Quantization (Kaggle GPU)

Converts all four trained PyTorch checkpoints to ONNX, then applies
INT8 dynamic quantization using onnxruntime.quantization.

Run this after training all four models (6.1–6.4).

Input  (all under /kaggle/working/):
    efficientnet_lstm.pt
    wav2vec2_classifier.pt
    xlm_roberta_sentiment.pt
    cross_modal_transformer.pt

Output (all under /kaggle/working/):
    *_quant.onnx  ← copy each to its corresponding models/<modality>/ directory

Usage:
    python export_and_quantize.py
    python export_and_quantize.py --model video
    python export_and_quantize.py --skip_export   # only quantize existing .onnx files
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export + quantize DeepCue models.")
    parser.add_argument("--model", default="all",
                        choices=["all", "video", "audio", "text", "fusion"])
    parser.add_argument("--skip_export", action="store_true",
                        help="Skip PyTorch → ONNX export; only quantize existing .onnx files.")
    args = parser.parse_args()

    target = args.model

    if target in ("all", "video"):
        _handle("video", args.skip_export)
    if target in ("all", "audio"):
        _handle("audio", args.skip_export)
    if target in ("all", "text"):
        _handle("text", args.skip_export)
    if target in ("all", "fusion"):
        _handle("fusion", args.skip_export)

    print("\nAll requested models exported and quantized.")
    _print_copy_instructions()


def _handle(model_name: str, skip_export: bool) -> None:
    onnx_path = _onnx_path(model_name)

    if not skip_export:
        print(f"\n[{model_name}] Exporting PyTorch → ONNX ...")
        _export(model_name)

    if not onnx_path.exists():
        print(f"[{model_name}] ONNX file not found: {onnx_path} — skipping quantization.")
        return

    quant_path = _quant_path(model_name)
    print(f"[{model_name}] Quantizing {onnx_path.name} → {quant_path.name} ...")
    _quantize(onnx_path, quant_path)
    size_fp = onnx_path.stat().st_size / 1024 / 1024
    size_q  = quant_path.stat().st_size / 1024 / 1024
    print(f"[{model_name}] Full: {size_fp:.1f} MB  →  Quantized: {size_q:.1f} MB  "
          f"({100 * (1 - size_q / size_fp):.0f}% reduction)")


def _export(model_name: str) -> None:
    """Delegate to the training script's export_onnx() function."""
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


def _quantize(onnx_path: Path, quant_path: Path) -> None:
    """Apply INT8 dynamic quantization using onnxruntime.quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )


def _onnx_path(model_name: str) -> Path:
    names = {
        "video":  "efficientnet_lstm.onnx",
        "audio":  "wav2vec2_classifier.onnx",
        "text":   "xlm_roberta_sentiment.onnx",
        "fusion": "cross_modal_transformer.onnx",
    }
    return WORKING_DIR / names[model_name]


def _quant_path(model_name: str) -> Path:
    p = _onnx_path(model_name)
    return p.with_name(p.stem + "_quant.onnx")


def _print_copy_instructions() -> None:
    print("\n" + "="*60)
    print("Copy quantized models to the Django backend:")
    print()
    print("  /kaggle/working/efficientnet_lstm_quant.onnx")
    print("    → models/video/efficientnet_lstm.onnx")
    print()
    print("  /kaggle/working/wav2vec2_classifier_quant.onnx")
    print("    → models/audio/wav2vec2_classifier.onnx")
    print()
    print("  /kaggle/working/xlm_roberta_sentiment_quant.onnx")
    print("    → models/text/xlm_roberta_sentiment.onnx")
    print()
    print("  /kaggle/working/cross_modal_transformer_quant.onnx")
    print("    → models/fusion/cross_modal_transformer.onnx")
    print()
    print("Then update VIDEO/AUDIO/TEXT/FUSION_MODEL_PATH in your .env file.")
    print("="*60)


if __name__ == "__main__":
    main()
