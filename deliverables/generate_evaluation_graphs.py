"""Generate presentation-ready evaluation graphs for the DeepCue project.

Standalone script — no project imports. Produces three PNG figures (300 DPI)
in an ``evaluation_graphs/`` folder next to this script:

1. ``modality_f1_comparison.png``  — macro F1 bar chart across classifiers.
2. ``text_mae_visual.png``         — MAE gauge-style visual for the text regressor.
3. ``metrics_dashboard.png``       — combined summary figure.

Usage:
    python generate_evaluation_graphs.py
"""

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Final DeepCue evaluation metrics (hardcoded per RESULTS.md, 2026-07-05)
# ---------------------------------------------------------------------------
VIDEO_F1: Final[float] = 0.80    # EfficientNet-B0 + LSTM, RAVDESS, actor-disjoint
AUDIO_F1: Final[float] = 0.45    # wav2vec 2.0, cross-source evaluation
FUSION_F1: Final[float] = 0.98   # Cross-Modal Transformer, architectural simulation
TEXT_MAE: Final[float] = 0.046   # XLM-RoBERTa Hebrew sentiment regression
F1_THRESHOLD: Final[float] = 0.50

PALETTE: Final[dict[str, str]] = {
    "video": "#4C72B0",
    "audio": "#DD8452",
    "fusion": "#55A868",
    "text": "#8172B3",
    "threshold": "#C44E52",
}

OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent / "evaluation_graphs"


def _apply_style() -> None:
    """Set a consistent academic look for all figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_f1_comparison(output_path: Path) -> None:
    """Bar chart comparing macro F1 across the three classifiers.

    Args:
        output_path: Destination PNG path.
    """
    labels = [
        "Video\nEfficientNet-B0 + LSTM",
        "Audio\nwav2vec 2.0",
        "Fusion*\nCross-Modal Transformer",
    ]
    scores = [VIDEO_F1, AUDIO_F1, FUSION_F1]
    colors = [PALETTE["video"], PALETTE["audio"], PALETTE["fusion"]]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(labels, scores, color=colors, width=0.55, edgecolor="white")

    ax.axhline(
        F1_THRESHOLD,
        color=PALETTE["threshold"],
        linestyle="--",
        linewidth=2,
        label=f"Quality gate (F1 = {F1_THRESHOLD:.2f})",
    )
    for bar, score in zip(bars, scores):
        ax.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=16,
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro F1 score")
    ax.set_title("DeepCue — Macro F1 by Modality (8 emotion classes)")
    ax.legend(loc="upper left", frameon=True)
    fig.text(
        0.99,
        0.01,
        "*Fusion evaluated on the Architectural Viability Simulation (synthetic paired data).",
        ha="right",
        fontsize=10,
        style="italic",
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_text_mae(output_path: Path) -> None:
    """Gauge-style visual for the text regressor's MAE.

    Shows the error on a 0–0.5 scale with qualitative bands, making
    "lower is better" immediately readable.

    Args:
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))

    bands = [
        (0.00, 0.05, "#55A868", "Excellent"),
        (0.05, 0.15, "#A1C99A", "Good"),
        (0.15, 0.30, "#F2CC8F", "Fair"),
        (0.30, 0.50, "#E07A5F", "Poor"),
    ]
    for start, end, color, label in bands:
        ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.85)
        ax.text(
            (start + end) / 2,
            -0.42,
            label,
            ha="center",
            fontsize=12,
            color="dimgray",
        )

    ax.axvline(TEXT_MAE, color="black", linewidth=3)
    ax.annotate(
        f"XLM-RoBERTa (Hebrew)\nMAE = {TEXT_MAE:.3f}",
        xy=(TEXT_MAE, 0.25),
        xytext=(TEXT_MAE + 0.05, 0.62),
        fontsize=14,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
    )

    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.6, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Mean Absolute Error (normalized sentiment scale — lower is better)")
    ax.set_title("DeepCue — Text Sentiment Regression Error")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(output_path: Path) -> None:
    """Combined summary figure: F1 bars plus a normalized-performance view.

    The right panel converts MAE to an accuracy-like score (1 - MAE) so all
    four models appear on one comparable 0-1 axis, clearly footnoted.

    Args:
        output_path: Destination PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: classification F1
    labels = ["Video", "Audio", "Fusion*"]
    scores = [VIDEO_F1, AUDIO_F1, FUSION_F1]
    colors = [PALETTE["video"], PALETTE["audio"], PALETTE["fusion"]]
    bars = ax1.bar(labels, scores, color=colors, width=0.5)
    ax1.axhline(F1_THRESHOLD, color=PALETTE["threshold"], linestyle="--", linewidth=2)
    for bar, score in zip(bars, scores):
        ax1.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Classification — Macro F1")
    ax1.set_ylabel("Macro F1")

    # Right: all four models, normalized 0-1 "performance" view
    all_labels = ["Video", "Audio", "Text**", "Fusion*"]
    all_scores = [VIDEO_F1, AUDIO_F1, 1.0 - TEXT_MAE, FUSION_F1]
    all_colors = [
        PALETTE["video"],
        PALETTE["audio"],
        PALETTE["text"],
        PALETTE["fusion"],
    ]
    bars2 = ax2.bar(all_labels, all_scores, color=all_colors, width=0.5)
    for bar, score in zip(bars2, all_scores):
        ax2.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax2.set_ylim(0, 1.1)
    ax2.set_title("All Experts — Normalized Performance")
    ax2.set_ylabel("Score (higher is better)")

    fig.suptitle("DeepCue — Final Evaluation Summary", fontweight="bold", y=1.02)
    fig.text(
        0.99,
        -0.03,
        "*Fusion: Architectural Viability Simulation (synthetic paired data).   "
        "**Text: shown as 1 − MAE (regression model, MAE = 0.046).",
        ha="right",
        fontsize=10,
        style="italic",
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    """Generate all evaluation graphs into ``evaluation_graphs/``."""
    _apply_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_f1_comparison(OUTPUT_DIR / "modality_f1_comparison.png")
    plot_text_mae(OUTPUT_DIR / "text_mae_visual.png")
    plot_dashboard(OUTPUT_DIR / "metrics_dashboard.png")

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"Saved: {path}")


if __name__ == "__main__":
    run()
