"""
Fusion Pipeline (5D)

Cross-modal Transformer + MLP head that fuses three modality scores into
an 8-class emotion distribution.

Architecture (as exported from Kaggle training):
    Input  : [1, 3]  — [video_score, audio_score, text_score]
    Encoder: Cross-modal Transformer (self-attention over 3 modality tokens)
    Head   : Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 8) → Softmax
    Output : [1, 8]  — confidence per emotion class (sums to 1.0)

8 emotion classes (5D.5):
    neutral, confident, anxious, happy, sad, angry, surprised, uncertain

Fallback output (5D.6):
    {neutral: 1.0, all others: 0.0}
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEUTRAL_FALLBACK: float = 0.5

# Speech-rate stress heuristic — normal conversational Hebrew speech sits
# roughly in [_WPM_CALM, _WPM_STRESSED]; faster speech nudges 'anxious' up.
_WPM_CALM: float     = 90.0
_WPM_STRESSED: float = 200.0
_MAX_NUDGE: float    = 0.1

EMOTION_CLASSES: list[str] = [
    "neutral",
    "confident",
    "anxious",
    "happy",
    "sad",
    "angry",
    "surprised",
    "uncertain",
]

# Returned when the model is unavailable or throws any exception.
_FALLBACK_OUTPUT: dict[str, float] = {
    emotion: (1.0 if emotion == "neutral" else 0.0)
    for emotion in EMOTION_CLASSES
}


class FusionPipeline:
    """Singleton CPU-inference pipeline that fuses three modality scores."""

    _instance: FusionPipeline | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "FusionPipeline":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.load_model()
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        from django.conf import settings
        self._model_path: str = settings.FUSION_MODEL_PATH
        self._session: Any = None

    def load_model(self) -> None:
        import os
        if not os.path.exists(self._model_path):
            logger.warning(
                "FusionPipeline: model not found at %s — returning neutral fallback "
                "until weights are placed there after Kaggle training.",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info("FusionPipeline loaded: %s", self._model_path)
        except Exception:
            logger.exception("FusionPipeline: failed to load ONNX model.")

    # ------------------------------------------------------------------
    # Public predict — NEUTRAL_FALLBACK wraps everything (5D.6)
    # ------------------------------------------------------------------

    def predict(
        self,
        video_logits: list[float],
        audio_logits: list[float],
        text_score: float,
    ) -> dict[str, float]:
        """
        Fuse video/audio logits and text score into an 8-class emotion distribution.

        Parameters
        ----------
        video_logits : list of 8 floats — raw logits from VideoEmotionPipeline
        audio_logits : list of 8 floats — raw logits from AudioEmotionPipeline
        text_score   : float in [0,1] from TextEmotionPipeline

        Returns
        -------
        dict mapping each of the 8 emotion labels to a confidence float summing to 1.0.
        Returns {neutral: 1.0, ...} on any exception.
        """
        try:
            return self._predict(video_logits, audio_logits, text_score)
        except Exception:
            logger.exception("FusionPipeline.predict failed")
            return dict(_FALLBACK_OUTPUT)

    def apply_speech_rate(
        self,
        scores: dict[str, float],
        wpm: float | None,
    ) -> dict[str, float]:
        """
        Nudge the 'anxious' score based on speaking rate (words-per-minute),
        then renormalize so the distribution still sums to 1.0.

        Parameters
        ----------
        scores : output of predict()
        wpm    : words-per-minute from TextEmotionPipeline.compute_speech_rate(),
                 or None if no transcript was available for this window.

        Returns
        -------
        Adjusted scores dict; unchanged if wpm is None.
        """
        if wpm is None or wpm <= 0:
            return scores
        try:
            stress = float(np.clip((wpm - _WPM_CALM) / (_WPM_STRESSED - _WPM_CALM), 0.0, 1.0))
            nudge = (stress - 0.5) * 2 * _MAX_NUDGE  # in [-_MAX_NUDGE, +_MAX_NUDGE]

            adjusted = dict(scores)
            adjusted["anxious"] = float(np.clip(adjusted["anxious"] + nudge, 0.0, 1.0))

            total = sum(adjusted.values())
            return {k: v / total for k, v in adjusted.items()} if total > 0 else adjusted
        except Exception:
            logger.exception("FusionPipeline.apply_speech_rate failed")
            return scores

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(
        self,
        video_logits: list[float],
        audio_logits: list[float],
        text_score: float,
    ) -> dict[str, float]:
        if self._session is None:
            return dict(_FALLBACK_OUTPUT)

        # Input: [1, 17] — video[8] + audio[8] + text[1]
        model_input = np.array(
            [video_logits + audio_logits + [text_score]],
            dtype=np.float32,
        )

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: model_input})

        raw: np.ndarray = outputs[0].flatten()  # [8,]

        # Apply softmax in case the model outputs logits rather than probabilities.
        probs = _softmax(raw)

        return {
            emotion: float(round(probs[i], 6))
            for i, emotion in enumerate(EMOTION_CLASSES)
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
