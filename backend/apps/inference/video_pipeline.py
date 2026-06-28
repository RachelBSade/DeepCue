"""
Video Emotion Pipeline (5A)

EfficientNet-B0 + LSTM inference on MediaPipe Face Mesh landmarks.

Preprocessing:  468 (x, y, z) landmarks → 224×224 three-channel image
                  ch0: XY presence map
                  ch1: Z depth map (normalised to [0,1])
                  ch2: XY presence map (repeated for CNN symmetry)
Temporal model: last VIDEO_LSTM_WINDOW_SIZE frames buffered per session,
                stacked into [1, window, 3, 224, 224] for ONNX model input.
Output:         float in [0.0, 1.0] — per-frame positive-emotion intensity
                fed into the fusion pipeline.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEUTRAL_FALLBACK: float = 0.5
_IMAGE_SIZE: int = 224
_EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]


class VideoEmotionPipeline:
    """Singleton CPU-inference pipeline for facial micro-expression analysis."""

    _instance: VideoEmotionPipeline | None = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "VideoEmotionPipeline":
        """Return the process-wide singleton, initialising it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.load_model()
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        from django.conf import settings
        self._model_path: str = settings.VIDEO_MODEL_PATH
        self._window_size: int = settings.VIDEO_LSTM_WINDOW_SIZE
        self._session: Any = None  # onnxruntime.InferenceSession
        # Per-session deque of preprocessed frames: session_id → deque[ndarray]
        self._buffers: dict[str, deque] = {}
        self._buf_lock = threading.Lock()

    def load_model(self) -> None:
        """Load the quantised ONNX model from disk. Safe to call if file is absent."""
        if not os.path.exists(self._model_path):
            logger.warning(
                "VideoEmotionPipeline: model not found at %s — returning fallback scores "
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
            logger.info("VideoEmotionPipeline loaded: %s", self._model_path)
        except Exception:
            logger.exception("VideoEmotionPipeline: failed to load ONNX model.")

    # ------------------------------------------------------------------
    # Public predict — NEUTRAL_FALLBACK wraps the entire call (5A.5)
    # ------------------------------------------------------------------

    def predict(
        self,
        landmarks: list[dict],
        frame_index: int,
        session_id: str,
    ) -> float:
        """
        Return an emotion intensity score for one landmark frame.

        Parameters
        ----------
        landmarks   : list of 468 {x, y, z} dicts from MediaPipe
        frame_index : monotonically increasing within the session
        session_id  : used to key the per-session LSTM frame buffer

        Returns
        -------
        float in [0.0, 1.0]; NEUTRAL_FALLBACK (0.5) on any exception.
        """
        try:
            return self._predict(landmarks, frame_index, session_id)
        except Exception:
            logger.exception("VideoEmotionPipeline.predict failed (session=%s)", session_id)
            return NEUTRAL_FALLBACK

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(
        self,
        landmarks: list[dict],
        frame_index: int,
        session_id: str,
    ) -> float:
        """Run the loaded ONNX session on a buffered landmark window; raises on failure (caller handles fallback)."""
        if self._session is None:
            return NEUTRAL_FALLBACK

        img = _landmarks_to_image(landmarks)  # [3, H, W] float32

        with self._buf_lock:
            if session_id not in self._buffers:
                self._buffers[session_id] = deque(maxlen=self._window_size)
            buf = self._buffers[session_id]
            buf.append(img)
            current_len = len(buf)

        # Wait until the buffer is full before running the LSTM.
        if current_len < self._window_size:
            return NEUTRAL_FALLBACK

        # Shape: [1, window_size, 3, H, W]
        with self._buf_lock:
            sequence = np.stack(list(self._buffers[session_id]), axis=0)
        sequence = np.expand_dims(sequence, axis=0).astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: sequence})
        score = float(outputs[0].flatten()[0])
        return float(np.clip(score, 0.0, 1.0))

    def clear_session_buffer(self, session_id: str) -> None:
        """Release the LSTM frame buffer for a completed session."""
        with self._buf_lock:
            self._buffers.pop(session_id, None)


# ---------------------------------------------------------------------------
# Preprocessing helpers (5A.2)
# ---------------------------------------------------------------------------

def _landmarks_to_image(landmarks: list[dict]) -> np.ndarray:
    """
    Render 468 MediaPipe landmarks onto a 224×224 three-channel float32 canvas.

    Ch 0 & 2 — XY presence map  : 1.0 in a 3×3 neighbourhood around each point.
    Ch 1     — Z depth map       : normalised depth value at each landmark position.

    Returns ndarray of shape [3, 224, 224].
    """
    S = _IMAGE_SIZE
    xy_canvas = np.zeros((S, S), dtype=np.float32)
    z_canvas  = np.zeros((S, S), dtype=np.float32)

    for lm in landmarks:
        px = int(np.clip(lm["x"] * (S - 1), 0, S - 1))
        py = int(np.clip(lm["y"] * (S - 1), 0, S - 1))
        z  = float(lm.get("z", 0.0))

        # 3×3 neighbourhood for the XY channel.
        x1, x2 = max(0, px - 1), min(S - 1, px + 1)
        y1, y2 = max(0, py - 1), min(S - 1, py + 1)
        xy_canvas[y1 : y2 + 1, x1 : x2 + 1] = 1.0
        z_canvas[py, px] = z

    # Normalise Z to [0, 1].
    z_min, z_max = z_canvas.min(), z_canvas.max()
    if z_max > z_min:
        z_canvas = (z_canvas - z_min) / (z_max - z_min)

    return np.stack([xy_canvas, z_canvas, xy_canvas], axis=0)  # [3, S, S]
