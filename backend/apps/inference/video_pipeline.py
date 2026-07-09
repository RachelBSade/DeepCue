"""
Video Emotion Pipeline (5A)

EfficientNet-B0 + LSTM inference on raw 224×224 RGB video frames.

Input:   base64-encoded JPEG captured from the browser video element,
         decoded and resized to 224×224, normalised to [0, 1] — matches
         the preprocessing used during training in train_video_model.py.

Temporal model: last VIDEO_LSTM_WINDOW_SIZE frames buffered per session,
                stacked into [1, window, 3, 224, 224] for ONNX model input.

Output:  np.ndarray of shape [8,] — raw logits for the 8 emotion classes,
         stored in Redis as JSON and consumed by fusion_tasks.run_fusion().
"""
from __future__ import annotations

import base64
import logging
import os
import threading
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEUTRAL_FALLBACK: float = 0.5
_IMAGE_SIZE: int = 224
_NUM_CLASSES: int = 8
_EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]
_NEUTRAL_LOGITS = np.zeros(_NUM_CLASSES, dtype=np.float32)


class VideoEmotionPipeline:
    """Singleton CPU-inference pipeline for facial micro-expression analysis."""

    _instance: VideoEmotionPipeline | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VideoEmotionPipeline":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.load_model()
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        from django.conf import settings
        self._model_path: str = settings.VIDEO_MODEL_PATH
        self._window_size: int = settings.VIDEO_LSTM_WINDOW_SIZE
        self._session: Any = None
        self._buffers: dict[str, deque] = {}
        self._buf_lock = threading.Lock()

    def load_model(self) -> None:
        if not os.path.exists(self._model_path):
            logger.warning(
                "VideoEmotionPipeline: model not found at %s — returning neutral logits.",
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

    def predict(
        self,
        frame_jpeg: str,
        frame_index: int,
        session_id: str,
    ) -> np.ndarray:
        """
        Return 8-class emotion logits for one video frame.

        Parameters
        ----------
        frame_jpeg  : base64-encoded JPEG string (224×224) from the browser
        frame_index : monotonically increasing within the session
        session_id  : used to key the per-session LSTM frame buffer

        Returns
        -------
        np.ndarray of shape [8,] — raw logits; zeros (neutral) on any exception.
        """
        try:
            return self._predict(frame_jpeg, frame_index, session_id)
        except Exception:
            logger.exception("VideoEmotionPipeline.predict failed (session=%s)", session_id)
            return _NEUTRAL_LOGITS.copy()

    def _predict(
        self,
        frame_jpeg: str,
        frame_index: int,
        session_id: str,
    ) -> np.ndarray:
        if self._session is None:
            return _NEUTRAL_LOGITS.copy()

        img = _decode_frame(frame_jpeg)  # [3, 224, 224] float32

        with self._buf_lock:
            if session_id not in self._buffers:
                self._buffers[session_id] = deque(maxlen=self._window_size)
            buf = self._buffers[session_id]
            buf.append(img)
            current_len = len(buf)

        if current_len < self._window_size:
            return _NEUTRAL_LOGITS.copy()

        with self._buf_lock:
            sequence = np.stack(list(self._buffers[session_id]), axis=0)
        sequence = np.expand_dims(sequence, axis=0).astype(np.float32)  # [1, W, 3, H, W]

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: sequence})
        return outputs[0].flatten().astype(np.float32)  # [8,]

    def clear_session_buffer(self, session_id: str) -> None:
        with self._buf_lock:
            self._buffers.pop(session_id, None)


# ---------------------------------------------------------------------------
# Preprocessing — matches train_video_model.py's _load_frames() exactly
# ---------------------------------------------------------------------------

def _decode_frame(frame_jpeg: str) -> np.ndarray:
    """
    Decode a base64 JPEG into a [3, 224, 224] float32 array in [0, 1].
    Matches the preprocessing in train_video_model.py:
        cv2.cvtColor → BGR to RGB, resize to (224, 224), transpose(2,0,1), / 255.
    """
    from PIL import Image
    import io

    jpeg_bytes = base64.b64decode(frame_jpeg)
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3] in [0, 1]
    return arr.transpose(2, 0, 1)                    # [3, H, W]
