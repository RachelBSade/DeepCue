"""
Text Emotion Pipeline (5C)

Hebrew speech-to-text (Whisper) → sentiment + uncertainty analysis (XLM-RoBERTa).

transcribe()  (5C.2):
    Loads OpenAI Whisper (base model by default) from WHISPER_CACHE_DIR.
    Decodes audio bytes to a float32 waveform, runs transcription with
    language="he" to bias towards Hebrew output.

predict()  (5C.3 / 5C.4):
    Tokenizes Hebrew text with the XLM-RoBERTa fast tokenizer (HuggingFace).
    Runs the fine-tuned ONNX model to produce a scalar emotion intensity score.

Both methods return NEUTRAL_FALLBACK = 0.5 on any exception.  (5C.5)
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEUTRAL_FALLBACK: float = 0.5
_TOKENIZER_NAME: str = "xlm-roberta-base"
_MAX_TOKEN_LEN: int = 128


class TextEmotionPipeline:
    """Singleton pipeline: Whisper transcription + XLM-RoBERTa scoring."""

    _instance: TextEmotionPipeline | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TextEmotionPipeline":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.load_model()
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        from django.conf import settings
        self._model_path: str     = settings.TEXT_MODEL_PATH
        self._whisper_size: str   = settings.WHISPER_MODEL_SIZE
        self._whisper_cache: str  = settings.WHISPER_CACHE_DIR
        self._whisper_model: Any  = None
        self._ort_session: Any    = None
        self._tokenizer: Any      = None

    def load_model(self) -> None:
        """Load Whisper and the XLM-RoBERTa ONNX model. Each is independent."""
        self._load_whisper()
        self._load_xlm_roberta()

    def _load_whisper(self) -> None:
        try:
            import whisper
            import os
            os.makedirs(self._whisper_cache, exist_ok=True)
            self._whisper_model = whisper.load_model(
                self._whisper_size,
                download_root=self._whisper_cache,
            )
            logger.info("Whisper '%s' model loaded.", self._whisper_size)
        except Exception:
            logger.exception("TextEmotionPipeline: failed to load Whisper.")

    def _load_xlm_roberta(self) -> None:
        import os
        if not os.path.exists(self._model_path):
            logger.warning(
                "TextEmotionPipeline: XLM-RoBERTa ONNX not found at %s — "
                "returning fallback scores until weights are placed there.",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._ort_session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            # Tokenizer is loaded from HuggingFace Hub (cached locally).
            self._tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
            logger.info("XLM-RoBERTa ONNX loaded: %s", self._model_path)
        except Exception:
            logger.exception("TextEmotionPipeline: failed to load XLM-RoBERTa.")

    # ------------------------------------------------------------------
    # Public API — NEUTRAL_FALLBACK wraps both methods (5C.5)
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe Hebrew audio to text using Whisper.

        Parameters
        ----------
        audio_bytes : WAV bytes (16-bit mono, 16 kHz)

        Returns
        -------
        Hebrew transcript string, or "" on failure.
        """
        try:
            return self._transcribe(audio_bytes)
        except Exception:
            logger.exception("TextEmotionPipeline.transcribe failed")
            return ""

    def predict(self, text: str) -> float:
        """
        Score a Hebrew text string for emotion intensity.

        Parameters
        ----------
        text : Hebrew transcript from Whisper

        Returns
        -------
        float in [0.0, 1.0]; NEUTRAL_FALLBACK on any exception or empty text.
        """
        if not text.strip():
            return NEUTRAL_FALLBACK
        try:
            return self._predict(text)
        except Exception:
            logger.exception("TextEmotionPipeline.predict failed")
            return NEUTRAL_FALLBACK

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transcribe(self, audio_bytes: bytes) -> str:
        if self._whisper_model is None:
            return ""

        waveform = _decode_audio(audio_bytes)  # float32 numpy array at 16 kHz

        result = self._whisper_model.transcribe(
            waveform,
            language="he",      # Hebrew (5C.4)
            fp16=False,         # CPU inference — fp16 not supported
            task="transcribe",
        )
        return result.get("text", "").strip()

    def _predict(self, text: str) -> float:
        if self._ort_session is None or self._tokenizer is None:
            return NEUTRAL_FALLBACK

        # Tokenise with Hebrew-aware XLM-RoBERTa tokenizer (5C.4).
        encoding = self._tokenizer(
            text,
            return_tensors="np",
            max_length=_MAX_TOKEN_LEN,
            truncation=True,
            padding="max_length",
        )

        feed = {
            "input_ids":      encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

        outputs = self._ort_session.run(None, feed)
        score = float(outputs[0].flatten()[0])
        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to a mono float32 waveform at 16 kHz."""
    import soundfile as sf
    import librosa

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)
