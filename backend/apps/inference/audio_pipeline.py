"""
Audio Emotion Pipeline (5B)

wav2vec 2.0 deep embeddings + paralinguistic features → emotion score.

Feature extraction  (5B.2):
    Pitch (F0 via librosa.yin), RMS energy, zero-crossing rate,
    and 13 MFCCs — all computed from the raw waveform.
    Combined into a [16,] feature vector alongside the wav2vec embedding.

ONNX model  (5B.3 / 5B.4):
    Single combined model with two inputs:
      • "audio_waveform" : [1, 48000]  — raw 16 kHz mono float32 PCM
      • "features"       : [1, 16]     — paralinguistic feature vector
    Output: [1]  — scalar emotion intensity in [0, 1].
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEUTRAL_FALLBACK: float = 0.5
_SAMPLE_RATE: int = 16000
_EXPECTED_SAMPLES: int = _SAMPLE_RATE * 3  # 3-second chunks → 48 000 samples
_N_MFCC: int = 13
_FEATURE_DIM: int = 3 + _N_MFCC  # pitch, rms, zcr + 13 MFCCs = 16


class AudioEmotionPipeline:
    """Singleton CPU-inference pipeline for paralinguistic emotion analysis."""

    _instance: AudioEmotionPipeline | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AudioEmotionPipeline":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst.load_model()
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        from django.conf import settings
        self._model_path: str = settings.AUDIO_MODEL_PATH
        self._session: Any = None

    def load_model(self) -> None:
        import os
        if not os.path.exists(self._model_path):
            logger.warning(
                "AudioEmotionPipeline: model not found at %s — returning fallback scores.",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info("AudioEmotionPipeline loaded: %s", self._model_path)
        except Exception:
            logger.exception("AudioEmotionPipeline: failed to load ONNX model.")

    # ------------------------------------------------------------------
    # Public predict — NEUTRAL_FALLBACK wraps everything (5B.5)
    # ------------------------------------------------------------------

    def predict(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """
        Return 8-class emotion logits for one audio chunk.

        Parameters
        ----------
        audio_bytes : raw WAV bytes (16-bit mono, 16 kHz)
        sample_rate : declared sample rate from the client

        Returns
        -------
        np.ndarray of shape [8,] — raw logits; zeros (neutral) on any exception.
        """
        try:
            return self._predict(audio_bytes, sample_rate)
        except Exception:
            logger.exception("AudioEmotionPipeline.predict failed")
            return np.zeros(8, dtype=np.float32)

    def _predict(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        if self._session is None:
            return np.zeros(8, dtype=np.float32)

        waveform = _decode_audio(audio_bytes, sample_rate)  # [48000,] float32
        features = _extract_features(waveform)               # [16,]    float32

        audio_input = waveform[np.newaxis, :]                # [1, 48000]
        feat_input  = features[np.newaxis, :]                # [1, 16]

        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {input_names[0]: audio_input, input_names[1]: feat_input}

        outputs = self._session.run(None, feed)
        return outputs[0].flatten().astype(np.float32)  # [8,]


# ---------------------------------------------------------------------------
# Audio helpers (5B.2)
# ---------------------------------------------------------------------------

def _decode_audio(audio_bytes: bytes, declared_sr: int) -> np.ndarray:
    """
    Decode WAV bytes to a normalised float32 mono waveform at 16 kHz.
    Pads or truncates to exactly _EXPECTED_SAMPLES.
    """
    import soundfile as sf
    import librosa

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    # Convert stereo to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed.
    if sr != _SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)

    # Zero-mean / unit-variance normalisation — matches train_audio_model.py's _load_audio()
    # and wav2vec2's expected preprocessing (it was pretrained on normalised input).
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

    # Pad or truncate to fixed length.
    if len(audio) < _EXPECTED_SAMPLES:
        audio = np.pad(audio, (0, _EXPECTED_SAMPLES - len(audio)))
    else:
        audio = audio[:_EXPECTED_SAMPLES]

    return audio.astype(np.float32)


def _extract_features(waveform: np.ndarray) -> np.ndarray:
    """
    Extract 16-dimensional paralinguistic feature vector:
      [0]     mean pitch (Hz, normalised by 400)
      [1]     mean RMS energy
      [2]     mean zero-crossing rate
      [3-15]  mean of 13 MFCCs
    """
    import librosa

    # Pitch via YIN algorithm.
    f0 = librosa.yin(waveform, fmin=80.0, fmax=400.0, sr=_SAMPLE_RATE)
    voiced = f0[f0 > 0]
    mean_pitch = float(np.mean(voiced) / 400.0) if len(voiced) > 0 else 0.0

    # RMS energy.
    rms = librosa.feature.rms(y=waveform)[0]
    mean_rms = float(np.mean(rms))

    # Zero-crossing rate.
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    mean_zcr = float(np.mean(zcr))

    # MFCCs — 13 coefficients, mean over time.
    mfccs = librosa.feature.mfcc(y=waveform, sr=_SAMPLE_RATE, n_mfcc=_N_MFCC)
    mfcc_means = mfccs.mean(axis=1).tolist()  # [13,]

    features = np.array([mean_pitch, mean_rms, mean_zcr] + mfcc_means, dtype=np.float32)
    return features
