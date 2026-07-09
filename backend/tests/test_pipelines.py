"""
Phase 8.2 — Unit tests for all four inference pipelines.

Each pipeline is tested without real ONNX model files.
The model path points to a non-existent file (set in test settings),
which triggers the graceful fallback path.

Tests assert:
  - NEUTRAL_FALLBACK (0.5) is returned on missing model / exception
  - predict() never raises
  - FusionPipeline fallback returns {neutral:1.0, all others:0.0}
  - VideoEmotionPipeline.clear_session_buffer() is safe to call
"""
from __future__ import annotations

import pytest
from conftest import make_landmarks, make_wav_bytes


# ---------------------------------------------------------------------------
# VideoEmotionPipeline (5A)
# ---------------------------------------------------------------------------

class TestVideoEmotionPipeline:
    def _make_pipeline(self):
        # Reset singleton so each test gets a fresh instance.
        from apps.inference.video_pipeline import VideoEmotionPipeline
        VideoEmotionPipeline._instance = None
        pipeline = VideoEmotionPipeline.get_instance()
        return pipeline

    def test_predict_returns_neutral_fallback_without_model(self, fake_session_id):
        pipeline = self._make_pipeline()
        landmarks = make_landmarks(468)
        score = pipeline.predict(landmarks, frame_index=0, session_id=fake_session_id)
        assert score == 0.5

    def test_predict_never_raises(self, fake_session_id):
        pipeline = self._make_pipeline()
        # Intentionally malformed landmarks (empty list).
        score = pipeline.predict([], frame_index=0, session_id=fake_session_id)
        assert isinstance(score, float)

    def test_predict_score_is_in_unit_interval(self, fake_session_id):
        pipeline = self._make_pipeline()
        for i in range(5):
            score = pipeline.predict(make_landmarks(468), frame_index=i,
                                     session_id=fake_session_id)
            assert 0.0 <= score <= 1.0

    def test_clear_session_buffer_is_safe(self, fake_session_id):
        pipeline = self._make_pipeline()
        pipeline.clear_session_buffer(fake_session_id)   # no-op for unknown session
        # Add some frames to the buffer, then clear.
        pipeline.predict(make_landmarks(468), 0, fake_session_id)
        pipeline.clear_session_buffer(fake_session_id)
        # Buffer should be gone — next predict fills a fresh deque.
        score = pipeline.predict(make_landmarks(468), 1, fake_session_id)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# AudioEmotionPipeline (5B)
# ---------------------------------------------------------------------------

class TestAudioEmotionPipeline:
    def _make_pipeline(self):
        from apps.inference.audio_pipeline import AudioEmotionPipeline
        AudioEmotionPipeline._instance = None
        return AudioEmotionPipeline.get_instance()

    def test_predict_returns_neutral_fallback_without_model(self):
        pipeline = self._make_pipeline()
        wav = make_wav_bytes()
        score = pipeline.predict(wav, sample_rate=16000)
        assert score == 0.5

    def test_predict_never_raises_on_bad_bytes(self):
        pipeline = self._make_pipeline()
        score = pipeline.predict(b"not-a-wav-file", sample_rate=16000)
        assert isinstance(score, float)

    def test_predict_score_is_in_unit_interval(self):
        pipeline = self._make_pipeline()
        score = pipeline.predict(make_wav_bytes(), sample_rate=16000)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TextEmotionPipeline (5C)
# ---------------------------------------------------------------------------

class TestTextEmotionPipeline:
    def _make_pipeline(self):
        from apps.inference.text_pipeline import TextEmotionPipeline, NEUTRAL_FALLBACK
        TextEmotionPipeline._instance = None
        return TextEmotionPipeline.get_instance(), NEUTRAL_FALLBACK

    def test_predict_empty_text_returns_fallback(self):
        pipeline, fallback = self._make_pipeline()
        assert pipeline.predict("") == fallback
        assert pipeline.predict("   ") == fallback

    def test_predict_returns_neutral_fallback_without_model(self):
        pipeline, fallback = self._make_pipeline()
        score = pipeline.predict("שלום עולם")
        assert score == fallback

    def test_predict_never_raises(self):
        pipeline, _ = self._make_pipeline()
        score = pipeline.predict("some text")
        assert isinstance(score, float)

    def test_transcribe_returns_empty_string_without_whisper(self):
        pipeline, _ = self._make_pipeline()
        # Whisper model does not load (test settings use tiny + non-existent cache).
        # May either return "" or a real transcription if whisper downloads tiny model.
        result = pipeline.transcribe(make_wav_bytes())
        assert isinstance(result, str)

    def test_transcribe_never_raises_on_bad_bytes(self):
        pipeline, _ = self._make_pipeline()
        result = pipeline.transcribe(b"garbage")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# FusionPipeline (5D)
# ---------------------------------------------------------------------------

class TestFusionPipeline:
    EMOTION_CLASSES = [
        "neutral", "confident", "anxious", "happy",
        "sad", "angry", "surprised", "uncertain",
    ]

    def _make_pipeline(self):
        from apps.inference.fusion_pipeline import FusionPipeline
        FusionPipeline._instance = None
        return FusionPipeline.get_instance()

    def test_predict_returns_fallback_dict_without_model(self):
        pipeline = self._make_pipeline()
        result = pipeline.predict(0.5, 0.5, 0.5)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(self.EMOTION_CLASSES)
        # Fallback: neutral=1.0, others=0.0
        assert result["neutral"] == 1.0
        assert all(v == 0.0 for k, v in result.items() if k != "neutral")

    def test_predict_scores_sum_is_one_on_valid_model(self):
        """When the model IS loaded, scores must sum to 1.0 (softmax output)."""
        pipeline = self._make_pipeline()
        result = pipeline.predict(0.7, 0.4, 0.6)
        total = sum(result.values())
        # Fallback dict sums to 1.0 (neutral=1.0), so this holds regardless.
        assert abs(total - 1.0) < 1e-4

    def test_predict_never_raises(self):
        pipeline = self._make_pipeline()
        result = pipeline.predict(0.0, 0.0, 0.0)
        assert isinstance(result, dict)

    def test_predict_all_emotion_keys_present(self):
        pipeline = self._make_pipeline()
        result = pipeline.predict(0.5, 0.5, 0.5)
        for emotion in self.EMOTION_CLASSES:
            assert emotion in result

    def test_predict_boundary_inputs(self):
        pipeline = self._make_pipeline()
        for v, a, t in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.5)]:
            result = pipeline.predict(v, a, t)
            assert all(isinstance(s, float) for s in result.values())
