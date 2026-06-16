"""
Phase 8.3 — Celery task unit tests (eager mode).

Tasks run synchronously via CELERY_TASK_ALWAYS_EAGER = True in test settings.
MongoDB and Channels calls are patched out with unittest.mock so tests need
no external services.

Tests assert:
  - Redis cache is written with the correct key pattern and float value
  - Fusion task reads from Redis and writes an EmotionFrame document
  - Report task calls InterviewReportGenerator.generate() and store_report()
"""
from __future__ import annotations

import base64
import struct
import wave
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import make_landmarks, make_wav_bytes

SESSION_ID  = "aaaaaaaa-bbbb-4000-a000-cccccccccccc"
GROUP_NAME  = f"interview_{SESSION_ID}"
REDIS_KEY_V = f"deepcue:scores:{SESSION_ID}:video"
REDIS_KEY_A = f"deepcue:scores:{SESSION_ID}:audio"
REDIS_KEY_T = f"deepcue:scores:{SESSION_ID}:text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_wav() -> str:
    return base64.b64encode(make_wav_bytes()).decode()


# ---------------------------------------------------------------------------
# Video task (4.1)
# ---------------------------------------------------------------------------

class TestProcessVideoFrame:
    @patch("tasks.video_tasks._get_redis")
    @patch("tasks.video_tasks.run_fusion")
    def test_caches_score_in_redis(self, mock_fusion, mock_get_redis):
        redis_mock = MagicMock()
        mock_get_redis.return_value = redis_mock
        mock_fusion.apply_async = MagicMock()

        from tasks.video_tasks import process_video_frame
        process_video_frame.apply(kwargs={
            "session_id":  SESSION_ID,
            "frame_index": 0,
            "timestamp":   1.0,
            "landmarks":   make_landmarks(468),
            "group_name":  GROUP_NAME,
        })

        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args
        key, ttl, value = call_args.args
        assert key == REDIS_KEY_V
        assert ttl == 60
        assert float(value) == 0.5  # NEUTRAL_FALLBACK — no real model

    @patch("tasks.video_tasks._get_redis")
    @patch("tasks.video_tasks.run_fusion")
    def test_triggers_fusion_task(self, mock_fusion, mock_get_redis):
        mock_get_redis.return_value = MagicMock()
        mock_fusion.apply_async = MagicMock()

        from tasks.video_tasks import process_video_frame
        process_video_frame.apply(kwargs={
            "session_id":  SESSION_ID,
            "frame_index": 1,
            "timestamp":   2.0,
            "landmarks":   make_landmarks(468),
            "group_name":  GROUP_NAME,
        })

        mock_fusion.apply_async.assert_called_once()
        kwargs = mock_fusion.apply_async.call_args.kwargs
        assert kwargs["queue"] == "fusion_queue"


# ---------------------------------------------------------------------------
# Audio task (4.2)
# ---------------------------------------------------------------------------

class TestProcessAudioChunk:
    @patch("tasks.audio_tasks._get_redis")
    def test_caches_score_in_redis(self, mock_get_redis):
        redis_mock = MagicMock()
        mock_get_redis.return_value = redis_mock

        from tasks.audio_tasks import process_audio_chunk
        process_audio_chunk.apply(kwargs={
            "session_id":  SESSION_ID,
            "chunk_index": 0,
            "timestamp":   1.0,
            "audio_data":  _b64_wav(),
            "sample_rate": 16000,
            "group_name":  GROUP_NAME,
        })

        redis_mock.setex.assert_called_once()
        key, ttl, value = redis_mock.setex.call_args.args
        assert key == REDIS_KEY_A
        assert float(value) == 0.5

    @patch("tasks.audio_tasks._get_redis")
    def test_invalid_base64_does_not_raise(self, mock_get_redis):
        mock_get_redis.return_value = MagicMock()
        from tasks.audio_tasks import process_audio_chunk
        # Should not propagate — task catches exceptions internally.
        process_audio_chunk.apply(kwargs={
            "session_id":  SESSION_ID,
            "chunk_index": 0,
            "timestamp":   1.0,
            "audio_data":  "not-valid-base64!!!",
            "sample_rate": 16000,
            "group_name":  GROUP_NAME,
        })


# ---------------------------------------------------------------------------
# Text task (4.3)
# ---------------------------------------------------------------------------

class TestProcessTranscriptSegment:
    @patch("tasks.text_tasks._get_redis")
    @patch("tasks.text_tasks.async_to_sync")
    def test_caches_score_in_redis(self, mock_a2s, mock_get_redis):
        redis_mock = MagicMock()
        mock_get_redis.return_value = redis_mock
        mock_a2s.return_value = MagicMock(return_value=None)

        from tasks.text_tasks import process_transcript_segment
        process_transcript_segment.apply(kwargs={
            "session_id":  SESSION_ID,
            "chunk_index": 0,
            "timestamp":   1.0,
            "audio_data":  _b64_wav(),
            "sample_rate": 16000,
            "group_name":  GROUP_NAME,
        })

        redis_mock.setex.assert_called_once()
        key, ttl, value = redis_mock.setex.call_args.args
        assert key == REDIS_KEY_T
        assert float(value) == 0.5


# ---------------------------------------------------------------------------
# Fusion task (4.4)
# ---------------------------------------------------------------------------

class TestRunFusion:
    def _make_redis(self, video=0.6, audio=0.4, text=0.5):
        """Return a Redis mock pre-loaded with the given modality scores."""
        r = MagicMock()
        def _get(key):
            if "video" in key:
                return str(video)
            if "audio" in key:
                return str(audio)
            if "text" in key:
                return str(text)
            return None
        r.get.side_effect = _get
        return r

    @patch("tasks.fusion_tasks._get_redis")
    @patch("tasks.fusion_tasks.get_channel_layer")
    @patch("tasks.fusion_tasks.get_sync_db")
    def test_writes_emotion_frame_to_mongo(self, mock_db, mock_cl, mock_redis):
        mock_redis.return_value = self._make_redis()
        db_mock = MagicMock()
        mock_db.return_value = db_mock
        channel_mock = MagicMock()
        channel_mock.group_send = AsyncMock()
        mock_cl.return_value = channel_mock

        from tasks.fusion_tasks import run_fusion
        run_fusion.apply(kwargs={
            "session_id":  SESSION_ID,
            "frame_index": 0,
            "timestamp":   1.0,
            "group_name":  GROUP_NAME,
        })

        db_mock.emotion_frames.insert_one.assert_called_once()
        doc = db_mock.emotion_frames.insert_one.call_args.args[0]
        assert doc["session_id"] == SESSION_ID
        assert "fusion_scores" in doc
        assert "dominant_emotion" in doc

    @patch("tasks.fusion_tasks._get_redis")
    @patch("tasks.fusion_tasks.get_channel_layer")
    @patch("tasks.fusion_tasks.get_sync_db")
    def test_sends_emotion_result_to_channel_group(self, mock_db, mock_cl, mock_redis):
        mock_redis.return_value = self._make_redis()
        mock_db.return_value = MagicMock()
        channel_mock = MagicMock()
        mock_cl.return_value = channel_mock

        from tasks.fusion_tasks import run_fusion
        with patch("tasks.fusion_tasks.async_to_sync") as mock_a2s:
            group_send = MagicMock()
            mock_a2s.return_value = group_send

            run_fusion.apply(kwargs={
                "session_id":  SESSION_ID,
                "frame_index": 0,
                "timestamp":   1.0,
                "group_name":  GROUP_NAME,
            })

            group_send.assert_called_once()
            call_args = group_send.call_args.args
            assert call_args[0] == GROUP_NAME
            assert call_args[1]["type"] == "emotion_result"

    @patch("tasks.fusion_tasks._get_redis")
    @patch("tasks.fusion_tasks.get_channel_layer")
    @patch("tasks.fusion_tasks.get_sync_db")
    def test_missing_redis_keys_use_fallback(self, mock_db, mock_cl, mock_redis):
        r = MagicMock()
        r.get.return_value = None   # All keys missing → all fallback to 0.5
        mock_redis.return_value = r
        mock_db.return_value = MagicMock()
        channel_mock = MagicMock()
        channel_mock.group_send = AsyncMock()
        mock_cl.return_value = channel_mock

        from tasks.fusion_tasks import run_fusion
        with patch("tasks.fusion_tasks.async_to_sync", return_value=MagicMock()):
            run_fusion.apply(kwargs={
                "session_id":  SESSION_ID,
                "frame_index": 0,
                "timestamp":   1.0,
                "group_name":  GROUP_NAME,
            })
        # No exception should have been raised.


# ---------------------------------------------------------------------------
# Report task (4.7)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    @patch("tasks.report_tasks.get_sync_db")
    @patch("tasks.report_tasks.get_channel_layer")
    @patch("tasks.report_tasks.store_report")
    def test_calls_generator_and_storage(self, mock_store, mock_cl, mock_db):
        mock_store.return_value = f"/api/report/{SESSION_ID}/"
        mock_cl.return_value = MagicMock()

        fake_session = {
            "session_id":       SESSION_ID,
            "candidate_name":   "Test User",
            "created_at":       "2026-01-01T00:00:00Z",
            "duration_seconds": 120.0,
            "frame_count":      10,
            "dominant_emotion": "neutral",
        }
        db_mock = MagicMock()
        db_mock.interview_sessions.find_one.return_value = fake_session
        db_mock.emotion_frames.find.return_value = MagicMock(
            __iter__=lambda s: iter([]),
            sort=lambda *a, **k: iter([]),
        )
        db_mock.transcript_segments.find.return_value = MagicMock(
            __iter__=lambda s: iter([]),
            sort=lambda *a, **k: iter([]),
        )
        mock_db.return_value = db_mock

        from tasks.report_tasks import generate_report
        with patch("tasks.report_tasks.async_to_sync", return_value=MagicMock()):
            generate_report.apply(kwargs={
                "session_id": SESSION_ID,
                "group_name": GROUP_NAME,
            })

        mock_store.assert_called_once()
        store_args = mock_store.call_args.args
        assert store_args[0] == SESSION_ID
        assert isinstance(store_args[1], bytes)

    @patch("tasks.report_tasks.get_sync_db")
    def test_missing_session_does_not_raise(self, mock_db):
        db_mock = MagicMock()
        db_mock.interview_sessions.find_one.return_value = None
        mock_db.return_value = db_mock

        from tasks.report_tasks import generate_report
        generate_report.apply(kwargs={
            "session_id": SESSION_ID,
            "group_name": GROUP_NAME,
        })
        # Should log an error and return cleanly.
