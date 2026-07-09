"""
Celery task: run_fusion (4.4)

Reads the latest per-modality scores from Redis, runs the FusionPipeline,
writes the EmotionFrame document to MongoDB, and pushes an emotion_result
message to the browser via the Channels group.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import redis
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.conf import settings

from db.mongo_client import get_sync_db
from db.schemas import EmotionFrame

logger = logging.getLogger(__name__)

# Redis key patterns written by the three modality tasks (4.5).
_KEY_VIDEO        = "deepcue:scores:{session_id}:video"
_KEY_AUDIO        = "deepcue:scores:{session_id}:audio"
_KEY_TEXT         = "deepcue:scores:{session_id}:text"
_KEY_SPEECH_RATE  = "deepcue:scores:{session_id}:speech_rate"

NEUTRAL_FALLBACK = 0.5


@shared_task(name="tasks.fusion_tasks.run_fusion", bind=True)
def run_fusion(
    self,
    session_id: str,
    frame_index: int,
    timestamp: float,
    group_name: str,
) -> None:
    """
    Fuse the three latest modality scores into an 8-class emotion distribution.

    Steps:
      1. Read video / audio / text scores from Redis (fall back to 0.5 if absent).
      2. Run FusionPipeline.predict() → dict of 8 emotion scores.
      3. Write an EmotionFrame document to MongoDB.
      4. Push emotion_result to the browser via Channels group.
      5. Increment the session frame_count in MongoDB.
    """
    from apps.inference.fusion_pipeline import FusionPipeline

    r = _get_redis()

    video_logits = _read_logits(r, _KEY_VIDEO.format(session_id=session_id))
    audio_logits = _read_logits(r, _KEY_AUDIO.format(session_id=session_id))
    text_score   = _read_score(r, _KEY_TEXT.format(session_id=session_id))
    speech_rate_wpm = _read_optional_score(r, _KEY_SPEECH_RATE.format(session_id=session_id))

    pipeline = FusionPipeline.get_instance()
    fusion_scores: dict[str, float] = pipeline.predict(video_logits, audio_logits, text_score)
    fusion_scores = pipeline.apply_speech_rate(fusion_scores, speech_rate_wpm)
    dominant_emotion: str = max(fusion_scores, key=fusion_scores.get)

    # --- Persist EmotionFrame to MongoDB -----------------------------------
    try:
        db = get_sync_db()
        frame_doc: EmotionFrame = {
            "session_id":     session_id,
            "timestamp":      timestamp,
            "frame_index":    frame_index,
            "video_score":    float(max(video_logits)),
            "audio_score":    float(max(audio_logits)),
            "text_score":     text_score,
            "fusion_scores":  fusion_scores,
            "dominant_emotion": dominant_emotion,
            "speech_rate_wpm": speech_rate_wpm,
        }
        db.emotion_frames.insert_one(frame_doc)
        db.interview_sessions.update_one(
            {"session_id": session_id},
            {
                "$inc": {"frame_count": 1},
                "$set": {
                    "dominant_emotion": dominant_emotion,
                    "updated_at": datetime.now(timezone.utc),
                },
            },
        )
    except Exception:
        logger.warning("fusion: MongoDB write failed (session=%s) — continuing without persistence.", session_id)

    logger.debug(
        "fusion session=%s frame=%d dominant=%s scores=%s",
        session_id, frame_index, dominant_emotion,
        {k: f"{v:.3f}" for k, v in fusion_scores.items()},
    )

    # --- Push to browser via Channels --------------------------------------
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                "type":             "emotion_result",
                "session_id":       session_id,
                "timestamp":        timestamp,
                "frame_index":      frame_index,
                "scores":           fusion_scores,
                "dominant_emotion": dominant_emotion,
                "speech_rate_wpm":  speech_rate_wpm,
            },
        )
        logger.info("fusion: pushed emotion_result to group=%s dominant=%s", group_name, dominant_emotion)
    except Exception:
        logger.exception("fusion: group_send failed session=%s", session_id)


_NEUTRAL_LOGITS: list[float] = [0.0] * 8


def _read_logits(r: redis.Redis, key: str) -> list[float]:
    """Read an 8-element logit array stored as JSON; return neutral zeros if missing."""
    val = r.get(key)
    if val is None:
        return list(_NEUTRAL_LOGITS)
    try:
        logits = json.loads(val)
        if isinstance(logits, list) and len(logits) == 8:
            return [float(x) for x in logits]
        return list(_NEUTRAL_LOGITS)
    except (json.JSONDecodeError, ValueError):
        return list(_NEUTRAL_LOGITS)


def _read_score(r: redis.Redis, key: str) -> float:
    """Read a scalar modality score from Redis; return NEUTRAL_FALLBACK if missing."""
    val = r.get(key)
    if val is None:
        return NEUTRAL_FALLBACK
    try:
        return float(val)
    except ValueError:
        return NEUTRAL_FALLBACK


def _read_optional_score(r: redis.Redis, key: str) -> float | None:
    """Read an optional score (e.g. speech rate) from Redis; None if absent/invalid."""
    val = r.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _get_redis() -> redis.Redis:
    """Open a Redis client against the Celery broker, used as the modality-score cache."""
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
