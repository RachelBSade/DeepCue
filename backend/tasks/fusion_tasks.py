"""
Celery task: run_fusion (4.4)

Reads the latest per-modality scores from Redis, runs the FusionPipeline,
writes the EmotionFrame document to MongoDB, and pushes an emotion_result
message to the browser via the Channels group.
"""
from __future__ import annotations

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
_KEY_VIDEO  = "deepcue:scores:{session_id}:video"
_KEY_AUDIO  = "deepcue:scores:{session_id}:audio"
_KEY_TEXT   = "deepcue:scores:{session_id}:text"

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

    video_score = _read_score(r, _KEY_VIDEO.format(session_id=session_id))
    audio_score = _read_score(r, _KEY_AUDIO.format(session_id=session_id))
    text_score  = _read_score(r, _KEY_TEXT.format(session_id=session_id))

    pipeline = FusionPipeline.get_instance()
    fusion_scores: dict[str, float] = pipeline.predict(video_score, audio_score, text_score)
    dominant_emotion: str = max(fusion_scores, key=fusion_scores.get)

    # --- Persist EmotionFrame to MongoDB -----------------------------------
    db = get_sync_db()
    frame_doc: EmotionFrame = {
        "session_id":     session_id,
        "timestamp":      timestamp,
        "frame_index":    frame_index,
        "video_score":    video_score,
        "audio_score":    audio_score,
        "text_score":     text_score,
        "fusion_scores":  fusion_scores,
        "dominant_emotion": dominant_emotion,
    }
    db.emotion_frames.insert_one(frame_doc)

    # Increment frame_count and update dominant_emotion on the session doc.
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

    logger.debug(
        "fusion session=%s frame=%d dominant=%s scores=%s",
        session_id, frame_index, dominant_emotion,
        {k: f"{v:.3f}" for k, v in fusion_scores.items()},
    )

    # --- Push to browser via Channels --------------------------------------
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
        },
    )


def _read_score(r: redis.Redis, key: str) -> float:
    """Read a modality score from Redis; return NEUTRAL_FALLBACK if missing."""
    val = r.get(key)
    if val is None:
        return NEUTRAL_FALLBACK
    try:
        return float(val)
    except ValueError:
        return NEUTRAL_FALLBACK


def _get_redis() -> redis.Redis:
    """Open a Redis client against the Celery broker, used as the modality-score cache."""
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
