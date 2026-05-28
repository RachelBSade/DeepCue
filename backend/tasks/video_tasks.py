"""
Celery task: process_video_frame (4.1)

Receives a MediaPipe landmark payload from the InterviewConsumer,
runs it through the VideoEmotionPipeline, caches the score in Redis,
and pushes the result to the Channels group for the fusion task.
"""
from __future__ import annotations

import json
import logging

import redis
from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)

# Redis key pattern: deepcue:scores:<session_id>:video
_REDIS_KEY = "deepcue:scores:{session_id}:video"
_SCORE_TTL = 60  # seconds — score expires if no new frames arrive


@shared_task(name="tasks.video_tasks.process_video_frame", bind=True)
def process_video_frame(
    self,
    session_id: str,
    frame_index: int,
    timestamp: float,
    landmarks: list[dict],
    group_name: str,
) -> None:
    """
    Process one MediaPipe landmark frame through the video emotion pipeline.

    Steps:
      1. Load (or reuse) the VideoEmotionPipeline singleton.
      2. Run predict() → float score.
      3. Cache the score in Redis keyed by session_id.
      4. Trigger the fusion task to re-evaluate the unified emotion state.
    """
    from apps.inference.video_pipeline import VideoEmotionPipeline
    from tasks.fusion_tasks import run_fusion

    pipeline = VideoEmotionPipeline.get_instance()
    score: float = pipeline.predict(landmarks, frame_index)

    # Cache latest video score for this session.
    r = _get_redis()
    r.setex(_REDIS_KEY.format(session_id=session_id), _SCORE_TTL, str(score))

    logger.debug("video_score session=%s frame=%d score=%.4f", session_id, frame_index, score)

    # Trigger fusion on every video frame — fusion reads all three cached scores.
    run_fusion.apply_async(
        kwargs={
            "session_id": session_id,
            "frame_index": frame_index,
            "timestamp":   timestamp,
            "group_name":  group_name,
        },
        queue="fusion_queue",
    )


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
