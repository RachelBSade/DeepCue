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

from tasks.fusion_tasks import run_fusion  # noqa: E402 — after Celery app init

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
    frame_jpeg: str,
    group_name: str,
) -> None:
    """
    Decode the 224×224 JPEG frame, run it through the video emotion pipeline,
    and cache the resulting 8-class logit array in Redis for fusion_tasks.
    """
    from apps.inference.video_pipeline import VideoEmotionPipeline

    pipeline = VideoEmotionPipeline.get_instance()
    logits = pipeline.predict(frame_jpeg, frame_index, session_id)  # np.ndarray [8,]

    r = _get_redis()
    r.setex(_REDIS_KEY.format(session_id=session_id), _SCORE_TTL, json.dumps(logits.tolist()))

    logger.debug("video_logits session=%s frame=%d argmax=%d", session_id, frame_index, int(logits.argmax()))

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
    """Open a Redis client against the Celery broker, used as the modality-score cache."""
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
