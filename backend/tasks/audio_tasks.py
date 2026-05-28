"""
Celery task: process_audio_chunk (4.2)

Receives a base64-encoded WAV chunk from the InterviewConsumer,
runs it through the AudioEmotionPipeline, and caches the score in Redis.
The fusion task reads this cached score on its next trigger.
"""
from __future__ import annotations

import base64
import logging

import redis
from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)

_REDIS_KEY = "deepcue:scores:{session_id}:audio"
_SCORE_TTL = 60


@shared_task(name="tasks.audio_tasks.process_audio_chunk", bind=True)
def process_audio_chunk(
    self,
    session_id: str,
    chunk_index: int,
    timestamp: float,
    audio_data: str,
    sample_rate: int,
    group_name: str,
) -> None:
    """
    Decode base64 audio, run it through the AudioEmotionPipeline,
    and cache the resulting score in Redis.
    """
    from apps.inference.audio_pipeline import AudioEmotionPipeline

    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception:
        logger.exception("audio_chunk: base64 decode failed session=%s chunk=%d", session_id, chunk_index)
        return

    pipeline = AudioEmotionPipeline.get_instance()
    score: float = pipeline.predict(audio_bytes, sample_rate)

    r = _get_redis()
    r.setex(_REDIS_KEY.format(session_id=session_id), _SCORE_TTL, str(score))

    logger.debug("audio_score session=%s chunk=%d score=%.4f", session_id, chunk_index, score)


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
