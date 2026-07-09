"""
Celery task: process_transcript_segment (4.3)

Receives the same base64 audio chunk as the audio task, runs Whisper
transcription followed by XLM-RoBERTa sentiment scoring, caches the
text score, and pushes a transcript_update message to the browser via
the Channels group.
"""
from __future__ import annotations

import base64
import logging

import redis
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.conf import settings

logger = logging.getLogger(__name__)

_REDIS_KEY        = "deepcue:scores:{session_id}:text"
_SPEECH_RATE_KEY  = "deepcue:scores:{session_id}:speech_rate"
_SCORE_TTL = 60


@shared_task(name="tasks.text_tasks.process_transcript_segment", bind=True)
def process_transcript_segment(
    self,
    session_id: str,
    chunk_index: int,
    timestamp: float,
    audio_data: str,
    sample_rate: int,
    group_name: str,
) -> None:
    """
    Transcribe audio with Whisper and score the Hebrew text with XLM-RoBERTa.

    Steps:
      1. Decode base64 audio bytes.
      2. Transcribe → Hebrew text string via Whisper.
      3. Score text → float via XLM-RoBERTa.
      4. Cache the text score in Redis.
      5. Push transcript_update to the browser via Channels group.
    """
    from apps.inference.text_pipeline import TextEmotionPipeline

    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception:
        logger.exception("text_task: base64 decode failed session=%s chunk=%d", session_id, chunk_index)
        return

    pipeline = TextEmotionPipeline.get_instance()
    text: str   = pipeline.transcribe(audio_bytes)
    score: float = pipeline.predict(text)
    wpm: float  = pipeline.compute_speech_rate(text, audio_bytes)

    # Cache text score (and speech rate, if we got a usable transcript).
    r = _get_redis()
    r.setex(_REDIS_KEY.format(session_id=session_id), _SCORE_TTL, str(score))
    if wpm > 0:
        r.setex(_SPEECH_RATE_KEY.format(session_id=session_id), _SCORE_TTL, str(wpm))

    logger.debug("text_score session=%s chunk=%d score=%.4f text=%r", session_id, chunk_index, score, text[:40])

    # Push Hebrew transcript to browser.
    if text.strip():
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                "type":          "transcript_update",
                "session_id":    session_id,
                "text":          text,
                "segment_index": chunk_index,
                "timestamp":     timestamp,
            },
        )


def _get_redis() -> redis.Redis:
    """Open a Redis client against the Celery broker, used as the modality-score cache."""
    return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
