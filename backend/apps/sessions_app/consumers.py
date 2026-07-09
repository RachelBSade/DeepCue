"""
InterviewConsumer — Django Channels WebSocket consumer (2.2 – 2.7, 2.9)

URL pattern: ws/interview/<session_id>/

Lifecycle
─────────
1. connect()     Accept the WebSocket, join the Channels group for this session.
2. receive()     Deserialise JSON, route to the correct inbound handler.
3. disconnect()  Leave the group; finalise the MongoDB document if still active.

Celery tasks push results back to the browser by calling
  channel_layer.group_send(group_name, {...})
which triggers the matching async handler below (emotion_result,
transcript_update, session_ended).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from celery import current_app as celery_app
from channels.generic.websocket import AsyncWebsocketConsumer
from pydantic import ValidationError

from db.mongo_client import async_db
from db.schemas import InterviewSession

from . import protocol
from .rate_limit import (
    AUDIO_CHUNK_CAPACITY,
    AUDIO_CHUNK_RATE,
    VIDEO_FRAME_CAPACITY,
    VIDEO_FRAME_RATE,
    TokenBucket,
)
from .validation import (
    AudioChunkSchema,
    SessionEndSchema,
    SessionStartSchema,
    VideoFrameSchema,
)

logger = logging.getLogger(__name__)


class InterviewConsumer(AsyncWebsocketConsumer):
    """Async WebSocket consumer managing one DeepCue interview session."""

    # ------------------------------------------------------------------
    # Connection lifecycle (2.2)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Accept the connection and join the session's Channels group."""
        self.session_id: str = self.scope["url_route"]["kwargs"]["session_id"]
        self.group_name: str = f"interview_{self.session_id}"
        self.session_active: bool = False
        self.session_start_time: datetime | None = None

        # Per-connection rate limiters (9.1) — see rate_limit.py for chosen thresholds.
        self._video_bucket = TokenBucket(VIDEO_FRAME_RATE, VIDEO_FRAME_CAPACITY)
        self._audio_bucket = TokenBucket(AUDIO_CHUNK_RATE, AUDIO_CHUNK_CAPACITY)

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        logger.info("WebSocket connected: session=%s", self.session_id)

    async def disconnect(self, close_code: int) -> None:
        """Leave group and finalise session document on unexpected disconnect."""
        logger.info(
            "WebSocket disconnected: session=%s code=%s", self.session_id, close_code
        )
        if self.session_active:
            await self._finalize_session(status="error")

        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data: str) -> None:
        """Deserialise incoming JSON and dispatch to the matching handler."""
        try:
            data: dict[str, Any] = json.loads(text_data)
        except json.JSONDecodeError:
            await self._send_error("Malformed payload: expected JSON.")
            return

        msg_type: str | None = data.get("type")

        if msg_type not in protocol.INBOUND_TYPES:
            await self._send_error(f"Unknown message type: {msg_type!r}.")
            return

        handlers: dict[str, Any] = {
            protocol.SESSION_START: self._handle_session_start,
            protocol.VIDEO_FRAME:   self._handle_video_frame,
            protocol.AUDIO_CHUNK:   self._handle_audio_chunk,
            protocol.SESSION_END:   self._handle_session_end,
            protocol.ERROR:         self._handle_client_error,
        }

        await handlers[msg_type](data)

    # ------------------------------------------------------------------
    # Inbound handlers
    # ------------------------------------------------------------------

    async def _handle_session_start(self, data: dict[str, Any]) -> None:
        """
        Create the MongoDB session document and confirm to the client. (2.3)

        Idempotent: if the session document already exists (e.g. reconnect),
        the insert is skipped and a session_started confirmation is still sent.
        """
        try:
            msg = SessionStartSchema.model_validate(data)
        except ValidationError as exc:
            await self._send_error(f"session_start: invalid payload — {exc.errors()[0]['msg']}")
            return

        candidate_name: str = msg.candidate_name.strip() or "Unknown"
        candidate_email: str | None = msg.candidate_email
        now = datetime.now(timezone.utc)

        session_doc: InterviewSession = {
            "session_id":       self.session_id,
            "created_at":       now,
            "updated_at":       now,
            "status":           "active",
            "candidate_name":   candidate_name,
            "candidate_email":  candidate_email,
            "duration_seconds": 0.0,
            "frame_count":      0,
            "dominant_emotion": "neutral",
            "report_url":       None,
        }

        try:
            await async_db.interview_sessions.insert_one(session_doc)
        except Exception:
            # Document already exists from a prior connect — safe to continue.
            logger.warning("session_start: document already exists for %s", self.session_id)

        self.session_active = True
        self.session_start_time = now

        await self.send(json.dumps({
            "type":       protocol.SESSION_STARTED,
            "session_id": self.session_id,
        }))
        logger.info("Session started: %s candidate=%r", self.session_id, candidate_name)

    async def _handle_video_frame(self, data: dict[str, Any]) -> None:
        """
        Validate the MediaPipe landmark payload and dispatch the video
        Celery task. (2.4)

        Expected: `landmarks` is a list of exactly 468 {x, y, z} dicts.
        Silently dropped (not an error) if the client exceeds the per-connection
        rate limit — the browser sends frames frequently enough that an
        occasional drop is harmless, unlike a flood of error messages back.
        """
        if not self._video_bucket.allow():
            logger.warning("video_frame rate limit exceeded: session=%s", self.session_id)
            return

        try:
            msg = VideoFrameSchema.model_validate(data)
        except ValidationError as exc:
            await self._send_error(f"video_frame: invalid payload — {exc.errors()[0]['msg']}")
            return

        await asyncio.to_thread(
            _dispatch,
            "tasks.video_tasks.process_video_frame",
            {
                "session_id":  self.session_id,
                "frame_index": msg.frame_index,
                "timestamp":   msg.timestamp,
                "landmarks":   [p.model_dump() for p in msg.landmarks],
                "frame_jpeg":  msg.frame_jpeg,
                "group_name":  self.group_name,
            },
            "video_queue",
        )

    async def _handle_audio_chunk(self, data: dict[str, Any]) -> None:
        """
        Validate the base64 audio payload and dispatch both the audio
        pipeline task and the text (Whisper) pipeline task. (2.5)

        Both tasks receive the same payload; they run concurrently on
        the same `audio_queue` worker. Dropped silently if it exceeds the
        per-connection rate limit (see _handle_video_frame for rationale).
        """
        if not self._audio_bucket.allow():
            logger.warning("audio_chunk rate limit exceeded: session=%s", self.session_id)
            return

        try:
            msg = AudioChunkSchema.model_validate(data)
        except ValidationError as exc:
            await self._send_error(f"audio_chunk: invalid payload — {exc.errors()[0]['msg']}")
            return

        payload: dict[str, Any] = {
            "session_id":  self.session_id,
            "chunk_index": msg.chunk_index,
            "timestamp":   msg.timestamp,
            "audio_data":  msg.audio_data,
            "sample_rate": msg.sample_rate,
            "group_name":  self.group_name,
        }

        # Dispatch audio + text tasks concurrently via the thread pool.
        await asyncio.gather(
            asyncio.to_thread(
                _dispatch,
                "tasks.audio_tasks.process_audio_chunk",
                payload,
                "audio_queue",
            ),
            asyncio.to_thread(
                _dispatch,
                "tasks.text_tasks.process_transcript_segment",
                payload,
                "audio_queue",
            ),
        )

    async def _handle_session_end(self, data: dict[str, Any]) -> None:
        """
        Finalise the MongoDB session document and trigger report
        generation. (2.6)

        The browser receives an immediate `session_ended` with
        `report_url: null`. When the report task completes, it pushes
        a second `session_ended` message via the group with the real URL.
        """
        try:
            SessionEndSchema.model_validate(data)
        except ValidationError as exc:
            await self._send_error(f"session_end: invalid payload — {exc.errors()[0]['msg']}")
            return

        await self._finalize_session(status="completed")

        await asyncio.to_thread(
            _dispatch,
            "tasks.report_tasks.generate_report",
            {
                "session_id": self.session_id,
                "group_name": self.group_name,
            },
            "fusion_queue",
        )

        await self.send(json.dumps({
            "type":       protocol.SESSION_ENDED,
            "session_id": self.session_id,
            "report_url": None,
        }))
        logger.info("Session ended: %s", self.session_id)

    async def _handle_client_error(self, data: dict[str, Any]) -> None:
        """
        Handle a client-reported error: log it and finalize the
        session cleanly. (2.7)
        """
        logger.warning(
            "Client error on session %s: %s",
            self.session_id,
            data.get("message", "(no message)"),
        )
        if self.session_active:
            await self._finalize_session(status="error")

    # ------------------------------------------------------------------
    # Outbound group message handlers  (Celery tasks → browser)
    # Method names must match the `type` field in group_send() payloads.
    # ------------------------------------------------------------------

    async def emotion_result(self, event: dict[str, Any]) -> None:
        """Forward fused emotion scores from the fusion task to the browser."""
        await self.send(json.dumps({
            "type":             protocol.EMOTION_RESULT,
            "session_id":       event["session_id"],
            "timestamp":        event["timestamp"],
            "frame_index":      event["frame_index"],
            "scores":           event["scores"],
            "dominant_emotion": event["dominant_emotion"],
            "speech_rate_wpm":  event.get("speech_rate_wpm"),
        }))

    async def transcript_update(self, event: dict[str, Any]) -> None:
        """Forward Whisper Hebrew transcript from the text task to the browser."""
        await self.send(json.dumps({
            "type":          protocol.TRANSCRIPT_UPDATE,
            "session_id":    event["session_id"],
            "text":          event["text"],
            "segment_index": event["segment_index"],
            "timestamp":     event["timestamp"],
        }))

    async def session_ended(self, event: dict[str, Any]) -> None:
        """Forward the final report URL once the report task completes."""
        await self.send(json.dumps({
            "type":       protocol.SESSION_ENDED,
            "session_id": event["session_id"],
            "report_url": event.get("report_url"),
        }))

    async def interviewer_audio(self, event: dict[str, Any]) -> None:
        """[STUBBED — Phase 10] TTS audio stream for the bidirectional AI interviewer."""
        pass  # 2.9

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _finalize_session(self, status: str) -> None:
        """
        Write final status and duration to the MongoDB session document.
        Guards against double-finalization via `session_active` flag.
        """
        if not self.session_active:
            return

        self.session_active = False
        now = datetime.now(timezone.utc)
        duration = (
            (now - self.session_start_time).total_seconds()
            if self.session_start_time
            else 0.0
        )

        try:
            await async_db.interview_sessions.update_one(
                {"session_id": self.session_id},
                {"$set": {
                    "status":           status,
                    "updated_at":       now,
                    "duration_seconds": duration,
                }},
            )
        except Exception:
            logger.exception("Failed to finalize session %s", self.session_id)

    async def _send_error(self, message: str) -> None:
        """Send a typed error message to the browser."""
        logger.warning("Sending error to client session=%s: %s", self.session_id, message)
        await self.send(json.dumps({
            "type":    protocol.ERROR,
            "message": message,
        }))


# ---------------------------------------------------------------------------
# Module-level helper — called via asyncio.to_thread to avoid blocking the
# event loop on the synchronous Redis round-trip of celery send_task.
# ---------------------------------------------------------------------------

def _dispatch(task_name: str, payload: dict[str, Any], queue: str) -> None:
    """Synchronous Celery task dispatch, intended to run in a thread pool."""
    celery_app.send_task(task_name, kwargs=payload, queue=queue)
