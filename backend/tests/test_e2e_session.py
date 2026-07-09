"""
Phase 3.8 — End-to-end session flow test.

Simulates a complete browser session over an in-memory channel layer:
  1. Connect to the WebSocket endpoint.
  2. Send session_start → assert session_started.
  3. Send video_frame (468 MediaPipe landmarks) → no error returned.
  4. Send audio_chunk (base64 WAV) → no error returned.
  5. Send session_end → assert session_ended.

No real Redis, MongoDB, or ML models are required — all external calls are
patched.  Celery tasks run in eager mode (CELERY_TASK_ALWAYS_EAGER=True).
"""
from __future__ import annotations

import base64
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from channels.testing import WebsocketCommunicator

from conftest import make_landmarks, make_wav_bytes


def _make_application():
    from channels.routing import ProtocolTypeRouter, URLRouter
    from apps.sessions_app.routing import websocket_urlpatterns
    return ProtocolTypeRouter({"websocket": URLRouter(websocket_urlpatterns)})


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_full_session_e2e():
    """
    Browser-equivalent end-to-end session:
    connect → session_start → video_frame → audio_chunk → session_end.

    Verifies that Django Channels accepts the connection, MediaPipe
    landmark payloads are dispatched without error, audio chunks are
    accepted, and session_end returns a valid session_ended message.
    """
    app = _make_application()
    session_id = str(uuid.uuid4())

    with patch("apps.sessions_app.consumers.async_db") as mock_db, \
         patch("apps.sessions_app.consumers._dispatch") as mock_dispatch:

        mock_db.interview_sessions.insert_one = AsyncMock(return_value=None)
        mock_db.interview_sessions.update_one = AsyncMock(return_value=None)
        mock_dispatch.return_value = None

        comm = WebsocketCommunicator(app, f"/ws/interview/{session_id}/")
        connected, _ = await comm.connect()
        assert connected, "WebSocket connection rejected"

        # ── 1. session_start ────────────────────────────────────────────────
        await comm.send_json_to({
            "type": "session_start",
            "candidate_name": "ישראל ישראלי",
        })
        resp = await comm.receive_json_from(timeout=3)
        assert resp["type"] == "session_started", f"Expected session_started, got: {resp}"
        assert resp["session_id"] == session_id

        # ── 2. video_frame (468 MediaPipe landmarks) ─────────────────────────
        await comm.send_json_to({
            "type":        "video_frame",
            "frame_index": 0,
            "timestamp":   0.033,
            "landmarks":   make_landmarks(468),
        })
        # The consumer dispatches to Celery and does NOT send a WS response
        # for individual frames — check that no error message arrives.
        assert await comm.receive_nothing(timeout=0.1), \
            "Unexpected message received after video_frame"

        # ── 3. audio_chunk (base64 WAV) ──────────────────────────────────────
        audio_b64 = base64.b64encode(make_wav_bytes(duration_seconds=1.0)).decode()
        await comm.send_json_to({
            "type":        "audio_chunk",
            "chunk_index": 0,
            "timestamp":   0.0,
            "audio_data":  audio_b64,
            "sample_rate": 16000,
        })
        assert await comm.receive_nothing(timeout=0.1), \
            "Unexpected message received after audio_chunk"

        # ── 4. session_end ───────────────────────────────────────────────────
        await comm.send_json_to({"type": "session_end"})
        resp = await comm.receive_json_from(timeout=3)
        assert resp["type"] == "session_ended", f"Expected session_ended, got: {resp}"
        assert resp["session_id"] == session_id

        await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_multiple_video_frames_do_not_error():
    """Streaming multiple consecutive video frames must not produce error responses."""
    app = _make_application()
    session_id = str(uuid.uuid4())

    with patch("apps.sessions_app.consumers.async_db") as mock_db, \
         patch("apps.sessions_app.consumers._dispatch"):

        mock_db.interview_sessions.insert_one = AsyncMock(return_value=None)

        comm = WebsocketCommunicator(app, f"/ws/interview/{session_id}/")
        connected, _ = await comm.connect()
        assert connected

        await comm.send_json_to({"type": "session_start", "candidate_name": "Test"})
        await comm.receive_json_from(timeout=3)  # session_started

        for i in range(5):
            await comm.send_json_to({
                "type":        "video_frame",
                "frame_index": i,
                "timestamp":   i * 0.033,
                "landmarks":   make_landmarks(468),
            })

        assert await comm.receive_nothing(timeout=0.2), \
            "Unexpected error message after streaming video frames"

        await comm.disconnect()
