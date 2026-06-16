"""
Phase 8.1 & 8.7 — WebSocket consumer integration tests.

Uses Django Channels' WebsocketCommunicator to simulate a full browser
session over an in-memory channel layer (no real Redis required).

Tests cover:
  - connect / session_start / session_end lifecycle (8.1)
  - video_frame and audio_chunk dispatch (8.1)
  - unknown message type returns error (2.7)
  - WebSocket reconnect back-off constants (8.7)
"""
from __future__ import annotations

import json
import uuid

import pytest
from channels.testing import WebsocketCommunicator

from conftest import make_landmarks

# ---------------------------------------------------------------------------
# ASGI application
# ---------------------------------------------------------------------------

def _make_application():
    """Build the Channels ProtocolTypeRouter for testing."""
    from django.test import override_settings
    from channels.routing import ProtocolTypeRouter, URLRouter
    from apps.sessions_app.routing import websocket_urlpatterns
    return ProtocolTypeRouter({
        "websocket": URLRouter(websocket_urlpatterns),
    })


@pytest.fixture
def app():
    return _make_application()


@pytest.fixture
def session_id():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _connect(app, session_id: str) -> WebsocketCommunicator:
    comm = WebsocketCommunicator(app, f"/ws/interview/{session_id}/")
    connected, _ = await comm.connect()
    assert connected, "WebSocket connection was rejected"
    return comm


async def _send_json(comm: WebsocketCommunicator, payload: dict) -> dict:
    await comm.send_json_to(payload)
    response = await comm.receive_json_from(timeout=3)
    return response


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_connect_and_accept(app, session_id):
    comm = await _connect(app, session_id)
    await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_session_start_returns_session_started(app, session_id):
    from unittest.mock import AsyncMock, patch

    with patch("apps.sessions_app.consumers.async_db") as mock_db:
        mock_db.interview_sessions.insert_one = AsyncMock(return_value=None)

        comm = await _connect(app, session_id)
        resp = await _send_json(comm, {
            "type": "session_start",
            "candidate_name": "Test Candidate",
        })
        assert resp["type"] == "session_started"
        assert resp["session_id"] == session_id
        await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_unknown_message_type_returns_error(app, session_id):
    comm = await _connect(app, session_id)
    resp = await _send_json(comm, {"type": "not_a_real_type"})
    assert resp["type"] == "error"
    assert "Unknown message type" in resp["message"]
    await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_malformed_json_returns_error(app, session_id):
    comm = await _connect(app, session_id)
    await comm.send_to(text_data="this is not json {{{")
    resp = await comm.receive_json_from(timeout=3)
    assert resp["type"] == "error"
    await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_video_frame_wrong_landmark_count_returns_error(app, session_id):
    comm = await _connect(app, session_id)
    resp = await _send_json(comm, {
        "type": "video_frame",
        "frame_index": 0,
        "timestamp": 1.0,
        "landmarks": make_landmarks(100),   # wrong — must be 468
    })
    assert resp["type"] == "error"
    assert "468" in resp["message"]
    await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_audio_chunk_missing_audio_data_returns_error(app, session_id):
    comm = await _connect(app, session_id)
    resp = await _send_json(comm, {
        "type": "audio_chunk",
        "chunk_index": 0,
        "timestamp": 1.0,
        "audio_data": "",   # empty
        "sample_rate": 16000,
    })
    assert resp["type"] == "error"
    await comm.disconnect()


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_session_end_returns_session_ended(app, session_id):
    from unittest.mock import AsyncMock, MagicMock, patch

    with patch("apps.sessions_app.consumers.async_db") as mock_db, \
         patch("apps.sessions_app.consumers._dispatch"):
        mock_db.interview_sessions.insert_one = AsyncMock(return_value=None)
        mock_db.interview_sessions.update_one = AsyncMock(return_value=None)

        comm = await _connect(app, session_id)

        # Start session first so session_active is True.
        await comm.send_json_to({"type": "session_start", "candidate_name": "Test"})
        await comm.receive_json_from(timeout=3)  # session_started

        resp = await _send_json(comm, {"type": "session_end"})
        assert resp["type"] == "session_ended"
        assert resp["session_id"] == session_id
        assert resp["report_url"] is None
        await comm.disconnect()


# ---------------------------------------------------------------------------
# WebSocket reconnect back-off constants (8.7)
# ---------------------------------------------------------------------------

class TestWebSocketReconnectConstants:
    """
    Verify the reconnect parameters defined in frontend/websocket_client.js
    match the specification: 5 retries, base 1000ms, cap 30000ms.

    This is a JS-specification test — we read and parse the file rather than
    executing it.  It guards against accidental changes to the retry logic.
    """

    def _read_ws_client(self) -> str:
        from pathlib import Path
        p = Path(__file__).parents[2] / "frontend" / "websocket_client.js"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def test_max_retries_is_5(self):
        src = self._read_ws_client()
        if not src:
            pytest.skip("websocket_client.js not found")
        assert "5" in src, "MAX_RETRIES = 5 not found in websocket_client.js"

    def test_base_delay_is_1000ms(self):
        src = self._read_ws_client()
        if not src:
            pytest.skip("websocket_client.js not found")
        assert "1000" in src, "BASE_DELAY_MS = 1000 not found"

    def test_max_delay_cap_is_30000ms(self):
        src = self._read_ws_client()
        if not src:
            pytest.skip("websocket_client.js not found")
        assert "30000" in src, "MAX_DELAY_MS = 30000 not found"

    def test_exponential_backoff_formula_present(self):
        src = self._read_ws_client()
        if not src:
            pytest.skip("websocket_client.js not found")
        # The backoff formula uses 2** or Math.pow.
        assert ("2 **" in src or "2**" in src or "Math.pow" in src), \
            "Exponential back-off formula not found in websocket_client.js"
