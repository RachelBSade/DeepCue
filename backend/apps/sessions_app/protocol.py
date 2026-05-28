"""
DeepCue WebSocket Message Protocol (2.1)

All messages are JSON objects with a required `type` field.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INBOUND  (browser → server)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
session_start
  { "type": "session_start", "candidate_name": "<str>" }
  Sent immediately after WebSocket connection is established.

video_frame
  { "type": "video_frame",
    "session_id":  "<uuid>",
    "frame_index": <int>,
    "timestamp":   <float>,        # Unix epoch, UTC
    "landmarks":   [[x,y,z], ...]  # 468 MediaPipe Face Mesh points }

audio_chunk
  { "type": "audio_chunk",
    "session_id":  "<uuid>",
    "chunk_index": <int>,
    "timestamp":   <float>,
    "audio_data":  "<base64>",     # WAV/PCM bytes encoded as base64
    "sample_rate": <int> }         # typically 16000

session_end
  { "type": "session_end", "session_id": "<uuid>" }

error
  { "type": "error", "message": "<str>", "session_id": "<uuid|null>" }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTBOUND  (server → browser)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
session_started
  { "type": "session_started", "session_id": "<uuid>" }

emotion_result
  { "type": "emotion_result",
    "session_id":        "<uuid>",
    "timestamp":         <float>,
    "frame_index":       <int>,
    "scores":            { "neutral": 0.0, "confident": 0.0, ... },
    "dominant_emotion":  "<str>" }

transcript_update
  { "type": "transcript_update",
    "session_id":    "<uuid>",
    "text":          "<str>",     # Hebrew transcript from Whisper
    "segment_index": <int>,
    "timestamp":     <float> }

session_ended
  { "type": "session_ended",
    "session_id": "<uuid>",
    "report_url": "<str|null>" }  # populated once report task completes

error
  { "type": "error", "message": "<str>" }

interviewer_audio  [STUBBED — Phase 10, Bidirectional AI Interviewer]
  { "type": "interviewer_audio", ... }
"""
from __future__ import annotations

from typing import Optional, TypedDict

# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

# Inbound
SESSION_START: str = "session_start"
VIDEO_FRAME: str   = "video_frame"
AUDIO_CHUNK: str   = "audio_chunk"
SESSION_END: str   = "session_end"
ERROR: str         = "error"

# Outbound
SESSION_STARTED:    str = "session_started"
EMOTION_RESULT:     str = "emotion_result"
TRANSCRIPT_UPDATE:  str = "transcript_update"
SESSION_ENDED:      str = "session_ended"
INTERVIEWER_AUDIO:  str = "interviewer_audio"  # stubbed

INBOUND_TYPES: frozenset[str] = frozenset(
    {SESSION_START, VIDEO_FRAME, AUDIO_CHUNK, SESSION_END, ERROR}
)

# ---------------------------------------------------------------------------
# Inbound payload schemas (browser → server)
# ---------------------------------------------------------------------------


class LandmarkPoint(TypedDict):
    x: float
    y: float
    z: float


class SessionStartMessage(TypedDict):
    type: str
    candidate_name: str


class VideoFrameMessage(TypedDict):
    type: str
    session_id: str
    frame_index: int
    timestamp: float
    landmarks: list[LandmarkPoint]  # exactly 468 points


class AudioChunkMessage(TypedDict):
    type: str
    session_id: str
    chunk_index: int
    timestamp: float
    audio_data: str   # base64-encoded WAV/PCM
    sample_rate: int


class SessionEndMessage(TypedDict):
    type: str
    session_id: str


class ClientErrorMessage(TypedDict):
    type: str
    message: str
    session_id: Optional[str]


# ---------------------------------------------------------------------------
# Outbound payload schemas (server → browser)
# ---------------------------------------------------------------------------


class SessionStartedMessage(TypedDict):
    type: str
    session_id: str


class EmotionResultMessage(TypedDict):
    type: str
    session_id: str
    timestamp: float
    frame_index: int
    scores: dict[str, float]
    dominant_emotion: str


class TranscriptUpdateMessage(TypedDict):
    type: str
    session_id: str
    text: str
    segment_index: int
    timestamp: float


class SessionEndedMessage(TypedDict):
    type: str
    session_id: str
    report_url: Optional[str]


class ServerErrorMessage(TypedDict):
    type: str
    message: str
