"""
Pydantic validation schemas for inbound WebSocket messages. (9.2)

These mirror the TypedDicts in protocol.py (kept for static typing /
documentation) but add real runtime validation — TypedDict performs no
checks at runtime, so malformed payloads previously reached handler code
before failing in less predictable ways (e.g. KeyError, wrong types passed
into Celery tasks).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class LandmarkPointSchema(BaseModel):
    x: float
    y: float
    z: float


class SessionStartSchema(BaseModel):
    type: str
    candidate_name: str = Field(default="Unknown", max_length=200)
    candidate_email: Optional[EmailStr] = None

    @field_validator("candidate_email", mode="before")
    @classmethod
    def _blank_email_to_none(cls, v: object) -> object:
        """Treat an empty string the same as omitting the field entirely."""
        return None if v == "" else v


class VideoFrameSchema(BaseModel):
    # session_id in the body is informational only — the consumer authoritatively
    # uses self.session_id from the URL route, so it's optional here.
    type: str
    session_id: Optional[str] = None
    frame_index: int = Field(ge=0)
    timestamp: float
    landmarks: list[LandmarkPointSchema] = Field(min_length=468, max_length=468)


class AudioChunkSchema(BaseModel):
    type: str
    session_id: Optional[str] = None
    chunk_index: int = Field(ge=0)
    timestamp: float
    audio_data: str = Field(min_length=1)
    sample_rate: int = Field(default=16000, gt=0, le=192000)


class SessionEndSchema(BaseModel):
    type: str
    session_id: Optional[str] = None


class ClientErrorSchema(BaseModel):
    type: str
    message: str
    session_id: Optional[str] = None
