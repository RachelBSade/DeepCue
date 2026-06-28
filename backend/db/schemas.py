"""MongoDB document schemas as TypedDicts."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TypedDict


class EmotionScores(TypedDict):
    neutral: float
    confident: float
    anxious: float
    happy: float
    sad: float
    angry: float
    surprised: float
    uncertain: float


class EmotionFrame(TypedDict):
    session_id: str
    timestamp: float
    frame_index: int
    video_score: float
    audio_score: float
    text_score: float
    fusion_scores: EmotionScores
    dominant_emotion: str


class TranscriptSegment(TypedDict):
    session_id: str
    timestamp: float
    segment_index: int
    text: str
    language: str
    duration_seconds: float


class InterviewSession(TypedDict):
    session_id: str
    created_at: datetime
    updated_at: datetime
    status: str
    candidate_name: str
    candidate_email: Optional[str]
    duration_seconds: float
    frame_count: int
    dominant_emotion: str
    report_url: Optional[str]
