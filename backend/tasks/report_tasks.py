"""
Celery task: generate_report (4.7)

Triggered on session_end. Pulls all EmotionFrame and TranscriptSegment
documents from MongoDB, generates a ReportLab PDF, stores it in GridFS,
and pushes the download URL back to the browser via the Channels group.
"""
from __future__ import annotations

import logging

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)


@shared_task(name="tasks.report_tasks.generate_report", bind=True)
def generate_report(
    self,
    session_id: str,
    group_name: str,
) -> None:
    """
    Generate the post-interview PDF report.

    Steps:
      1. Pull session, emotion frames, and transcript segments from MongoDB.
      2. Generate PDF bytes via InterviewReportGenerator.
      3. Store PDF in MongoDB GridFS via pdf_storage module.
      4. Push session_ended message with report_url to the browser.
    """
    from apps.reporting.report_generator import InterviewReportGenerator
    from apps.reporting.pdf_storage import store_report
    from db.mongo_client import get_sync_db

    db = get_sync_db()

    session = db.interview_sessions.find_one({"session_id": session_id})
    if not session:
        logger.error("generate_report: session %s not found", session_id)
        return

    emotion_frames = list(
        db.emotion_frames.find({"session_id": session_id}).sort("frame_index", 1)
    )
    transcript_segments = list(
        db.transcript_segments.find({"session_id": session_id}).sort("segment_index", 1)
    )

    logger.info(
        "Generating report for session=%s frames=%d segments=%d",
        session_id, len(emotion_frames), len(transcript_segments),
    )

    generator = InterviewReportGenerator()
    pdf_bytes: bytes = generator.generate(session, emotion_frames, transcript_segments)

    report_url: str = store_report(session_id, pdf_bytes)

    # Persist report URL on the session document.
    from datetime import datetime, timezone
    db.interview_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"report_url": report_url, "updated_at": datetime.now(timezone.utc)}},
    )

    logger.info("Report stored: session=%s url=%s", session_id, report_url)

    # Push final session_ended with the real report URL to the browser.
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        group_name,
        {
            "type":       "session_ended",
            "session_id": session_id,
            "report_url": report_url,
        },
    )
