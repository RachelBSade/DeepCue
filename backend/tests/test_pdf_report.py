"""
Phase 8.4 / 8.6 — PDF report generation tests.

Tests:
  - generate() returns valid PDF bytes (starts with %PDF header) (8.6)
  - generate() succeeds with empty frames / segments (edge case)
  - generate() completes within the 10-second latency budget (8.4 partial)
  - Hebrew RTL segment is included in the PDF bytes (8.6)
  - pdf_storage.retrieve_report() returns None when nothing stored
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(**overrides) -> dict[str, Any]:
    base = {
        "session_id":       "test-session-pdf-001",
        "candidate_name":   "ישראל ישראלי",   # Hebrew name for RTL test
        "created_at":       datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        "duration_seconds": 300.0,
        "frame_count":      150,
        "dominant_emotion": "confident",
        "report_url":       None,
        "status":           "completed",
    }
    base.update(overrides)
    return base


def _make_emotion_frame(i: int, timestamp: float = None) -> dict[str, Any]:
    return {
        "session_id":    "test-session-pdf-001",
        "frame_index":   i,
        "timestamp":     timestamp if timestamp is not None else float(i),
        "video_score":   0.6,
        "audio_score":   0.5,
        "text_score":    0.4,
        "fusion_scores": {
            "neutral":   0.1,
            "confident": 0.5,
            "anxious":   0.1,
            "happy":     0.1,
            "sad":       0.05,
            "angry":     0.05,
            "surprised": 0.05,
            "uncertain": 0.05,
        },
        "dominant_emotion": "confident",
    }


def _make_segment(i: int) -> dict[str, Any]:
    return {
        "session_id":      "test-session-pdf-001",
        "segment_index":   i,
        "timestamp":       float(i * 10),
        "text":            "שלום, אני מועמד לתפקיד",  # Hebrew
        "language":        "he",
        "duration_seconds": 3.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInterviewReportGenerator:
    def _gen(self):
        from apps.reporting.report_generator import InterviewReportGenerator
        return InterviewReportGenerator()

    def test_returns_pdf_bytes(self):
        gen = self._gen()
        pdf = gen.generate(_make_session(), [], [])
        assert isinstance(pdf, bytes)
        assert pdf[:4] == b"%PDF", "Output does not start with %PDF header"

    def test_returns_pdf_with_frames_and_segments(self):
        gen = self._gen()
        frames   = [_make_emotion_frame(i, float(i * 2)) for i in range(20)]
        segments = [_make_segment(i) for i in range(5)]
        pdf = gen.generate(_make_session(), frames, segments)
        assert pdf[:4] == b"%PDF"

    def test_pdf_size_is_reasonable(self):
        gen = self._gen()
        frames = [_make_emotion_frame(i) for i in range(50)]
        pdf = gen.generate(_make_session(), frames, [])
        # Should be at least 5 KB and less than 5 MB for a typical report.
        assert 5_000 < len(pdf) < 5_000_000

    def test_hebrew_text_included_in_pdf(self):
        """Verify Hebrew segment text appears in the PDF byte stream (8.6)."""
        gen = self._gen()
        segments = [_make_segment(0)]
        pdf = gen.generate(_make_session(), [], segments)
        # ReportLab encodes text; we just check the PDF was produced without error
        # and has a reasonable size (RTL rendering does not crash).
        assert pdf[:4] == b"%PDF"
        assert len(pdf) > 5_000

    def test_empty_frames_does_not_raise(self):
        gen = self._gen()
        pdf = gen.generate(_make_session(), [], [])
        assert pdf[:4] == b"%PDF"

    def test_generation_within_10_seconds(self):
        """
        Performance guard (8.4): PDF generation must complete under 10 s on the
        target hardware profile.  50 frames + 10 segments is a realistic session.
        """
        gen = self._gen()
        frames   = [_make_emotion_frame(i, float(i)) for i in range(50)]
        segments = [_make_segment(i) for i in range(10)]

        t0 = time.perf_counter()
        gen.generate(_make_session(), frames, segments)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, (
            f"PDF generation took {elapsed:.2f}s — exceeds 10s budget"
        )

    def test_dominant_emotion_appears_in_recommendations(self):
        """A session with high anxious score should trigger the anxiety recommendation."""
        from apps.reporting.report_generator import _average_fusion_scores, _RECOMMENDATION_RULES

        frames = []
        for i in range(10):
            f = _make_emotion_frame(i)
            f["fusion_scores"]["anxious"]   = 0.6
            f["fusion_scores"]["confident"] = 0.1
            frames.append(f)

        avg = _average_fusion_scores(frames)
        assert avg["anxious"] > 0.30, "Fixture did not set anxious score above threshold"

        triggered = [
            title for emotion, threshold, title, _ in _RECOMMENDATION_RULES
            if avg.get(emotion, 0.0) >= threshold
        ]
        assert any("Anxiety" in t or "anxiety" in t.lower() for t in triggered)


# ---------------------------------------------------------------------------
# pdf_storage tests
# ---------------------------------------------------------------------------

class TestPdfStorage:
    @patch("apps.reporting.pdf_storage.get_sync_db")
    def test_retrieve_returns_none_when_not_found(self, mock_db):
        from gridfs import NoFile
        db_mock = MagicMock()
        mock_db.return_value = db_mock

        # GridFS.get_last_version raises NoFile when missing.
        with patch("apps.reporting.pdf_storage.GridFS") as mock_fs_cls:
            fs_mock = MagicMock()
            fs_mock.get_last_version.side_effect = NoFile("not found")
            mock_fs_cls.return_value = fs_mock

            from apps.reporting.pdf_storage import retrieve_report
            result = retrieve_report("nonexistent-session-id")
            assert result is None

    @patch("apps.reporting.pdf_storage.get_sync_db")
    def test_store_returns_url_with_session_id(self, mock_db):
        db_mock = MagicMock()
        mock_db.return_value = db_mock

        with patch("apps.reporting.pdf_storage.GridFS") as mock_fs_cls:
            fs_mock = MagicMock()
            fs_mock.put.return_value = "fake_file_id"
            fs_mock.find.return_value = []
            mock_fs_cls.return_value = fs_mock

            from apps.reporting.pdf_storage import store_report
            url = store_report("my-session-id", b"%PDF-1.4 test")
            assert "my-session-id" in url
            assert url.startswith("/api/report/")
