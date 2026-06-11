"""
Phase 7.8 — PDF Storage via MongoDB GridFS

Stores generated PDF bytes in GridFS and returns a URL path that the
Django download endpoint can resolve.

Interface used by report_tasks.py:
    report_url: str = store_report(session_id, pdf_bytes)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# URL prefix served by the Django report download endpoint (7.9).
_DOWNLOAD_URL_PREFIX = "/api/report/"


def store_report(session_id: str, pdf_bytes: bytes) -> str:
    """
    Save PDF bytes to MongoDB GridFS.

    Parameters
    ----------
    session_id : str  — used as the GridFS filename
    pdf_bytes  : bytes — raw PDF content

    Returns
    -------
    str — URL path to download the report, e.g. "/api/report/<session_id>/"
    """
    from gridfs import GridFS
    from db.mongo_client import get_sync_db

    db = get_sync_db()
    fs = GridFS(db)

    filename = f"{session_id}.pdf"

    # Delete any existing file for this session before writing a new one.
    _delete_existing(fs, filename)

    file_id = fs.put(
        pdf_bytes,
        filename=filename,
        content_type="application/pdf",
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("PDF stored in GridFS: session=%s file_id=%s size=%d bytes",
                session_id, file_id, len(pdf_bytes))

    return f"{_DOWNLOAD_URL_PREFIX}{session_id}/"


def retrieve_report(session_id: str) -> bytes | None:
    """
    Retrieve PDF bytes from GridFS for a given session.

    Returns None if no report exists.
    """
    from gridfs import GridFS, NoFile
    from db.mongo_client import get_sync_db

    db = get_sync_db()
    fs = GridFS(db)
    filename = f"{session_id}.pdf"

    try:
        grid_out = fs.get_last_version(filename=filename)
        return grid_out.read()
    except NoFile:
        return None


def _delete_existing(fs: "GridFS", filename: str) -> None:
    """Remove all GridFS chunks for a given filename (idempotent re-generation)."""
    for grid_out in fs.find({"filename": filename}):
        fs.delete(grid_out._id)
