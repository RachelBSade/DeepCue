"""Phase 7.9 — HTTP endpoint: GET /api/report/<session_id>/"""
from __future__ import annotations

from django.http import HttpRequest, HttpResponse, Http404


def download_report(request: HttpRequest, session_id: str) -> HttpResponse:
    """
    Stream a generated PDF to the browser.

    Returns 200 with Content-Type application/pdf on success.
    Returns 404 if no report exists for the given session.
    """
    from apps.reporting.pdf_storage import retrieve_report

    pdf_bytes = retrieve_report(session_id)
    if pdf_bytes is None:
        raise Http404(f"No report found for session {session_id}")

    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="deepcue_report_{session_id[:8]}.pdf"'
    )
    response["Content-Length"] = str(len(pdf_bytes))
    return response
