"""
HTTP views for the sessions_app.

Currently exposes only the health-check endpoint. Interview session
management happens over WebSocket (Phase 2); any HTTP session endpoints
(e.g. report download) will be added in later phases.
"""
from __future__ import annotations

import redis
from django.conf import settings
from django.http import HttpRequest, JsonResponse


def health_check(request: HttpRequest) -> JsonResponse:
    """
    GET /api/health/

    Probes Django (trivially alive), Redis, and MongoDB.
    Returns HTTP 200 if all services are reachable, 503 if any are degraded.

    Response body:
        {
            "status": "ok" | "degraded",
            "services": {
                "django":  "ok" | "error: <msg>",
                "redis":   "ok" | "error: <msg>",
                "mongodb": "ok" | "error: <msg>"
            }
        }
    """
    services: dict[str, str] = {
        "django": "ok",
        "redis": "unknown",
        "mongodb": "unknown",
    }

    # --- Redis probe ---------------------------------------------------------
    try:
        r = redis.from_url(settings.CELERY_BROKER_URL, socket_connect_timeout=2)
        r.ping()
        services["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        services["redis"] = f"error: {exc}"

    # --- MongoDB probe -------------------------------------------------------
    try:
        from db.mongo_client import get_sync_db
        db = get_sync_db()
        db.command("ping")
        services["mongodb"] = "ok"
    except Exception as exc:  # noqa: BLE001
        services["mongodb"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in services.values())
    return JsonResponse(
        {"status": "ok" if all_ok else "degraded", "services": services},
        status=200 if all_ok else 503,
    )
