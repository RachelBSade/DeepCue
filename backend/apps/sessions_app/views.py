"""HTTP views for sessions_app."""
from __future__ import annotations

import redis
from django.conf import settings
from django.http import HttpRequest, JsonResponse


def health_check(request: HttpRequest) -> JsonResponse:
    """GET /api/health/ — probe Django, Redis, and MongoDB."""
    services: dict[str, str] = {"django": "ok", "redis": "unknown", "mongodb": "unknown"}

    try:
        r = redis.from_url(settings.CELERY_BROKER_URL, socket_connect_timeout=2)
        r.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"

    try:
        from db.mongo_client import get_sync_db
        get_sync_db().command("ping")
        services["mongodb"] = "ok"
    except Exception as exc:
        services["mongodb"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in services.values())
    return JsonResponse(
        {"status": "ok" if all_ok else "degraded", "services": services},
        status=200 if all_ok else 503,
    )
