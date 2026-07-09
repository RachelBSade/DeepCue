"""
WebSocket URL routing for the sessions_app. (2.8)

Registered in deepcue_backend/asgi.py via:
    from apps.sessions_app.routing import websocket_urlpatterns

URL pattern: ws/interview/<session_id>/
  session_id — UUID4 string generated client-side before connecting.
"""
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r"ws/interview/(?P<session_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/$",
        consumers.InterviewConsumer.as_asgi(),
    ),
]
