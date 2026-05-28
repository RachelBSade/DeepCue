"""
ASGI entry point for DeepCue.

Channels ProtocolTypeRouter dispatches:
  - HTTP  → standard Django ASGI application
  - WebSocket → InterviewConsumer (defined in Phase 2)

The websocket_urlpatterns import is deferred via try/except so this file
is importable during Phase 1 before the consumer and routing modules exist.
"""
import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepcue_backend.settings.local")

django_asgi_app = get_asgi_application()

# WebSocket routing is defined in apps/sessions_app/routing.py (Phase 2).
# Guard allows this file to load cleanly before that module exists.
try:
    from apps.sessions_app.routing import websocket_urlpatterns
except ImportError:
    websocket_urlpatterns = []

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        ),
    }
)
