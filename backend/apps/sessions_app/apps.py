"""AppConfig for the sessions_app Django application."""
from django.apps import AppConfig


class SessionsAppConfig(AppConfig):
    """Manages interview session lifecycle and MongoDB document writes."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.sessions_app"
    label = "sessions_app"
