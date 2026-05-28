"""AppConfig for the reporting Django application."""
from django.apps import AppConfig


class ReportingConfig(AppConfig):
    """Generates ReportLab PDF reports and stores them in MongoDB GridFS."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.reporting"
    label = "reporting"
