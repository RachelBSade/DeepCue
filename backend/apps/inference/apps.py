"""AppConfig for the inference Django application."""
from django.apps import AppConfig


class InferenceConfig(AppConfig):
    """Hosts the four CPU-optimized ONNX inference pipeline classes."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.inference"
    label = "inference"
