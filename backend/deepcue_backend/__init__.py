"""Expose the Celery app so `celery -A deepcue_backend` resolves correctly."""
from .celery import app as celery_app

__all__ = ["celery_app"]
