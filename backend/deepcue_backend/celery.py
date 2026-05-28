"""Celery application entry point for DeepCue."""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepcue_backend.settings.local")

app = Celery("deepcue")

# Pull all CELERY_* keys from Django settings.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks in all INSTALLED_APPS.
app.autodiscover_tasks()
