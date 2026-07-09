"""Celery application entry point for DeepCue."""
import os

from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepcue_backend.settings.local")

app = Celery("deepcue")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.include = [
    "tasks.video_tasks",
    "tasks.audio_tasks",
    "tasks.text_tasks",
    "tasks.fusion_tasks",
    "tasks.report_tasks",
]

# ---------------------------------------------------------------------------
# Celery Beat periodic schedule stubs (4.8)
# FUSION_INTERVAL_SECONDS controls how often fusion is triggered per session.
# Active-session fusion is driven by video_tasks dispatching run_fusion on
# each frame — Beat is used only for a safety heartbeat fallback.
# ---------------------------------------------------------------------------
app.conf.beat_schedule = {
    # Placeholder: uncomment and adjust if you need a time-driven fusion tick
    # independent of incoming video frames (e.g. audio-only mode).
    # "fusion-heartbeat": {
    #     "task":     "tasks.fusion_tasks.run_fusion",
    #     "schedule": 1.0,  # every 1 second
    # },
}
