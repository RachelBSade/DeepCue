"""Test settings — no real services required."""
import os

# These must be set before base.py is imported (it reads them at module level).
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-insecure-key-do-not-use-in-production")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")

from deepcue_backend.settings.base import *  # noqa: F401, F403

DEBUG = True

# Use in-memory channel layer so tests don't need Redis.
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

# Celery runs tasks synchronously in tests (eager mode).
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Dummy broker/backend — not used in eager mode.
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "cache+memory://"

# MongoDB URI is set via env; tests mock the client, so the value is unused.
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB_NAME = "deepcue_test"

# Use non-existent paths — pipelines fall back to NEUTRAL_FALLBACK gracefully.
VIDEO_MODEL_PATH  = "models/video/does_not_exist.onnx"
AUDIO_MODEL_PATH  = "models/audio/does_not_exist.onnx"
TEXT_MODEL_PATH   = "models/text/does_not_exist.onnx"
FUSION_MODEL_PATH = "models/fusion/does_not_exist.onnx"
WHISPER_MODEL_SIZE = "tiny"
WHISPER_CACHE_DIR  = "/tmp/deepcue_test_whisper"
VIDEO_LSTM_WINDOW_SIZE = 4   # Small window so tests fill the buffer quickly.
AUDIO_CHUNK_SECONDS    = 3
