"""Base Django settings for DeepCue."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

SECRET_KEY: str = os.environ["DJANGO_SECRET_KEY"]
DEBUG: bool = False
ALLOWED_HOSTS: list[str] = []

DJANGO_APPS: list[str] = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS: list[str] = [
    "channels",
    "django_celery_results",
    "corsheaders",
]

LOCAL_APPS: list[str] = [
    "apps.sessions_app",
    "apps.inference",
    "apps.reporting",
]

INSTALLED_APPS: list[str] = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE: list[str] = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF: str = "deepcue_backend.urls"
ASGI_APPLICATION: str = "deepcue_backend.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES: dict = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS: list[dict] = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE: str = "he"
TIME_ZONE: str = "Asia/Jerusalem"
USE_I18N: bool = True
USE_TZ: bool = True

STATIC_URL: str = "/static/"
DEFAULT_AUTO_FIELD: str = "django.db.models.BigAutoField"

# --- Django Channels ---------------------------------------------------------
CHANNEL_LAYERS: dict = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [os.environ.get("CHANNELS_REDIS_URL", "redis://localhost:6379/2")],
        },
    }
}

# --- Celery ------------------------------------------------------------------
CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_RESULT_EXTENDED: bool = True
CELERY_TASK_SERIALIZER: str = "json"
CELERY_RESULT_SERIALIZER: str = "json"
CELERY_ACCEPT_CONTENT: list[str] = ["json"]
CELERY_TIMEZONE: str = "Asia/Jerusalem"
CELERY_TASK_TRACK_STARTED: bool = True
CELERY_TASK_ROUTES: dict = {
    "tasks.video_tasks.*":  {"queue": "video_queue"},
    "tasks.audio_tasks.*":  {"queue": "audio_queue"},
    "tasks.text_tasks.*":   {"queue": "audio_queue"},
    "tasks.fusion_tasks.*": {"queue": "fusion_queue"},
    "tasks.report_tasks.*": {"queue": "fusion_queue"},
}

# --- MongoDB -----------------------------------------------------------------
MONGODB_URI: str = os.environ["MONGODB_URI"]
MONGODB_DB_NAME: str = os.environ.get("MONGODB_DB_NAME", "deepcue")

# --- Inference paths ---------------------------------------------------------
VIDEO_MODEL_PATH: str = os.environ.get("VIDEO_MODEL_PATH", "models/video/efficientnet_lstm.onnx")
AUDIO_MODEL_PATH: str = os.environ.get("AUDIO_MODEL_PATH", "models/audio/wav2vec2_classifier.onnx")
TEXT_MODEL_PATH: str  = os.environ.get("TEXT_MODEL_PATH",  "models/text/xlm_roberta_sentiment.onnx")
FUSION_MODEL_PATH: str = os.environ.get("FUSION_MODEL_PATH", "models/fusion/cross_modal_transformer.onnx")
WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_CACHE_DIR: str  = os.environ.get("WHISPER_CACHE_DIR",  "models/text/whisper_cache")
VIDEO_LSTM_WINDOW_SIZE: int = int(os.environ.get("VIDEO_LSTM_WINDOW_SIZE", "30"))
AUDIO_CHUNK_SECONDS: int    = int(os.environ.get("AUDIO_CHUNK_SECONDS", "3"))

# --- Email (Gmail SMTP) — sends the PDF report to candidates who opt in -------
EMAIL_BACKEND       = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST          = "smtp.gmail.com"
EMAIL_PORT          = 587
EMAIL_USE_TLS       = True
EMAIL_HOST_USER     = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
DEFAULT_FROM_EMAIL  = EMAIL_HOST_USER

# --- Logging (9.3) ------------------------------------------------------------
# JSON in production (machine-parseable, e.g. for log aggregation); plain text
# locally (human-readable in the terminal). Reads DJANGO_DEBUG directly from
# the environment rather than the DEBUG setting, since this module runs before
# local.py/production.py override DEBUG.
_LOG_FORMATTER: str = "verbose" if os.environ.get("DJANGO_DEBUG", "False") == "True" else "json"

LOGGING: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "deepcue_backend.logging_json.JSONFormatter",
        },
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": _LOG_FORMATTER,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        # "apps" catches apps.sessions_app.*, apps.inference.*, apps.reporting.*
        "django": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "apps":   {"handlers": ["console"], "level": "INFO", "propagate": False},
        "tasks":  {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
