"""Local development settings."""
from .base import *  # noqa: F401, F403

DEBUG = True
ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]
CORS_ALLOW_ALL_ORIGINS: bool = True
