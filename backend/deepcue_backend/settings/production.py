"""Production settings."""
import os
from .base import *  # noqa: F401, F403

DEBUG = False
ALLOWED_HOSTS: list[str] = os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")
CORS_ALLOWED_ORIGINS: list[str] = os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")
SECURE_SSL_REDIRECT: bool = True
SECURE_HSTS_SECONDS: int = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS: bool = True
SESSION_COOKIE_SECURE: bool = True
CSRF_COOKIE_SECURE: bool = True
