"""Local development settings — never use in production."""
from .base import *  # noqa: F401, F403

DEBUG = True

ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]

# Allow the frontend (served as a plain file or on any dev port) to call the API.
CORS_ALLOW_ALL_ORIGINS: bool = True
