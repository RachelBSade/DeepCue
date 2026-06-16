"""
Shared pytest fixtures and Django/Celery test configuration.

All tests use the `test` settings override defined here — no real Redis,
MongoDB, or model files required.
"""
from __future__ import annotations

import os
import struct
import wave
from io import BytesIO

import django
import pytest

# Point Django at test settings before any app code is imported.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepcue_backend.settings.test")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")


def pytest_configure(config: pytest.Config) -> None:
    django.setup()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def make_wav_bytes(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Return minimal silent WAV bytes (16-bit mono at 16 kHz)."""
    n_samples = int(sample_rate * duration_seconds)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def make_landmarks(n: int = 468) -> list[dict]:
    """Return n landmark dicts with deterministic (x, y, z) values."""
    return [
        {"x": (i % 100) / 100.0, "y": (i % 50) / 50.0, "z": 0.0}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def landmarks() -> list[dict]:
    return make_landmarks()


@pytest.fixture
def fake_session_id() -> str:
    return "12345678-1234-4000-a000-123456789abc"
