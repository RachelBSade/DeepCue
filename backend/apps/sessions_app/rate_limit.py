"""
Per-connection rate limiting for InterviewConsumer. (9.1)

Each WebSocket connection gets its own TokenBucket instances (one per message
type), since each browser tab holds a dedicated consumer instance for the
life of the connection — no cross-process coordination (e.g. via Redis) is
needed to protect against a single runaway/malicious client.
"""
from __future__ import annotations

import time


class TokenBucket:
    """Classic token-bucket limiter: refills at `rate` tokens/sec, holds at most `capacity`."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        """Consume one token if available; return False if the caller should be throttled."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# Limits chosen generously above expected client send rates (video ~10/s,
# audio chunks ~1 every 3s) so legitimate traffic never gets throttled —
# this only kicks in for a misbehaving or malicious client.
VIDEO_FRAME_RATE: float = 30.0
VIDEO_FRAME_CAPACITY: float = 60.0
AUDIO_CHUNK_RATE: float = 2.0
AUDIO_CHUNK_CAPACITY: float = 5.0
