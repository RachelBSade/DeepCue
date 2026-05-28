"""
MongoDB client module for DeepCue.

Provides two module-level singletons:
  - sync_db  — pymongo Database (use in Celery tasks and Django views)
  - async_db — motor AsyncIOMotorDatabase (use in Django Channels consumers)

Both are initialised lazily on first import and reuse a single client
connection pool for the lifetime of the process.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import motor.motor_asyncio
import pymongo
from django.conf import settings

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase
    from pymongo.database import Database


# ---------------------------------------------------------------------------
# Synchronous client (pymongo) — Celery tasks, Django views
# ---------------------------------------------------------------------------

_sync_client: pymongo.MongoClient | None = None


def get_sync_db() -> "Database":
    """Return the pymongo Database singleton, creating the client on first call."""
    global _sync_client
    if _sync_client is None:
        _sync_client = pymongo.MongoClient(settings.MONGODB_URI)
    return _sync_client[settings.MONGODB_DB_NAME]


# Module-level convenience alias.
sync_db: "Database" = None  # type: ignore[assignment]


def _init_sync() -> None:
    global sync_db
    sync_db = get_sync_db()


# ---------------------------------------------------------------------------
# Asynchronous client (motor) — Django Channels consumers
# ---------------------------------------------------------------------------

_async_client: motor.motor_asyncio.AsyncIOMotorClient | None = None


def get_async_db() -> "AsyncIOMotorDatabase":
    """Return the motor AsyncIOMotorDatabase singleton."""
    global _async_client
    if _async_client is None:
        _async_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    return _async_client[settings.MONGODB_DB_NAME]


# Module-level convenience alias (resolved on first use, not at import time,
# because Django settings may not be configured yet during app startup).
class _AsyncDBProxy:
    """Deferred proxy so `from db.mongo_client import async_db` works safely."""

    def __getattr__(self, name: str):  # type: ignore[override]
        return getattr(get_async_db(), name)


async_db: "AsyncIOMotorDatabase" = _AsyncDBProxy()  # type: ignore[assignment]
