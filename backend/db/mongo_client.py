"""MongoDB client singletons for DeepCue.

sync_db  — pymongo Database  (Celery tasks, Django views)
async_db — motor Database    (Django Channels consumers)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import motor.motor_asyncio
import pymongo
from django.conf import settings

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase
    from pymongo.database import Database

_sync_client: pymongo.MongoClient | None = None
_async_client: motor.motor_asyncio.AsyncIOMotorClient | None = None


def get_sync_db() -> "Database":
    """Return the lazily-initialised pymongo Database singleton (Celery tasks, Django views)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = pymongo.MongoClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=2000,
        )
    return _sync_client[settings.MONGODB_DB_NAME]


def get_async_db() -> "AsyncIOMotorDatabase":
    """Return the lazily-initialised motor AsyncIOMotorDatabase singleton (Channels consumers)."""
    global _async_client
    if _async_client is None:
        _async_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    return _async_client[settings.MONGODB_DB_NAME]


class _AsyncDBProxy:
    """Defers to get_async_db() on every attribute access, so the `async_db` module-level
    singleton below always reflects the current client even if it's (re)initialised later."""

    def __getattr__(self, name: str) -> object:
        return getattr(get_async_db(), name)


sync_db: "Database" = None  # type: ignore[assignment]
async_db: "AsyncIOMotorDatabase" = _AsyncDBProxy()  # type: ignore[assignment]
