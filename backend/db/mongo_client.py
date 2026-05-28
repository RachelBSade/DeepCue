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
    global _sync_client
    if _sync_client is None:
        _sync_client = pymongo.MongoClient(settings.MONGODB_URI)
    return _sync_client[settings.MONGODB_DB_NAME]


def get_async_db() -> "AsyncIOMotorDatabase":
    global _async_client
    if _async_client is None:
        _async_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    return _async_client[settings.MONGODB_DB_NAME]


class _AsyncDBProxy:
    def __getattr__(self, name: str):
        return getattr(get_async_db(), name)


sync_db: "Database" = None  # type: ignore[assignment]
async_db: "AsyncIOMotorDatabase" = _AsyncDBProxy()  # type: ignore[assignment]
