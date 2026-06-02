"""MongoDB Atlas client."""

from __future__ import annotations

import logging
import os

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def mongo_uri() -> str:
    return os.getenv("MONGODB_URI", "").strip()


def mongo_enabled() -> bool:
    return bool(mongo_uri())


def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = mongo_uri()
        if not uri:
            raise RuntimeError("MONGODB_URI is not set in .env")
        _client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    return _client


def get_db() -> Database:
    name = os.getenv("MONGODB_DB_NAME", "salevora").strip() or "salevora"
    return get_client()[name]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception as exc:
        logger.warning("MongoDB ping failed: %s", exc)
        return False


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
