"""User and OTP storage — MongoDB Atlas or local SQLite."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from src.utils.config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = BASE_DIR / "data" / "salevora.db"

DEFAULT_ALERT_TIME = "10:00"
DEFAULT_ALERT_COOLDOWN_HOURS = 24


def use_mongo() -> bool:
    from src.db.mongo import mongo_enabled

    return mongo_enabled()


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def init_db() -> None:
    if use_mongo():
        _init_mongo()
    else:
        _init_sqlite()


def get_db_info() -> dict[str, Any]:
    if use_mongo():
        from src.db.mongo import get_db, ping

        return {
            "backend": "mongodb",
            "connected": ping(),
            "database": os.getenv("MONGODB_DB_NAME", "salevora"),
        }
    return {
        "backend": "sqlite",
        "connected": DB_PATH.exists(),
        "path": str(DB_PATH),
    }


def get_user_by_email(email: str) -> dict[str, Any] | None:
    if use_mongo():
        return _mongo_get_user_by_email(email)
    return _sqlite_get_user_by_email(email)


def get_user_by_id(user_id: int | str) -> dict[str, Any] | None:
    if use_mongo():
        return _mongo_get_user_by_id(user_id)
    try:
        return _sqlite_get_user_by_id(int(user_id))
    except (TypeError, ValueError):
        return None


def create_user(email: str, name: str, password_hash: str) -> dict[str, Any]:
    if use_mongo():
        return _mongo_create_user(email, name, password_hash)
    return _sqlite_create_user(email, name, password_hash)


def update_alert_settings(
    user_id: int | str,
    *,
    alert_email: str | None = None,
    alerts_enabled: bool | None = None,
    alert_time: str | None = None,
    alert_cooldown_hours: int | None = None,
    alert_schedule_enabled: bool | None = None,
) -> dict[str, Any]:
    if use_mongo():
        return _mongo_update_alert_settings(
            user_id,
            alert_email=alert_email,
            alerts_enabled=alerts_enabled,
            alert_time=alert_time,
            alert_cooldown_hours=alert_cooldown_hours,
            alert_schedule_enabled=alert_schedule_enabled,
        )
    return _sqlite_update_alert_settings(
        int(user_id),
        alert_email=alert_email,
        alerts_enabled=alerts_enabled,
        alert_time=alert_time,
        alert_cooldown_hours=alert_cooldown_hours,
        alert_schedule_enabled=alert_schedule_enabled,
    )


def get_password_hash(email: str) -> str | None:
    if use_mongo():
        return _mongo_get_password_hash(email)
    return _sqlite_get_password_hash(email)


def update_password(email: str, password_hash: str) -> dict[str, Any] | None:
    if use_mongo():
        return _mongo_update_password(email, password_hash)
    return _sqlite_update_password(email, password_hash)


def mark_email_verified(user_id: int | str) -> None:
    if use_mongo():
        _mongo_mark_email_verified(user_id)
    else:
        _sqlite_mark_email_verified(int(user_id))


def save_otp(
    email: str,
    purpose: str,
    code_hash: str,
    meta_json: str | None,
    expires_at: str,
) -> None:
    if use_mongo():
        _mongo_save_otp(email, purpose, code_hash, meta_json, expires_at)
    else:
        _sqlite_save_otp(email, purpose, code_hash, meta_json, expires_at)


def consume_otp(email: str, purpose: str, code: str, verify_fn) -> dict[str, Any] | None:
    if use_mongo():
        return _mongo_consume_otp(email, purpose, code, verify_fn)
    return _sqlite_consume_otp(email, purpose, code, verify_fn)


# ------------------------------------------------------------------ #
#  MongoDB
# ------------------------------------------------------------------ #


def _init_mongo() -> None:
    from src.db.mongo import get_db, ping

    if not ping():
        raise RuntimeError(
            "Could not connect to MongoDB Atlas. Check MONGODB_URI in .env and that "
            "your IP is allowed in Atlas → Network Access."
        )
    db = get_db()
    db.users.create_index("email", unique=True)
    db.otp_codes.create_index([("email", 1), ("purpose", 1)])
    try:
        db.otp_codes.create_index("expires_at", expireAfterSeconds=0)
    except Exception as exc:
        logger.debug("OTP TTL index skipped: %s", exc)
    logger.info("MongoDB connected — database: %s", db.name)


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _mongo_doc_to_user(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "email": doc["email"],
        "name": doc["name"],
        "alert_email": doc.get("alert_email") or doc["email"],
        "alerts_enabled": bool(doc.get("alerts_enabled", True)),
        "email_verified": bool(doc.get("email_verified", False)),
        "alert_time": doc.get("alert_time") or DEFAULT_ALERT_TIME,
        "alert_cooldown_hours": int(doc.get("alert_cooldown_hours", DEFAULT_ALERT_COOLDOWN_HOURS)),
        "alert_schedule_enabled": bool(doc.get("alert_schedule_enabled", True)),
        "created_at": doc.get("created_at"),
    }


def _mongo_oid(user_id: int | str) -> ObjectId | None:
    try:
        return ObjectId(str(user_id))
    except (InvalidId, TypeError):
        return None


def _mongo_get_user_by_email(email: str) -> dict[str, Any] | None:
    from src.db.mongo import get_db

    doc = get_db().users.find_one({"email": email.lower()})
    return _mongo_doc_to_user(doc)


def _mongo_get_user_by_id(user_id: int | str) -> dict[str, Any] | None:
    from src.db.mongo import get_db

    oid = _mongo_oid(user_id)
    if not oid:
        return None
    doc = get_db().users.find_one({"_id": oid})
    return _mongo_doc_to_user(doc)


def _mongo_create_user(email: str, name: str, password_hash: str) -> dict[str, Any]:
    from src.db.mongo import get_db

    now = datetime.now(timezone.utc)
    email = email.lower()
    doc = {
        "email": email,
        "name": name,
        "password_hash": password_hash,
        "alert_email": email,
        "alerts_enabled": True,
        "email_verified": False,
        "alert_time": DEFAULT_ALERT_TIME,
        "alert_cooldown_hours": DEFAULT_ALERT_COOLDOWN_HOURS,
        "alert_schedule_enabled": True,
        "created_at": now,
    }
    result = get_db().users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _mongo_doc_to_user(doc)  # type: ignore[return-value]


def _mongo_update_alert_settings(
    user_id: int | str,
    *,
    alert_email: str | None = None,
    alerts_enabled: bool | None = None,
    alert_time: str | None = None,
    alert_cooldown_hours: int | None = None,
    alert_schedule_enabled: bool | None = None,
) -> dict[str, Any]:
    user = _mongo_get_user_by_id(user_id)
    if not user:
        raise ValueError("User not found")
    oid = _mongo_oid(user_id)
    updates: dict[str, Any] = {}
    if alert_email is not None:
        updates["alert_email"] = alert_email.lower() if alert_email else None
    if alerts_enabled is not None:
        updates["alerts_enabled"] = bool(alerts_enabled)
    if alert_time is not None:
        updates["alert_time"] = alert_time
    if alert_cooldown_hours is not None:
        updates["alert_cooldown_hours"] = int(alert_cooldown_hours)
    if alert_schedule_enabled is not None:
        updates["alert_schedule_enabled"] = bool(alert_schedule_enabled)
    if updates:
        from src.db.mongo import get_db

        get_db().users.update_one({"_id": oid}, {"$set": updates})
    return _mongo_get_user_by_id(user_id)  # type: ignore[return-value]


def _mongo_get_password_hash(email: str) -> str | None:
    from src.db.mongo import get_db

    doc = get_db().users.find_one({"email": email.lower()}, {"password_hash": 1})
    return doc["password_hash"] if doc else None


def _mongo_update_password(email: str, password_hash: str) -> dict[str, Any] | None:
    from src.db.mongo import get_db

    result = get_db().users.update_one(
        {"email": email.lower()},
        {"$set": {"password_hash": password_hash}},
    )
    if result.matched_count == 0:
        return None
    return get_user_by_email(email)


def _mongo_mark_email_verified(user_id: int | str) -> None:
    from src.db.mongo import get_db

    oid = _mongo_oid(user_id)
    if oid:
        get_db().users.update_one({"_id": oid}, {"$set": {"email_verified": True}})


def _mongo_save_otp(
    email: str,
    purpose: str,
    code_hash: str,
    meta_json: str | None,
    expires_at: str,
) -> None:
    from src.db.mongo import get_db

    now = datetime.now(timezone.utc)
    db = get_db()
    db.otp_codes.delete_many({"email": email.lower(), "purpose": purpose})
    db.otp_codes.insert_one(
        {
            "email": email.lower(),
            "purpose": purpose,
            "code_hash": code_hash,
            "meta_json": meta_json,
            "expires_at": _parse_dt(expires_at),
            "created_at": now,
        }
    )


def _mongo_consume_otp(email: str, purpose: str, code: str, verify_fn) -> dict[str, Any] | None:
    from src.db.mongo import get_db

    db = get_db()
    doc = db.otp_codes.find_one(
        {"email": email.lower(), "purpose": purpose},
        sort=[("created_at", -1)],
    )
    if not doc:
        return None
    if datetime.now(timezone.utc) > _parse_dt(doc["expires_at"]):
        db.otp_codes.delete_one({"_id": doc["_id"]})
        return None
    if not verify_fn(code.strip(), doc["code_hash"]):
        return None
    db.otp_codes.delete_one({"_id": doc["_id"]})
    return {
        "id": str(doc["_id"]),
        "email": doc["email"],
        "purpose": doc["purpose"],
        "code_hash": doc["code_hash"],
        "meta_json": doc.get("meta_json"),
        "expires_at": doc["expires_at"].isoformat()
        if isinstance(doc["expires_at"], datetime)
        else doc["expires_at"],
        "created_at": doc.get("created_at"),
    }


# ------------------------------------------------------------------ #
#  SQLite (local fallback when MONGODB_URI is not set)
# ------------------------------------------------------------------ #


def _sqlite_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sqlite() -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                alert_email TEXT,
                alerts_enabled INTEGER DEFAULT 1,
                email_verified INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                purpose TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                meta_json TEXT,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        for col, ddl in (
            ("alert_time", "TEXT DEFAULT '10:00'"),
            ("alert_cooldown_hours", "INTEGER DEFAULT 24"),
            ("alert_schedule_enabled", "INTEGER DEFAULT 1"),
        ):
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
    logger.info("SQLite database ready at %s", DB_PATH)


def _sqlite_row_to_user(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = row.keys()
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "alert_email": row["alert_email"] or row["email"],
        "alerts_enabled": bool(row["alerts_enabled"]),
        "email_verified": bool(row["email_verified"]) if "email_verified" in keys else False,
        "alert_time": row["alert_time"] if "alert_time" in keys and row["alert_time"] else DEFAULT_ALERT_TIME,
        "alert_cooldown_hours": int(row["alert_cooldown_hours"])
        if "alert_cooldown_hours" in keys and row["alert_cooldown_hours"] is not None
        else DEFAULT_ALERT_COOLDOWN_HOURS,
        "alert_schedule_enabled": bool(row["alert_schedule_enabled"])
        if "alert_schedule_enabled" in keys
        else True,
        "created_at": row["created_at"],
    }


def _sqlite_get_user_by_email(email: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return _sqlite_row_to_user(row)


def _sqlite_get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _sqlite_row_to_user(row)


def _sqlite_create_user(email: str, name: str, password_hash: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _sqlite_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (email, name, password_hash, alert_email, alerts_enabled, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (email.lower(), name, password_hash, email.lower(), now),
        )
        conn.commit()
        user_id = cur.lastrowid
    return _sqlite_get_user_by_id(int(user_id))  # type: ignore[arg-type]


def _sqlite_update_alert_settings(
    user_id: int,
    *,
    alert_email: str | None = None,
    alerts_enabled: bool | None = None,
    alert_time: str | None = None,
    alert_cooldown_hours: int | None = None,
    alert_schedule_enabled: bool | None = None,
) -> dict[str, Any]:
    user = _sqlite_get_user_by_id(user_id)
    if not user:
        raise ValueError("User not found")
    new_email = alert_email if alert_email is not None else user["alert_email"]
    new_enabled = alerts_enabled if alerts_enabled is not None else user["alerts_enabled"]
    new_time = alert_time if alert_time is not None else user["alert_time"]
    new_cooldown = (
        alert_cooldown_hours if alert_cooldown_hours is not None else user["alert_cooldown_hours"]
    )
    new_schedule = (
        alert_schedule_enabled
        if alert_schedule_enabled is not None
        else user["alert_schedule_enabled"]
    )
    with _sqlite_connect() as conn:
        conn.execute(
            """
            UPDATE users
            SET alert_email = ?, alerts_enabled = ?, alert_time = ?,
                alert_cooldown_hours = ?, alert_schedule_enabled = ?
            WHERE id = ?
            """,
            (
                new_email.lower() if new_email else None,
                int(new_enabled),
                new_time,
                int(new_cooldown),
                int(new_schedule),
                user_id,
            ),
        )
        conn.commit()
    return _sqlite_get_user_by_id(user_id)  # type: ignore[return-value]


def _sqlite_get_password_hash(email: str) -> str | None:
    with _sqlite_connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return row["password_hash"] if row else None


def _sqlite_update_password(email: str, password_hash: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (password_hash, email.lower()),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_user_by_email(email)


def _sqlite_mark_email_verified(user_id: int) -> None:
    with _sqlite_connect() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
        conn.commit()


def _sqlite_save_otp(
    email: str,
    purpose: str,
    code_hash: str,
    meta_json: str | None,
    expires_at: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM otp_codes WHERE email = ? AND purpose = ?", (email.lower(), purpose))
        conn.execute(
            """
            INSERT INTO otp_codes (email, purpose, code_hash, meta_json, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (email.lower(), purpose, code_hash, meta_json, expires_at, now),
        )
        conn.commit()


def _sqlite_consume_otp(email: str, purpose: str, code: str, verify_fn) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute(
            "SELECT * FROM otp_codes WHERE email = ? AND purpose = ? ORDER BY id DESC LIMIT 1",
            (email.lower(), purpose),
        ).fetchone()
        if not row:
            return None
        expires = _parse_dt(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            conn.execute("DELETE FROM otp_codes WHERE id = ?", (row["id"],))
            conn.commit()
            return None
        if not verify_fn(code.strip(), row["code_hash"]):
            return None
        conn.execute("DELETE FROM otp_codes WHERE id = ?", (row["id"],))
        conn.commit()
        return dict(row)
