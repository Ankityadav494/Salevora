"""Email OTP via Brevo for login and registration."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone

import bcrypt

from src.alerts import brevo
from src.auth.database import create_user, get_user_by_email, mark_email_verified
from src.auth.security import hash_password

OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))
OTP_EXPIRE_MINUTES = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_code() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(OTP_LENGTH))


def _hash_code(code: str) -> str:
    return bcrypt.hashpw(code.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_code(code: str, code_hash: str) -> bool:
    try:
        return bcrypt.checkpw(code.encode("utf-8"), code_hash.encode("utf-8"))
    except ValueError:
        return False


def _send_otp_email(email: str, code: str, purpose: str) -> None:
    if purpose == "reset_password":
        action = "reset your password on"
        subject = f"Reset your Salevora password: {code}"
        heading = "Reset your password"
    elif purpose == "login":
        action = "sign in to"
        subject = f"Your Salevora code: {code}"
        heading = "Salevora verification code"
    else:
        action = "verify your account on"
        subject = f"Your Salevora code: {code}"
        heading = "Salevora verification code"
    text = f"Your Salevora code is {code}. It expires in {OTP_EXPIRE_MINUTES} minutes."
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px">
      <h2 style="color:#1F1C18;margin:0 0 12px">{heading}</h2>
      <p style="color:#5C5650;line-height:1.6">Use this code to {action} Salevora:</p>
      <p style="font-size:32px;font-weight:700;letter-spacing:6px;color:#2F5233;margin:16px 0">{code}</p>
      <p style="color:#8A837A;font-size:14px">Expires in {OTP_EXPIRE_MINUTES} minutes. Do not share this code.</p>
    </div>
    """
    if not brevo.is_configured():
        raise ValueError(
            "Email is not set up yet. Ask whoever runs this app to configure Brevo in .env."
        )
    try:
        brevo.send_email(email, subject, html, text_content=text)
    except RuntimeError as exc:
        raise ValueError(
            "We could not send the email. Check your inbox spam folder, or try again in a minute. "
            f"({exc})"
        ) from exc


def request_otp(
    email: str,
    purpose: str,
    *,
    password: str | None = None,
    name: str | None = None,
) -> dict:
    email = email.lower().strip()
    purpose = purpose.lower().strip()

    if purpose == "login":
        raise ValueError("Sign-in codes are disabled. Use your email and password instead.")
    elif purpose == "register":
        if get_user_by_email(email):
            raise ValueError("An account with this email already exists.")
        if not name or not password:
            raise ValueError("Name and password are required.")
        meta = json.dumps({"name": name, "password_hash": hash_password(password)})
    else:
        raise ValueError("Invalid OTP purpose.")

    from src.auth.database import save_otp

    code = _generate_code()
    expires = (_utc_now() + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    save_otp(email, purpose, _hash_code(code), meta, expires)
    _send_otp_email(email, code, purpose)

    return {
        "otp_sent": True,
        "skip_otp": False,
        "message": f"We sent a sign-in code to {email}. Check your inbox and spam folder.",
        "expires_in_minutes": OTP_EXPIRE_MINUTES,
    }


def verify_otp(email: str, code: str, purpose: str) -> dict:
    from src.auth.database import consume_otp

    email = email.lower().strip()
    purpose = purpose.lower().strip()
    record = consume_otp(email, purpose, code, _verify_code)
    if not record:
        raise ValueError("Invalid or expired verification code.")

    if purpose == "login":
        raise ValueError("Sign-in codes are disabled. Use your email and password instead.")

    if purpose == "register":
        meta = json.loads(record["meta_json"] or "{}")
        user = create_user(email, meta["name"], meta["password_hash"])
        mark_email_verified(user["id"])
        return user

    raise ValueError("Invalid OTP purpose.")


def request_password_reset(email: str) -> dict:
    """Send a reset code if the account exists. Always returns the same message."""
    email = email.lower().strip()
    generic = {
        "message": (
            "If an account exists for that email, we sent a reset code. "
            "Check your inbox and spam folder."
        ),
        "expires_in_minutes": OTP_EXPIRE_MINUTES,
    }

    user = get_user_by_email(email)
    if not user:
        return generic

    from src.auth.database import save_otp

    code = _generate_code()
    expires = (_utc_now() + timedelta(minutes=OTP_EXPIRE_MINUTES)).isoformat()
    save_otp(email, "reset_password", _hash_code(code), None, expires)
    _send_otp_email(email, code, "reset_password")
    return generic


def reset_password_with_otp(email: str, code: str, new_password: str) -> dict:
    from src.auth.database import consume_otp, update_password

    email = email.lower().strip()
    if len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    record = consume_otp(email, "reset_password", code, _verify_code)
    if not record:
        raise ValueError("Invalid or expired reset code.")

    user = update_password(email, hash_password(new_password))
    if not user:
        raise ValueError("No account found for that email.")

    return user
