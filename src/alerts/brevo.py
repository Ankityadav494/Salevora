"""Brevo (Sendinblue) transactional email — REST API or SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


def is_configured() -> bool:
    key = os.getenv("BREVO_API_KEY", "").strip()
    if key.startswith("xkeysib-"):
        return True
    if key.startswith("xsmtpsib-"):
        return bool(os.getenv("BREVO_SMTP_LOGIN", "").strip())
    return bool(key)


def sender() -> dict[str, str]:
    return {
        "name": os.getenv("BREVO_SENDER_NAME", "Salevora"),
        "email": os.getenv("BREVO_SENDER_EMAIL", "").strip(),
    }


def _send_via_api(api_key: str, to_email: str, subject: str, html_content: str) -> dict[str, Any]:
    payload = {
        "sender": sender(),
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content,
    }
    resp = requests.post(
        BREVO_API_URL,
        headers={"api-key": api_key, "Content-Type": "application/json", "accept": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("message", detail)
        except Exception:
            pass
        raise RuntimeError(f"Brevo API error ({resp.status_code}): {detail}")

    data = resp.json() if resp.content else {}
    return {"message_id": data.get("messageId"), "status": "sent", "via": "api"}


def _send_via_smtp(
    smtp_key: str,
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str | None = None,
) -> dict[str, Any]:
    login = os.getenv("BREVO_SMTP_LOGIN", "").strip()
    if not login:
        raise RuntimeError(
            "BREVO_SMTP_LOGIN is missing. Add your Brevo SMTP login (e.g. xxx@smtp-brevo.com) to .env."
        )

    from_info = sender()
    from_addr = from_info["email"]
    if not from_addr:
        raise RuntimeError("BREVO_SENDER_EMAIL is missing in .env.")

    host = os.getenv("BREVO_SMTP_HOST", "smtp-relay.brevo.com").strip()
    port = int(os.getenv("BREVO_SMTP_PORT", "587"))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{from_info['name']} <{from_addr}>"
    msg["To"] = to_email

    plain = text_content or "Open this email in an HTML-capable client to view the message."
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(login, smtp_key)
            refused = smtp.sendmail(from_addr, [to_email], msg.as_string())
            if refused:
                raise RuntimeError(f"SMTP refused recipients: {refused}")
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            "Brevo SMTP login failed. Check BREVO_API_KEY (xsmtpsib-...) and BREVO_SMTP_LOGIN in .env."
        ) from exc
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP error: {exc}") from exc

    logger.info("Email sent via SMTP to %s from %s", to_email, from_addr)
    return {"message_id": None, "status": "sent", "via": "smtp"}


def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    *,
    text_content: str | None = None,
) -> dict[str, Any]:
    api_key = (os.getenv("BREVO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("BREVO_API_KEY is not set. Add it to your .env file.")

    if api_key.startswith("xsmtpsib-"):
        return _send_via_smtp(api_key, to_email, subject, html_content, text_content)

    if api_key.startswith("xkeysib-"):
        return _send_via_api(api_key, to_email, subject, html_content)

    raise RuntimeError(
        "BREVO_API_KEY must start with xsmtpsib- (SMTP) or xkeysib- (REST API)."
    )
