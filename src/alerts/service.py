"""Inventory alert evaluation and Brevo delivery."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.alerts import brevo
from src.inventory.service import build_alerts, build_items
from src.utils.config import BASE_DIR

LIVE_CSV = BASE_DIR / "data" / "processed" / "live_sales.csv"
STATE_FILE = BASE_DIR / "data" / "processed" / "alert_sent_state.json"
HISTORY_FILE = BASE_DIR / "logs" / "alerts.json"
COOLDOWN_HOURS = int(__import__("os").getenv("ALERT_COOLDOWN_HOURS", "24"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _append_history(entry: dict[str, Any]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(history[:200], indent=2), encoding="utf-8")


def _alert_key(alert: dict[str, Any]) -> str:
    return f"{alert.get('sku')}:{alert.get('status')}"


def _filter_new_alerts(
    alerts: list[dict[str, Any]],
    force: bool,
    cooldown_hours: int | None = None,
) -> list[dict[str, Any]]:
    if force:
        return alerts
    hours = cooldown_hours if cooldown_hours is not None else COOLDOWN_HOURS
    state = _load_state()
    cutoff = _utc_now() - timedelta(hours=hours)
    fresh: list[dict[str, Any]] = []
    for alert in alerts:
        key = _alert_key(alert)
        last = state.get(key)
        if not last:
            fresh.append(alert)
            continue
        try:
            sent_at = datetime.fromisoformat(last.replace("Z", "+00:00"))
        except Exception:
            fresh.append(alert)
            continue
        if sent_at < cutoff:
            fresh.append(alert)
    return fresh


def _build_html(alerts: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = ""
    for a in alerts:
        color = "#ef4444" if a.get("status") == "critical" else "#f59e0b"
        rows += f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #eee">{a.get('sku','')}</td>
          <td style="padding:8px;border-bottom:1px solid #eee">{a.get('name','')}</td>
          <td style="padding:8px;border-bottom:1px solid #eee;color:{color};font-weight:600">{a.get('status','').upper()}</td>
          <td style="padding:8px;border-bottom:1px solid #eee">{a.get('stock',0)}</td>
          <td style="padding:8px;border-bottom:1px solid #eee">{a.get('forecast_total_units',0)}</td>
          <td style="padding:8px;border-bottom:1px solid #eee">{a.get('days_of_stock',0)}</td>
        </tr>
        <tr><td colspan="6" style="padding:0 8px 12px;color:#555;font-size:13px">{a.get('alert_message','')}</td></tr>
        """

    return f"""
    <div style="font-family:Inter,Arial,sans-serif;max-width:640px;margin:0 auto">
      <h2 style="color:#111;margin-bottom:4px">Salevora Inventory Alert</h2>
      <p style="color:#666;margin-top:0">Forecasted demand exceeds synced stock for {len(alerts)} SKU(s).</p>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead>
          <tr style="background:#f8fafc;text-align:left">
            <th style="padding:8px">SKU</th>
            <th style="padding:8px">Product</th>
            <th style="padding:8px">Status</th>
            <th style="padding:8px">Stock</th>
            <th style="padding:8px">Forecast 4wk</th>
            <th style="padding:8px">Days Left</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="color:#888;font-size:12px;margin-top:20px">
        Critical: {summary.get('critical',0)} · Warning: {summary.get('warning',0)} ·
        Sent {_utc_now().strftime('%Y-%m-%d %H:%M UTC')}
      </p>
    </div>
    """


def evaluate_alerts() -> dict[str, Any]:
    if not LIVE_CSV.exists():
        raise FileNotFoundError("No sales data found. Upload a CSV first.")
    items = build_items(str(LIVE_CSV))
    alerts = build_alerts(items)
    critical = sum(1 for a in alerts if a.get("status") == "critical")
    warning = sum(1 for a in alerts if a.get("status") == "warning")
    return {
        "alerts": alerts,
        "count": len(alerts),
        "summary": {"critical": critical, "warning": warning, "total": len(alerts)},
    }


def send_inventory_alerts(
    to_email: str,
    *,
    force: bool = False,
    user_email: str | None = None,
    cooldown_hours: int | None = None,
) -> dict[str, Any]:
    if not brevo.is_configured():
        raise RuntimeError("Brevo is not configured. Set BREVO_API_KEY in .env.")

    evaluation = evaluate_alerts()
    alerts = evaluation["alerts"]
    if not alerts:
        return {
            "status": "no_alerts",
            "message": "Everything looks good — no low-stock warnings right now.",
            "sent_count": 0,
            "skipped_count": 0,
            "evaluation": evaluation,
        }

    to_send = _filter_new_alerts(alerts, force=force, cooldown_hours=cooldown_hours)
    if not to_send:
        return {
            "status": "cooldown",
            "message": "We already emailed you about these items recently.",
            "sent_count": 0,
            "skipped_count": len(alerts),
            "evaluation": evaluation,
        }

    subject = f"Salevora: {len(to_send)} item(s) may run out soon"
    html = _build_html(to_send, evaluation["summary"])
    result = brevo.send_email(to_email, subject, html)

    state = _load_state()
    now = _utc_now().isoformat()
    for alert in to_send:
        state[_alert_key(alert)] = now
    _save_state(state)

    entry = {
        "sent_at": now,
        "to": to_email,
        "user": user_email,
        "alert_count": len(to_send),
        "message_id": result.get("message_id"),
        "skus": [a.get("sku") for a in to_send],
    }
    _append_history(entry)

    return {
        "status": "sent",
        "message": f"We sent {len(to_send)} stock reminder(s) to {to_email}.",
        "sent_count": len(to_send),
        "skipped_count": len(alerts) - len(to_send),
        "message_id": result.get("message_id"),
        "evaluation": evaluation,
    }


def get_history(limit: int = 20) -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return history[:limit]


def auto_send_for_user(user: dict[str, Any] | None) -> dict[str, Any] | None:
    """Send inventory alerts if user has alerts enabled and Brevo is configured."""
    if not user or not user.get("alerts_enabled", True):
        return None
    if not brevo.is_configured():
        return None
    to_email = user.get("alert_email") or user.get("email")
    cooldown = user.get("alert_cooldown_hours")
    try:
        return send_inventory_alerts(
            to_email,
            force=False,
            user_email=user.get("email"),
            cooldown_hours=int(cooldown) if cooldown is not None else None,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None
