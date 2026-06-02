"""Alert API routes (Brevo email delivery)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.alerts import brevo
from src.alerts.service import evaluate_alerts, get_history, send_inventory_alerts
from src.auth.deps import get_current_user

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("/status")
def alert_status(user: dict = Depends(get_current_user)):
    return {
        "brevo_configured": brevo.is_configured(),
        "sender": brevo.sender() if brevo.is_configured() else None,
        "settings": {
            "alert_email": user.get("alert_email") or user.get("email"),
            "alerts_enabled": user.get("alerts_enabled", True),
            "alert_time": user.get("alert_time", "10:00"),
            "alert_cooldown_hours": int(user.get("alert_cooldown_hours", 24)),
            "alert_schedule_enabled": bool(user.get("alert_schedule_enabled", True)),
        },
    }


@router.get("/evaluate")
def alert_evaluate(user: dict = Depends(get_current_user)):
    try:
        return evaluate_alerts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/send")
def alert_send(
    force: bool = Query(False, description="Resend even if recently emailed"),
    user: dict = Depends(get_current_user),
):
    if not user.get("alerts_enabled", True):
        raise HTTPException(status_code=400, detail="Alerts are disabled for your account.")
    to_email = user.get("alert_email") or user.get("email")
    cooldown = user.get("alert_cooldown_hours")
    try:
        return send_inventory_alerts(
            to_email,
            force=force,
            user_email=user.get("email"),
            cooldown_hours=int(cooldown) if cooldown is not None else None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/history")
def alert_history(
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    return {"history": get_history(limit)}
