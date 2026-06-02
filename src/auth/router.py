"""Authentication API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.auth.database import get_password_hash, get_user_by_email, update_alert_settings
from src.auth.deps import get_current_user
from src.auth.otp import request_otp, request_password_reset, reset_password_with_otp, verify_otp
from src.auth.schemas import (
    AlertSettingsRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    OtpRequestBody,
    OtpSentResponse,
    OtpVerifyBody,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserPublic,
)
from src.auth.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["Auth"])


def _public(user: dict) -> UserPublic:
    return UserPublic(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        alert_email=user["alert_email"],
        alerts_enabled=user["alerts_enabled"],
        email_verified=user.get("email_verified", False),
        alert_time=user.get("alert_time", "10:00"),
        alert_cooldown_hours=int(user.get("alert_cooldown_hours", 24)),
        alert_schedule_enabled=bool(user.get("alert_schedule_enabled", True)),
    )


def _token_response(user: dict) -> TokenResponse:
    token = create_access_token(user["id"], user["email"])
    return TokenResponse(access_token=token, user=_public(user))


@router.post("/otp/request", response_model=OtpSentResponse)
def otp_request(body: OtpRequestBody):
    if body.purpose == "login":
        raise HTTPException(
            status_code=400,
            detail="Sign-in codes are disabled. Use your email and password instead.",
        )
    try:
        return request_otp(
            body.email,
            body.purpose,
            password=body.password,
            name=body.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/otp/verify", response_model=TokenResponse)
def otp_verify(body: OtpVerifyBody):
    if body.purpose == "login":
        raise HTTPException(
            status_code=400,
            detail="Sign-in codes are disabled. Use your email and password instead.",
        )
    try:
        user = verify_otp(body.email, body.otp, body.purpose)
        return _token_response(user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest):
    """Legacy direct register — prefer OTP flow via /otp/request + /otp/verify."""
    if get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    from src.auth.database import create_user

    user = create_user(body.email, body.name, hash_password(body.password))
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    stored = get_password_hash(body.email)
    if not stored or not verify_password(body.password, stored):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _token_response(user)


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(body: ForgotPasswordRequest):
    try:
        result = request_password_reset(body.email)
        return MessageResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/reset-password", response_model=TokenResponse)
def reset_password(body: ResetPasswordRequest):
    try:
        user = reset_password_with_otp(body.email, body.otp, body.password)
        return _token_response(user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/me", response_model=UserPublic)
def me(user: dict = Depends(get_current_user)):
    return _public(user)


@router.patch("/alert-settings", response_model=UserPublic)
def alert_settings(body: AlertSettingsRequest, user: dict = Depends(get_current_user)):
    updated = update_alert_settings(
        user["id"],
        alert_email=body.alert_email,
        alerts_enabled=body.alerts_enabled,
        alert_time=body.alert_time,
        alert_cooldown_hours=body.alert_cooldown_hours,
        alert_schedule_enabled=body.alert_schedule_enabled,
    )
    return _public(updated)
