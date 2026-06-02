"""Auth request/response schemas."""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class OtpRequestBody(BaseModel):
    email: EmailStr
    purpose: str = Field(pattern="^(login|register)$")
    password: str | None = None
    name: str | None = None


class OtpVerifyBody(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=4, max_length=8)
    purpose: str = Field(pattern="^(login|register)$")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=4, max_length=8)
    password: str = Field(min_length=6, max_length=128)


class MessageResponse(BaseModel):
    message: str
    expires_in_minutes: int | None = None


class OtpSentResponse(BaseModel):
    otp_sent: bool
    skip_otp: bool = False
    message: str
    expires_in_minutes: int | None = None


class AlertSettingsRequest(BaseModel):
    alert_email: EmailStr | None = None
    alerts_enabled: bool | None = None
    alert_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    alert_cooldown_hours: int | None = Field(default=None, ge=1, le=168)
    alert_schedule_enabled: bool | None = None


class UserPublic(BaseModel):
    id: Union[int, str]
    email: str
    name: str
    alert_email: str
    alerts_enabled: bool
    email_verified: bool = False
    alert_time: str = "10:00"
    alert_cooldown_hours: int = 24
    alert_schedule_enabled: bool = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
