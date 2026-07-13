from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(user_id: str, role: str, *, expiry_hours: int | None = None) -> str:
    now = datetime.now(UTC)
    hours = settings.jwt_expiry_hours if expiry_hours is None else expiry_hours
    payload = {
        "sub": user_id,
        "role": role,
        "exp": now + timedelta(hours=hours),
        "iat": now,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
