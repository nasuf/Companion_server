"""Authentication abuse controls and audit logging."""

from __future__ import annotations

import hashlib
import logging
import time

from fastapi import HTTPException, Request, status

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_LOGIN_WINDOW_S = 15 * 60
_LOGIN_MAX_FAILURES = 5
_REGISTER_WINDOW_S = 60 * 60
_REGISTER_MAX_ATTEMPTS = 5


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _hash_part(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:16]


def _login_key(username: str, ip: str) -> str:
    return f"auth:login_fail:{_hash_part(username)}:{_hash_part(ip)}"


def _register_key(ip: str) -> str:
    return f"auth:register:{_hash_part(ip)}"


async def enforce_login_rate_limit(request: Request, username: str) -> None:
    """Block repeated failed login attempts for a username+IP pair."""
    ip = _client_ip(request)
    try:
        redis = await get_redis()
        failures = await redis.get(_login_key(username, ip))
        if failures is not None and int(failures) >= _LOGIN_MAX_FAILURES:
            audit_auth_event(
                "login_rate_limited",
                username=username,
                ip=ip,
                outcome="blocked",
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="登录尝试过于频繁，请稍后再试",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            f"Login rate-limit check failed open: {e}",
            extra={"event": "auth_rate_limit_error", "kind": "login_check"},
        )


async def record_login_failure(request: Request, username: str) -> None:
    ip = _client_ip(request)
    try:
        redis = await get_redis()
        key = _login_key(username, ip)
        count = await redis.incr(key)
        if int(count) == 1:
            await redis.expire(key, _LOGIN_WINDOW_S)
        audit_auth_event("login_failed", username=username, ip=ip, outcome="failed")
    except Exception as e:
        logger.warning(
            f"Login failure rate-limit update failed: {e}",
            extra={"event": "auth_rate_limit_error", "kind": "login_failure"},
        )


async def clear_login_failures(request: Request, username: str) -> None:
    ip = _client_ip(request)
    try:
        redis = await get_redis()
        await redis.delete(_login_key(username, ip))
    except Exception as e:
        logger.debug(
            f"Login failure clear failed: {e}",
            extra={"event": "auth_rate_limit_error", "kind": "login_clear"},
        )


async def enforce_register_rate_limit(request: Request) -> None:
    """Limit account creation attempts by IP."""
    ip = _client_ip(request)
    try:
        redis = await get_redis()
        key = _register_key(ip)
        count = await redis.incr(key)
        if int(count) == 1:
            await redis.expire(key, _REGISTER_WINDOW_S)
        if int(count) > _REGISTER_MAX_ATTEMPTS:
            audit_auth_event(
                "register_rate_limited",
                ip=ip,
                outcome="blocked",
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="注册尝试过于频繁，请稍后再试",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            f"Register rate-limit check failed open: {e}",
            extra={"event": "auth_rate_limit_error", "kind": "register_check"},
        )


def audit_auth_event(
    event_type: str,
    *,
    username: str | None = None,
    user_id: str | None = None,
    ip: str | None = None,
    outcome: str,
) -> None:
    """Structured auth audit log without storing passwords or raw tokens."""
    logger.info(
        f"[AUTH-AUDIT] {event_type} outcome={outcome}",
        extra={
            "event": "auth_audit",
            "event_type": event_type,
            "username_hash": _hash_part(username) if username else None,
            "user_id": user_id,
            "ip_hash": _hash_part(ip) if ip else None,
            "outcome": outcome,
            "ts_ms": int(time.time() * 1000),
        },
    )


def audit_auth_request_event(
    event_type: str,
    request: Request,
    *,
    username: str | None = None,
    user_id: str | None = None,
    outcome: str,
) -> None:
    audit_auth_event(
        event_type,
        username=username,
        user_id=user_id,
        ip=_client_ip(request),
        outcome=outcome,
    )
