"""Verification-code lifecycle: generate, store, rate-limit, verify.

Redis keys:
* ``sms:code:{phone}``     — the active code (TTL = code validity)
* ``sms:cooldown:{phone}`` — per-phone resend cooldown (60s)
* ``sms:daily:{phone}``    — per-phone daily send counter (24h TTL)
* ``sms:tries:{phone}``    — failed verify attempts for the active code
* ``sms:iph:{ip}``         — per-IP hourly send counter (anti SMS-pumping)
* ``sms:ipd:{ip}``         — per-IP daily send counter

Security posture: the code is single-use, expires in 5 minutes, and is
invalidated after 5 wrong attempts (6-digit space is only 1e6 — attempt capping
is what makes it safe). Per-IP caps bound the cost of SMS-pumping attacks that
rotate phone numbers; verify brute force is additionally rate-limited at the
API layer via auth_security.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets

from app.config import settings
from app.redis_client import get_redis
from app.services.sms.tencent import SmsSendError, send_sms_code

logger = logging.getLogger(__name__)

CODE_TTL_MINUTES = 5
_CODE_TTL_S = CODE_TTL_MINUTES * 60
_RESEND_COOLDOWN_S = 60
_DAILY_LIMIT = 10
_MAX_VERIFY_TRIES = 5
# Per-IP caps: generous enough for NAT'd office networks, tight enough to
# bound the cost of a number-rotating SMS-pumping attack from one address.
_IP_HOURLY_LIMIT = 15
_IP_DAILY_LIMIT = 50

_CN_PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


class SmsRateLimited(Exception):
    """Raised when a phone hit the resend cooldown or the daily cap."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalize_cn_phone(raw: str) -> str | None:
    """Return the bare 11-digit mainland-CN number, or None when invalid.

    Accepts optional +86 / 86 prefixes and inline whitespace/dashes.
    """
    cleaned = re.sub(r"[\s\-]", "", raw or "")
    if cleaned.startswith("+86"):
        cleaned = cleaned[3:]
    elif cleaned.startswith("86") and len(cleaned) == 13:
        cleaned = cleaned[2:]
    return cleaned if _CN_PHONE_RE.fullmatch(cleaned) else None


def _sms_configured() -> bool:
    if not settings.sms_enabled:
        return False
    if _mock_mode():
        return True
    return bool(
        settings.tencent_sms_secret_id.strip()
        and settings.tencent_sms_secret_key.strip()
        and settings.tencent_sms_sdk_app_id.strip()
        and settings.tencent_sms_sign_name.strip()
        and settings.tencent_sms_template_id.strip()
    )


def _mock_mode() -> bool:
    # Mock is a dev convenience only — never honored in production.
    return settings.sms_mock_enabled and not settings.is_production()


def sms_login_available() -> bool:
    """Feature probe for the H5 frontend."""
    return _sms_configured()


async def _bump_counter(redis, key: str, ttl_s: int) -> int:
    count = await redis.incr(key)
    if int(count) == 1:
        await redis.expire(key, ttl_s)
    return int(count)


async def send_login_code(phone: str, client_ip: str | None = None) -> None:
    """Generate + store + deliver a verification code for ``phone``.

    Raises SmsRateLimited on cooldown/caps, SmsSendError on delivery failure
    (the stored code is rolled back so a retry regenerates cleanly).
    """
    redis = await get_redis()

    # Order matters: cooldown first, so button-spam during the 60s window is
    # rejected without consuming the shared per-IP quota (one impatient user
    # behind an office NAT must not lock the whole IP out for an hour).
    cooldown_key = f"sms:cooldown:{phone}"
    if not await redis.set(cooldown_key, "1", nx=True, ex=_RESEND_COOLDOWN_S):
        raise SmsRateLimited("cooldown")

    if client_ip:
        hashed = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:16]
        if await _bump_counter(redis, f"sms:iph:{hashed}", 3600) > _IP_HOURLY_LIMIT:
            raise SmsRateLimited("ip_limit")
        if await _bump_counter(redis, f"sms:ipd:{hashed}", 24 * 3600) > _IP_DAILY_LIMIT:
            raise SmsRateLimited("ip_limit")

    if await _bump_counter(redis, f"sms:daily:{phone}", 24 * 3600) > _DAILY_LIMIT:
        raise SmsRateLimited("daily_limit")

    # secrets.randbelow: uniform, cryptographically sourced 6-digit code.
    code = f"{secrets.randbelow(1_000_000):06d}"
    await redis.set(f"sms:code:{phone}", code, ex=_CODE_TTL_S)
    await redis.delete(f"sms:tries:{phone}")

    if _mock_mode():
        # Dev convenience: surface the code in server logs instead of sending.
        logger.info(
            f"[SMS-MOCK] code for {phone[:3]}****{phone[-4:]} = {code}",
            extra={"event": "sms_mock_send"},
        )
        return

    try:
        await send_sms_code(phone, code, CODE_TTL_MINUTES)
    except SmsSendError:
        # Roll back so the user can immediately retry (fresh cooldown/code).
        await redis.delete(f"sms:code:{phone}", cooldown_key)
        raise
    logger.info(
        "sms code sent",
        extra={"event": "sms_code_sent", "phone_tail": phone[-4:]},
    )


async def verify_code(phone: str, code: str) -> bool:
    """Check ``code`` against the active one. Single-use; capped attempts."""
    candidate = (code or "").strip()
    if not re.fullmatch(r"\d{6}", candidate):
        return False

    redis = await get_redis()
    stored = await redis.get(f"sms:code:{phone}")
    if stored is None:
        return False
    stored_text = stored.decode() if isinstance(stored, bytes) else str(stored)

    # Constant-time compare is unnecessary here (attempts are capped), but
    # secrets.compare_digest is free to use and removes the question entirely.
    if secrets.compare_digest(stored_text, candidate):
        await redis.delete(f"sms:code:{phone}", f"sms:tries:{phone}")
        return True

    tries_key = f"sms:tries:{phone}"
    tries = await redis.incr(tries_key)
    if int(tries) == 1:
        await redis.expire(tries_key, _CODE_TTL_S)
    if int(tries) >= _MAX_VERIFY_TRIES:
        # Burn the code: 6-digit space is small, capping attempts is the defense.
        await redis.delete(f"sms:code:{phone}")
        logger.info(
            "sms code burned after too many attempts",
            extra={"event": "sms_code_burned", "phone_tail": phone[-4:]},
        )
    return False
