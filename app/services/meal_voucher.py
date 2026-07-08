"""霸王餐 (free-meal voucher) domain service.

Three actors:
* 用户 — one voucher each; activates it with the staff page's rotating code,
  then redeems it with a merchant's fixed code.
* 服务员 — reads a TOTP-style 6-digit code that rotates every 5 minutes.
* 商家 — owns a fixed 6-digit redeem code; sees redemption stats.

The rotating code is derived (HMAC over the time-window index keyed off
``jwt_secret``), so nothing is stored and "rotation" is just the clock moving
to the next window. The admin kill switch lives in the singleton
``system_config`` row; flipping it off makes verification reject everything
and the staff endpoint stop returning codes.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import time
from datetime import UTC, datetime, timedelta, timezone

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

CODE_WINDOW_SECONDS = 300
# Accept the previous window's code for a short grace period after rotation so
# a code read out loud at 4:59 still validates when submitted at 5:05.
CODE_GRACE_SECONDS = 30

VOUCHER_INACTIVE = "inactive"
VOUCHER_ACTIVATED = "activated"
VOUCHER_REDEEMED = "redeemed"

_SIX_DIGITS = re.compile(r"^\d{6}$")


class MealVoucherError(Exception):
    """Domain error with a machine reason + user-facing message."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# ── rotating activation code ─────────────────────────────────────────


def _window_index(ts: float) -> int:
    return int(ts) // CODE_WINDOW_SECONDS


def code_for_window(window: int) -> str:
    """Deterministic 6-digit code for a 5-minute window index."""
    key = f"meal-voucher:{settings.jwt_secret}".encode("utf-8")
    digest = hmac.new(key, str(window).encode("utf-8"), hashlib.sha256).digest()
    return f"{int.from_bytes(digest[:4], 'big') % 1_000_000:06d}"


def current_activation_code(now: float | None = None) -> tuple[str, int]:
    """Return (code, seconds_until_rotation) for the current window."""
    ts = time.time() if now is None else now
    window = _window_index(ts)
    expires_in = CODE_WINDOW_SECONDS - (int(ts) % CODE_WINDOW_SECONDS)
    return code_for_window(window), expires_in


def verify_activation_code(candidate: str, now: float | None = None) -> bool:
    """Match against the current window, or the previous one within grace."""
    text = (candidate or "").strip()
    if not _SIX_DIGITS.fullmatch(text):
        return False
    ts = time.time() if now is None else now
    window = _window_index(ts)
    if hmac.compare_digest(code_for_window(window), text):
        return True
    in_grace = (int(ts) % CODE_WINDOW_SECONDS) < CODE_GRACE_SECONDS
    return in_grace and hmac.compare_digest(code_for_window(window - 1), text)


# ── feature toggle (singleton system_config row) ─────────────────────


async def is_code_enabled() -> bool:
    config = await db.systemconfig.find_unique(where={"id": 1})
    return bool(getattr(config, "mealCodeEnabled", True)) if config else True


async def set_code_enabled(enabled: bool) -> None:
    await db.systemconfig.upsert(
        where={"id": 1},
        data={
            "create": {"id": 1, "mealCodeEnabled": enabled},
            "update": {"mealCodeEnabled": enabled},
        },
    )
    logger.info(
        "meal activation code toggled",
        extra={"event": "meal_code_toggled", "enabled": enabled},
    )


# ── voucher state machine ────────────────────────────────────────────


async def get_or_create_voucher(user_id: str):
    """Lazy creation: every user gets an inactive voucher on first read."""
    voucher = await db.mealvoucher.find_unique(where={"userId": user_id})
    if voucher:
        return voucher
    try:
        return await db.mealvoucher.create(
            data={"user": {"connect": {"id": user_id}}, "status": VOUCHER_INACTIVE}
        )
    except Exception:
        # Unique(userId) race (double-tap on first load): converge on winner.
        voucher = await db.mealvoucher.find_unique(where={"userId": user_id})
        if voucher:
            return voucher
        raise


async def activate_voucher(user_id: str, code: str):
    """未激活 → 已激活: validated against the rotating staff code."""
    if not await is_code_enabled():
        raise MealVoucherError("disabled", "校验码功能暂未开放")

    voucher = await get_or_create_voucher(user_id)
    if voucher.status == VOUCHER_REDEEMED:
        raise MealVoucherError("already_redeemed", "该券已核销，无法重复操作")
    if voucher.status == VOUCHER_ACTIVATED:
        raise MealVoucherError("already_activated", "该券已激活，无需重复激活")

    if not verify_activation_code(code):
        raise MealVoucherError("bad_code", "校验码错误或已过期")

    # Conditional transition: only flips an *inactive* voucher, so a concurrent
    # double-submit can't re-activate (count==0 -> someone else got there first).
    count = await db.mealvoucher.update_many(
        where={"id": voucher.id, "status": VOUCHER_INACTIVE},
        data={"status": VOUCHER_ACTIVATED, "activatedAt": datetime.now(UTC)},
    )
    if not count:
        raise MealVoucherError("already_activated", "该券已激活，无需重复激活")
    logger.info(
        "meal voucher activated",
        extra={"event": "meal_voucher_activated", "user_id": user_id},
    )
    return await db.mealvoucher.find_unique(where={"id": voucher.id})


async def redeem_voucher(user_id: str, redeem_code: str):
    """已激活 → 已核销: matched against a merchant's fixed code."""
    text = (redeem_code or "").strip()
    if not _SIX_DIGITS.fullmatch(text):
        raise MealVoucherError("bad_code", "核销码格式不正确")

    voucher = await get_or_create_voucher(user_id)
    if voucher.status == VOUCHER_REDEEMED:
        raise MealVoucherError("already_redeemed", "该券已核销，无法重复核销")
    if voucher.status != VOUCHER_ACTIVATED:
        raise MealVoucherError("not_activated", "请先激活霸王餐券")

    merchant = await db.mealmerchant.find_first(
        where={"redeemCode": text, "codeActive": True}
    )
    if not merchant:
        raise MealVoucherError("bad_code", "核销码无效，请与商家确认")

    # Conditional transition (activated -> redeemed): concurrent double-redeem
    # loses the race and errors instead of silently re-attributing the voucher.
    count = await db.mealvoucher.update_many(
        where={"id": voucher.id, "status": VOUCHER_ACTIVATED},
        data={
            "status": VOUCHER_REDEEMED,
            "redeemedAt": datetime.now(UTC),
            "merchantId": merchant.id,
        },
    )
    if not count:
        raise MealVoucherError("already_redeemed", "该券已核销，无法重复核销")
    logger.info(
        "meal voucher redeemed",
        extra={
            "event": "meal_voucher_redeemed",
            "user_id": user_id,
            "merchant_id": merchant.id,
        },
    )
    return await db.mealvoucher.find_unique(where={"id": voucher.id})


# ── merchants ────────────────────────────────────────────────────────


async def generate_unique_redeem_code() -> str:
    for _ in range(20):
        code = f"{secrets.randbelow(1_000_000):06d}"
        existing = await db.mealmerchant.find_unique(where={"redeemCode": code})
        if not existing:
            return code
    raise MealVoucherError("code_space", "核销码生成失败，请重试")


def merchant_contact_matches(merchant, contact: str) -> bool:
    """商家 H5 自助身份确认: 联系人姓名或手机号任一精确匹配 (去空格)."""
    text = (contact or "").strip()
    if not text:
        return False
    name = (getattr(merchant, "contactName", None) or "").strip()
    phone = re.sub(r"[\s\-]", "", getattr(merchant, "contactPhone", None) or "")
    return (bool(name) and text == name) or (
        bool(phone) and re.sub(r"[\s\-]", "", text) == phone
    )


# ── display helpers (admin feed / merchant stats) ───────────────────


def mask_phone(phone: str | None) -> str | None:
    if not phone or len(phone) != 11:
        return phone
    return f"{phone[:3]}****{phone[-4:]}"


async def resolve_user_displays(user_ids: list[str]) -> dict[str, str]:
    """Batch: 微信昵称 > 脱敏手机号 > username — for polling feeds.

    Two queries total regardless of row count (identities + users).
    """
    unique_ids = list(dict.fromkeys(user_ids))
    if not unique_ids:
        return {}

    displays: dict[str, str] = {}
    phones: dict[str, str] = {}
    identities = await db.authidentity.find_many(
        where={"userId": {"in": unique_ids}}
    )
    for identity in identities:
        if identity.provider == "wechat" and identity.userId not in displays:
            profile = getattr(identity, "rawProfile", None)
            if isinstance(profile, dict):
                nickname = profile.get("nickname")
                if isinstance(nickname, str) and nickname.strip():
                    displays[identity.userId] = nickname.strip()
        elif identity.provider == "phone":
            phones[identity.userId] = identity.providerAccountId

    missing = [uid for uid in unique_ids if uid not in displays]
    for uid in missing:
        if uid in phones:
            displays[uid] = mask_phone(phones[uid]) or phones[uid]

    still_missing = [uid for uid in unique_ids if uid not in displays]
    if still_missing:
        users = await db.user.find_many(where={"id": {"in": still_missing}})
        for user in users:
            displays[user.id] = user.username

    for uid in unique_ids:
        displays.setdefault(uid, uid)
    return displays


async def resolve_user_display(user_id: str) -> str:
    """Single-user convenience wrapper over the batch resolver."""
    displays = await resolve_user_displays([user_id])
    return displays.get(user_id, user_id)


async def activation_feed(limit: int = 50) -> list[dict]:
    """Recent activations (newest first) with resolved display names."""
    rows = await db.mealvoucher.find_many(
        where={"activatedAt": {"not": None}},
        order={"activatedAt": "desc"},
        take=limit,
        include={"merchant": True},
    )
    displays = await resolve_user_displays([row.userId for row in rows])
    return [
        {
            "user_display": displays.get(row.userId, row.userId),
            "status": row.status,
            "activated_at": row.activatedAt.isoformat() if row.activatedAt else None,
            "redeemed_at": row.redeemedAt.isoformat() if row.redeemedAt else None,
            "merchant_name": row.merchant.name if row.merchant else None,
        }
        for row in rows
    ]


async def voucher_stats() -> dict:
    """Cumulative + today counters for the admin overview (UTC+8 天)."""
    tz_cn = timezone(timedelta(hours=8))
    day_start = datetime.now(tz_cn).replace(hour=0, minute=0, second=0, microsecond=0)
    total_activated = await db.mealvoucher.count(
        where={"activatedAt": {"not": None}}
    )
    total_redeemed = await db.mealvoucher.count(where={"status": VOUCHER_REDEEMED})
    today_activated = await db.mealvoucher.count(
        where={"activatedAt": {"gte": day_start}}
    )
    return {
        "total_activated": total_activated,
        "total_redeemed": total_redeemed,
        "today_activated": today_activated,
    }
