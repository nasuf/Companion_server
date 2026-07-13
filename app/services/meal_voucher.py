"""霸王餐 (free-meal voucher) domain service.

Three actors:
* 用户 — one voucher each; activates it with the staff page's rotating code,
  then presents a short-lived QR code to the merchant.
* 服务员 — reads a TOTP-style 6-digit code that rotates every 5 minutes.
* 商家 — authenticates in H5, scans the user's QR, and sees redemption stats.

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
import time
from datetime import UTC
from datetime import date as date_cls
from datetime import datetime, timedelta
from datetime import time as time_cls
from datetime import timezone

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


_CN_TZ = timezone(timedelta(hours=8))

# 失败留痕 reason: 当日核销量达上限 (先到先得, 超出被拒).
FAILURE_DAILY_CAP = "daily_cap"


# ── 有效期 / 当日边界 (业务口径固定 UTC+8 自然日) ────────────────────


def _cn_day_start(now: datetime | None = None) -> datetime:
    """给定时刻所在 UTC+8 自然日的 00:00 (tz-aware). 用于当日计数/去重."""
    ref = now.astimezone(_CN_TZ) if now else datetime.now(_CN_TZ)
    return ref.replace(hour=0, minute=0, second=0, microsecond=0)


def voucher_expires_at(voucher) -> datetime | None:
    """券的有效期截止时刻: activatedAt + N 天. 未激活 → None.

    Prisma 返回的 activatedAt 为 tz-aware UTC; 结果同样 tz-aware, 可直接跟
    ``datetime.now(UTC)`` 比较.
    """
    activated = getattr(voucher, "activatedAt", None)
    if not activated:
        return None
    if activated.tzinfo is None:
        activated = activated.replace(tzinfo=UTC)
    return activated + timedelta(days=settings.meal_validity_days)


def is_voucher_expired(voucher, now: datetime | None = None) -> bool:
    """仅「已激活未核销」的券会过期: 已过截止时刻即视为过期."""
    if getattr(voucher, "status", None) != VOUCHER_ACTIVATED:
        return False
    expires_at = voucher_expires_at(voucher)
    if not expires_at:
        return False
    ref = now or datetime.now(UTC)
    return ref >= expires_at


# ── rotating activation code ─────────────────────────────────────────
# Codes derive from HMAC(secret, f"{anchor}:{window}"). The anchor is a unix
# timestamp refreshed every time the admin *enables* the feature, so toggling
# off/on regenerates immediately (old codes die, new one gets a full window);
# window indexes and the countdown are computed relative to the anchor.


def _window_and_expiry(ts: float, anchor: int) -> tuple[int, int]:
    rel = int(ts) - anchor
    return rel // CODE_WINDOW_SECONDS, CODE_WINDOW_SECONDS - (rel % CODE_WINDOW_SECONDS)


def code_for_window(window: int, anchor: int = 0) -> str:
    """Deterministic 6-digit code for a window index under a given anchor."""
    key = f"meal-voucher:{settings.jwt_secret}".encode("utf-8")
    digest = hmac.new(
        key, f"{anchor}:{window}".encode("utf-8"), hashlib.sha256
    ).digest()
    return f"{int.from_bytes(digest[:4], 'big') % 1_000_000:06d}"


def current_activation_code(
    now: float | None = None, anchor: int = 0
) -> tuple[str, int]:
    """Return (code, seconds_until_rotation) for the current window."""
    ts = time.time() if now is None else now
    window, expires_in = _window_and_expiry(ts, anchor)
    return code_for_window(window, anchor), expires_in


def verify_activation_code(
    candidate: str, now: float | None = None, anchor: int = 0
) -> bool:
    """Match against the current window, or the previous one within grace."""
    text = (candidate or "").strip()
    if not _SIX_DIGITS.fullmatch(text):
        return False
    ts = time.time() if now is None else now
    window, _ = _window_and_expiry(ts, anchor)
    if hmac.compare_digest(code_for_window(window, anchor), text):
        return True
    in_grace = ((int(ts) - anchor) % CODE_WINDOW_SECONDS) < CODE_GRACE_SECONDS
    return in_grace and hmac.compare_digest(code_for_window(window - 1, anchor), text)


async def get_code_anchor() -> int:
    """Current anchor from the singleton config row (0 = absolute windows)."""
    config = await db.systemconfig.find_unique(where={"id": 1})
    value = getattr(config, "mealCodeAnchor", None) if config else None
    return int(value) if value else 0


async def activation_code_now() -> tuple[str, int]:
    """(code, expires_in) under the currently stored anchor."""
    return current_activation_code(anchor=await get_code_anchor())


async def verify_activation_code_now(candidate: str) -> bool:
    return verify_activation_code(candidate, anchor=await get_code_anchor())


# ── feature toggle (singleton system_config row) ─────────────────────


async def is_code_enabled() -> bool:
    config = await db.systemconfig.find_unique(where={"id": 1})
    return bool(getattr(config, "mealCodeEnabled", True)) if config else True


async def set_code_enabled(enabled: bool) -> None:
    create: dict = {"id": 1, "mealCodeEnabled": enabled}
    update: dict = {"mealCodeEnabled": enabled}
    if enabled:
        # 重新开启 = 重新生成: 刷新锚点让关闭前的码全部失效, 新码满窗口倒计时.
        anchor = int(time.time())
        create["mealCodeAnchor"] = anchor
        update["mealCodeAnchor"] = anchor
    await db.systemconfig.upsert(
        where={"id": 1},
        data={"create": create, "update": update},
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

    if not await verify_activation_code_now(code):
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


async def _redeem_loaded_voucher(voucher, merchant):
    """Apply the only supported redemption transition: merchant QR scan."""
    if voucher.status == VOUCHER_REDEEMED:
        raise MealVoucherError("already_redeemed", "该券已核销，无法重复核销")
    if voucher.status != VOUCHER_ACTIVATED:
        raise MealVoucherError("not_activated", "请先激活霸王餐券")

    # 有效期: 激活满 N 天未核销即过期, 不再允许核销 (spec 需求 1).
    if is_voucher_expired(voucher):
        raise MealVoucherError(
            "expired",
            f"霸王餐券已过有效期（激活后 {settings.meal_validity_days} 天内有效），"
            "无法核销",
        )

    # 每日核销上限 (先到先得, spec 需求 2): 商家码校验通过后, 若当日核销量已达
    # 上限则拒绝并留痕. 放在商家校验之后, 避免把「乱输码」也算成上限失败.
    # 竞态说明: 计数与状态转移之间存在窗口, 极端并发可能轻微超出上限 — 活动为
    # 人工念码, ~1000/日的量级下可接受 (与既有条件转移一致的实用取舍).
    if await today_redeemed_count() >= settings.meal_daily_redeem_cap:
        await record_redemption_failure(voucher.userId, merchant.id, FAILURE_DAILY_CAP)
        raise MealVoucherError(
            "daily_cap",
            "今日霸王餐已被抢完啦，明天再来试试吧（记得在券的有效期内哦）",
        )

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
            "user_id": voucher.userId,
            "merchant_id": merchant.id,
        },
    )
    return await db.mealvoucher.find_unique(where={"id": voucher.id})


async def redeem_voucher_by_merchant(
    voucher_id: str, user_id: str, merchant_id: str
):
    """Merchant-authenticated QR redemption; merchant identity never comes from QR."""
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant or not merchant.codeActive:
        raise MealVoucherError("merchant_disabled", "商家核销功能已停用")
    voucher = await db.mealvoucher.find_unique(where={"id": voucher_id})
    if not voucher or voucher.userId != user_id:
        raise MealVoucherError("invalid_qr", "二维码对应的霸王餐券不存在")
    return await _redeem_loaded_voucher(voucher, merchant)


# ── 每日核销上限 / 失败留痕 ──────────────────────────────────────────


async def today_redeemed_count(now: datetime | None = None) -> int:
    """当日 (UTC+8 自然日) 已核销总数, 用于先到先得上限判定."""
    day_start = _cn_day_start(now)
    return await db.mealvoucher.count(
        where={"status": VOUCHER_REDEEMED, "redeemedAt": {"gte": day_start}}
    )


async def record_redemption_failure(
    user_id: str, merchant_id: str | None, reason: str
) -> None:
    """留痕一条核销失败. 每用户每自然日每 reason 只记 1 条 (写前去重),
    这样后台统计的口径就是「今天有多少人/张券没抢到」而非「点了多少次」.
    留痕失败不应影响主流程 (用户已收到明确提示), 故吞异常仅日志.
    """
    try:
        day_start = _cn_day_start()
        existing = await db.mealredemptionfailure.find_first(
            where={
                "userId": user_id,
                "reason": reason,
                "createdAt": {"gte": day_start},
            }
        )
        if existing:
            return
        await db.mealredemptionfailure.create(
            data={
                "user": {"connect": {"id": user_id}},
                "merchantId": merchant_id,
                "reason": reason,
            }
        )
        logger.info(
            "meal redemption failure recorded",
            extra={
                "event": "meal_redemption_failure",
                "user_id": user_id,
                "reason": reason,
            },
        )
    except Exception:
        logger.exception("failed to record meal redemption failure")


# ── admin corrections (后台清除记录) ─────────────────────────────────


async def clear_redemption(voucher_id: str) -> None:
    """清除核销记录: redeemed → activated (核销时间/商家一并清空).

    条件转移: 只允许从已核销状态回退, 状态不符抛 not_redeemed.
    """
    count = await db.mealvoucher.update_many(
        where={"id": voucher_id, "status": VOUCHER_REDEEMED},
        data={
            "status": VOUCHER_ACTIVATED,
            "redeemedAt": None,
            "merchantId": None,
        },
    )
    if not count:
        raise MealVoucherError("not_redeemed", "该券当前不是已核销状态")
    logger.info(
        "meal redemption cleared",
        extra={"event": "meal_redemption_cleared", "voucher_id": voucher_id},
    )


async def clear_activation(voucher_id: str) -> None:
    """清除校验记录: activated/redeemed → inactive (整券归零, 用户可重新激活)."""
    count = await db.mealvoucher.update_many(
        where={
            "id": voucher_id,
            "status": {"in": [VOUCHER_ACTIVATED, VOUCHER_REDEEMED]},
        },
        data={
            "status": VOUCHER_INACTIVE,
            "activatedAt": None,
            "redeemedAt": None,
            "merchantId": None,
        },
    )
    if not count:
        raise MealVoucherError("not_activated", "该券当前未激活")
    logger.info(
        "meal activation cleared",
        extra={"event": "meal_activation_cleared", "voucher_id": voucher_id},
    )


# ── merchants ────────────────────────────────────────────────────────


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


async def resolve_user_redemption_profiles(user_ids: list[str]) -> dict[str, dict]:
    """Dense admin-facing identity data for voucher redemption operations."""
    unique_ids = list(dict.fromkeys(user_ids))
    if not unique_ids:
        return {}
    users = await db.user.find_many(where={"id": {"in": unique_ids}})
    identities = await db.authidentity.find_many(
        where={"userId": {"in": unique_ids}}, order={"updatedAt": "desc"}
    )
    result: dict[str, dict] = {
        user.id: {
            "user_id": user.id,
            "username": user.username,
            "user_display": user.username,
            "phone_masked": (
                mask_phone(user.username)
                if re.fullmatch(r"\d{11}", user.username or "")
                else None
            ),
            "wechat_nickname": None,
            "wechat_avatar_url": None,
            "wechat_openid": None,
            "wechat_unionid": None,
        }
        for user in users
    }
    for user_id in unique_ids:
        result.setdefault(
            user_id,
            {
                "user_id": user_id,
                "username": user_id,
                "user_display": user_id,
                "phone_masked": None,
                "wechat_nickname": None,
                "wechat_avatar_url": None,
                "wechat_openid": None,
                "wechat_unionid": None,
            },
        )

    seen_wechat: set[str] = set()
    for identity in identities:
        profile = result.get(identity.userId)
        if not profile:
            continue
        if identity.provider == "phone" and not profile["phone_masked"]:
            profile["phone_masked"] = mask_phone(identity.providerAccountId)
        if identity.provider != "wechat" or identity.userId in seen_wechat:
            continue
        seen_wechat.add(identity.userId)
        raw = identity.rawProfile if isinstance(identity.rawProfile, dict) else {}
        nickname = raw.get("nickname")
        avatar = raw.get("headimgurl")
        if isinstance(nickname, str) and nickname.strip():
            profile["wechat_nickname"] = nickname.strip()
            profile["user_display"] = nickname.strip()
        if isinstance(avatar, str) and avatar.strip():
            profile["wechat_avatar_url"] = avatar.strip()
        profile["wechat_openid"] = identity.openid or raw.get("openid")
        profile["wechat_unionid"] = identity.unionid or raw.get("unionid")
    return result


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
            "voucher_id": row.id,
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
    now = datetime.now(UTC)
    day_start = _cn_day_start(now)
    # 过期券: 已激活未核销, 且激活时间早于 (now - 有效期) → 已过 7 天窗口.
    expired_cutoff = now - timedelta(days=settings.meal_validity_days)
    total_activated = await db.mealvoucher.count(
        where={"activatedAt": {"not": None}}
    )
    total_redeemed = await db.mealvoucher.count(where={"status": VOUCHER_REDEEMED})
    today_activated = await db.mealvoucher.count(
        where={"activatedAt": {"gte": day_start}}
    )
    today_redeemed = await db.mealvoucher.count(
        where={"status": VOUCHER_REDEEMED, "redeemedAt": {"gte": day_start}}
    )
    total_expired = await db.mealvoucher.count(
        where={
            "status": VOUCHER_ACTIVATED,
            "activatedAt": {"lte": expired_cutoff},
        }
    )
    # 去重后每用户每日仅 1 条, 计数即「今日多少人/张券没抢到」.
    today_failed = await db.mealredemptionfailure.count(
        where={"reason": FAILURE_DAILY_CAP, "createdAt": {"gte": day_start}}
    )
    return {
        "total_activated": total_activated,
        "total_redeemed": total_redeemed,
        "today_activated": today_activated,
        "today_redeemed": today_redeemed,
        "daily_redeem_cap": settings.meal_daily_redeem_cap,
        "total_expired": total_expired,
        "today_failed": today_failed,
    }


async def expired_vouchers_feed(limit: int = 100) -> list[dict]:
    """已过期券明细 (已激活未核销且超 N 天), 最近激活的在前 + 显示名/截止时刻.

    活动开始不足 N 天时自然为空 (没有券会满 7 天) — 符合「7 天后才有数据」.
    """
    now = datetime.now(UTC)
    expired_cutoff = now - timedelta(days=settings.meal_validity_days)
    rows = await db.mealvoucher.find_many(
        where={"status": VOUCHER_ACTIVATED, "activatedAt": {"lte": expired_cutoff}},
        order={"activatedAt": "desc"},
        take=limit,
    )
    displays = await resolve_user_displays([row.userId for row in rows])
    return [
        {
            "voucher_id": row.id,
            "user_display": displays.get(row.userId, row.userId),
            "activated_at": row.activatedAt.isoformat() if row.activatedAt else None,
            "expired_at": (
                exp.isoformat() if (exp := voucher_expires_at(row)) else None
            ),
        }
        for row in rows
    ]


async def redemption_failures_feed(
    day: date_cls | None = None, limit: int = 200
) -> list[dict]:
    """指定自然日 (默认今天, UTC+8) 因当日上限被拒的核销失败明细.

    每用户每日仅 1 条 (去重), 故行数 = 当日没抢到的人数.
    """
    ref = (
        datetime.combine(day, time_cls(), tzinfo=_CN_TZ)
        if day
        else datetime.now(_CN_TZ)
    )
    day_start = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    rows = await db.mealredemptionfailure.find_many(
        where={
            "reason": FAILURE_DAILY_CAP,
            "createdAt": {"gte": day_start, "lt": day_end},
        },
        order={"createdAt": "desc"},
        take=limit,
    )
    displays = await resolve_user_displays([row.userId for row in rows])
    merchant_ids = [row.merchantId for row in rows if row.merchantId]
    merchant_names: dict[str, str] = {}
    if merchant_ids:
        merchants = await db.mealmerchant.find_many(
            where={"id": {"in": list(dict.fromkeys(merchant_ids))}}
        )
        merchant_names = {m.id: m.name for m in merchants}
    return [
        {
            "user_display": displays.get(row.userId, row.userId),
            "merchant_name": (
                merchant_names.get(row.merchantId) if row.merchantId else None
            ),
            "failed_at": row.createdAt.isoformat() if row.createdAt else None,
        }
        for row in rows
    ]
