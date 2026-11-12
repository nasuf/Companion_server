"""实名门槛: 手机号 + 身份证二要素核验, 仅限成年人 (COMP 实名, 2026-11-12)。

依据《人工智能拟人化互动服务管理暂行办法》(2026-07-15 施行): 不得向未成年人提供
虚拟伴侣等虚拟亲密关系服务, 须采取有效措施识别未成年人。手机号实名拿不到年龄,
所以在手机号之上再做姓名 + 身份证号核验, 年龄从证件号本身算出。

三档 (settings.realname_mode): off 不拦; phone 只要求短信验证过的手机号;
id_card 手机号 + 核验通过且成年。管理员豁免。

存储边界: 身份证号原文不落库, 只存 HMAC 摘要 (同一张证只能认证一个账号, 防止
拿家长证件给多个账号用) 和脱敏姓名。判定为未成年后账号锁定, 不能换一张证重来,
只有管理员能重置。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

ADULT_AGE = 18
VERIFIED = "verified"
MINOR = "minor"
REJECTED = "rejected"

_HOST = "faceid.tencentcloudapi.com"
_SERVICE = "faceid"
_VERSION = "2018-03-01"
_ACTION = "IdCardVerification"
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)

_ID_PATTERN = re.compile(r"^\d{17}[\dX]$")
_NAME_PATTERN = re.compile(r"^[一-鿿·]{2,20}$")
_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CHECK = "10X98765432"


class RealnameError(Exception):
    """Reason is a stable code; the API maps it to user-facing text."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RealnameState:
    mode: str
    phone_bound: bool
    status: str | None  # verified / minor / rejected / None (never attempted)

    @property
    def required(self) -> list[str]:
        """What the user still has to do before chatting, in order."""
        if self.mode == "off":
            return []
        steps = [] if self.phone_bound else ["phone"]
        if self.mode == "id_card":
            if self.status == MINOR:
                return ["minor_blocked"]
            if self.status != VERIFIED:
                steps.append("id_card")
        return steps


def mode() -> str:
    value = settings.realname_mode.strip().lower()
    return value if value in {"off", "phone", "id_card"} else "off"


def _mock_enabled() -> bool:
    return settings.realname_mock_enabled and not settings.is_production()


def normalize_id_number(raw: str) -> str | None:
    """18 位居民身份证号: 格式、出生日期与校验位都对才返回 (大写 X)。"""
    value = (raw or "").strip().upper()
    if not _ID_PATTERN.match(value):
        return None
    try:
        birth = date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return None
    if birth.year < 1900:
        return None
    total = sum(int(d) * w for d, w in zip(value[:17], _WEIGHTS))
    return value if _CHECK[total % 11] == value[17] else None


def normalize_name(raw: str) -> str | None:
    value = re.sub(r"\s+", "", raw or "").replace("•", "·")
    return value if _NAME_PATTERN.match(value) else None


def birth_date(id_number: str) -> date:
    return date(int(id_number[6:10]), int(id_number[10:12]), int(id_number[12:14]))


def age_on(id_number: str, today: date) -> int:
    born = birth_date(id_number)
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def mask_name(name: str) -> str:
    return name[0] + "*" * (len(name) - 1)


def id_digest(id_number: str) -> str:
    key = settings.realname_hash_key.strip()
    if not key:
        if settings.is_production():
            raise RealnameError("not_configured")
        key = "development-only-realname-key"
    return hmac.new(key.encode(), id_number.encode(), hashlib.sha256).hexdigest()


def _today() -> date:
    return datetime.now(ZoneInfo(settings.schedule_timezone)).date()


async def load_state(user_id: str, *, role: str | None = None) -> RealnameState:
    """role=None (后台任务, 如主动消息) 时自己查角色, 管理员照样豁免。"""
    current = mode()
    if current == "off" or role == "admin":
        return RealnameState("off", True, None)
    if role is None:
        user = await db.user.find_unique(where={"id": user_id})
        if user is not None and user.role == "admin":
            return RealnameState("off", True, None)
    from app.services.phone_auth import PHONE_PROVIDER

    phone = await db.authidentity.find_first(where={"userId": user_id, "provider": PHONE_PROVIDER})
    status = None
    if current == "id_card":
        rows = await db.query_raw(
            "SELECT status FROM user_identity_verifications WHERE user_id=$1", user_id)
        status = rows[0]["status"] if rows else None
    return RealnameState(current, phone is not None, status)


async def chat_block_reason(user_id: str, *, role: str | None = None) -> str | None:
    """None = allowed to chat; otherwise the first missing step (phone / id_card / minor_blocked)."""
    required = (await load_state(user_id, role=role)).required
    return required[0] if required else None


async def _tencent_check(name: str, id_number: str) -> bool:
    """True = 姓名与证件号一致。传输或接口错误抛 RealnameError("provider_unavailable")。"""
    from app.services.sms.tencent import tc3_authorization

    payload = json.dumps({"IdCard": id_number, "Name": name}, ensure_ascii=False,
                         separators=(",", ":"))
    timestamp = int(time.time())
    headers = {
        "Authorization": tc3_authorization(
            payload, timestamp, host=_HOST, service=_SERVICE,
            secret_id=settings.tencent_realname_secret_id.strip(),
            secret_key=settings.tencent_realname_secret_key.strip()),
        "Content-Type": "application/json; charset=utf-8",
        "Host": _HOST,
        "X-TC-Action": _ACTION,
        "X-TC-Version": _VERSION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": settings.tencent_realname_region.strip() or "ap-guangzhou",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            response = await client.post(f"https://{_HOST}/", content=payload.encode(), headers=headers)
            response.raise_for_status()
            body = response.json().get("Response") or {}
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Realname provider transport failure",
                       extra={"event": "realname_provider_error", "reason": type(exc).__name__})
        raise RealnameError("provider_unavailable") from exc
    if body.get("Error"):
        logger.warning("Realname provider API error",
                       extra={"event": "realname_provider_error",
                              "code": (body["Error"] or {}).get("Code")})
        raise RealnameError("provider_unavailable")
    # Result "0" = 一致; "-1" = 不一致; 其余 (查无此人、证件号无效等) 都按不通过处理
    return str(body.get("Result")) == "0"


async def _take_attempt(user_id: str) -> None:
    from app.redis_client import get_redis

    redis = await get_redis()
    key = f"realname:attempts:{user_id}:{_today().isoformat()}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 2 * 86400)
    if count > max(1, settings.realname_daily_attempts):
        raise RealnameError("rate_limited")


async def _record(user_id: str, *, status: str, name: str | None, digest: str | None,
                  adult: bool | None, reason: str | None) -> None:
    await db.execute_raw(
        """INSERT INTO user_identity_verifications
             (id, user_id, status, masked_name, id_digest, is_adult, provider, failure_reason,
              attempts, verified_at, updated_at)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,1,
                   CASE WHEN $3='verified' THEN now() END, now())
           ON CONFLICT (user_id) DO UPDATE SET
             status=EXCLUDED.status, masked_name=EXCLUDED.masked_name,
             id_digest=EXCLUDED.id_digest, is_adult=EXCLUDED.is_adult,
             provider=EXCLUDED.provider, failure_reason=EXCLUDED.failure_reason,
             attempts=user_identity_verifications.attempts+1,
             verified_at=EXCLUDED.verified_at, updated_at=now()""",
        str(uuid.uuid4()), user_id, status, mask_name(name) if name else None, digest, adult,
        "mock" if _mock_enabled() else "tencent", reason)


async def _record_claim(user_id: str, *, status: str, name: str, digest: str, adult: bool) -> None:
    """写入占用证件的一行。两个账号同时用同一张证时, 唯一索引让后到的失败 → id_in_use。"""
    try:
        await _record(user_id, status=status, name=name, digest=digest, adult=adult, reason=None)
    except Exception as exc:
        if "user_identity_verifications_digest_uq" not in str(exc):
            raise
        await _record(user_id, status=REJECTED, name=None, digest=None, adult=None,
                      reason="id_in_use")
        raise RealnameError("id_in_use") from None


async def verify(user_id: str, *, name: str, id_number: str) -> str:
    """跑一次核验, 返回最终状态 (verified / minor)。不通过抛 RealnameError。"""
    if mode() != "id_card":
        raise RealnameError("not_required")
    if not _mock_enabled() and not (settings.tencent_realname_secret_id.strip()
                                    and settings.tencent_realname_secret_key.strip()):
        raise RealnameError("not_configured")
    state = await load_state(user_id)
    if state.status in {VERIFIED, MINOR}:
        return state.status
    if not state.phone_bound:
        raise RealnameError("phone_required")
    clean_name, clean_id = normalize_name(name), normalize_id_number(id_number)
    if clean_name is None:
        raise RealnameError("invalid_name")
    if clean_id is None:
        raise RealnameError("invalid_id_number")
    await _take_attempt(user_id)
    digest = id_digest(clean_id)
    taken = await db.query_raw(
        "SELECT 1 FROM user_identity_verifications WHERE id_digest=$1 AND user_id<>$2 "
        "AND status IN ('verified','minor') LIMIT 1", digest, user_id)
    if taken:
        await _record(user_id, status=REJECTED, name=None, digest=None, adult=None,
                      reason="id_in_use")
        raise RealnameError("id_in_use")
    # 未成年直接按证件号判定, 不必再调核验接口 (也就不为未成年人的信息付费外发)
    if age_on(clean_id, _today()) < ADULT_AGE:
        await _record_claim(user_id, status=MINOR, name=clean_name, digest=digest, adult=False)
        logger.info("Realname minor blocked", extra={"event": "realname_verified", "outcome": "minor"})
        return MINOR
    matched = True if _mock_enabled() else await _tencent_check(clean_name, clean_id)
    if not matched:
        await _record(user_id, status=REJECTED, name=None, digest=None, adult=None,
                      reason="mismatch")
        logger.info("Realname mismatch", extra={"event": "realname_verified", "outcome": "mismatch"})
        raise RealnameError("mismatch")
    await _record_claim(user_id, status=VERIFIED, name=clean_name, digest=digest, adult=True)
    logger.info("Realname verified", extra={"event": "realname_verified", "outcome": "verified"})
    return VERIFIED


async def admin_reset(user_id: str) -> int:
    """管理员重置 (例如证件被冒用的申诉): 删掉这一行, 用户可重新认证。"""
    return int(await db.execute_raw(
        "DELETE FROM user_identity_verifications WHERE user_id=$1", user_id) or 0)


async def summary(*, days: int = 30) -> dict:
    """后台统计: 有界聚合, 不返回姓名、证件摘要或单个用户明细。"""
    days = max(1, min(int(days), 90))
    users = await db.query_raw(
        """SELECT count(*) AS active_users,
             count(*) FILTER (WHERE EXISTS (SELECT 1 FROM auth_identities a
               WHERE a.user_id=u.id AND a.provider='phone')) AS phone_bound
           FROM users u WHERE u.status='active' AND u.role<>'admin' AND u.archived_at IS NULL""")
    statuses = await db.query_raw(
        """SELECT v.status, count(*) AS n FROM user_identity_verifications v
           JOIN users u ON u.id=v.user_id WHERE u.status='active' AND u.role<>'admin'
           GROUP BY v.status""")
    failures = await db.query_raw(
        """SELECT failure_reason AS reason, count(*) AS n FROM user_identity_verifications
           WHERE status='rejected' AND updated_at > now() - ($1 || ' days')::interval
           GROUP BY failure_reason ORDER BY n DESC LIMIT 10""", str(days))
    row = users[0] if users else {}
    return {
        "mode": mode(),
        "provider_configured": bool(settings.tencent_realname_secret_id.strip()
                                    and settings.tencent_realname_secret_key.strip()),
        "active_users": int(row.get("active_users") or 0),
        "phone_bound": int(row.get("phone_bound") or 0),
        "statuses": {r["status"]: int(r["n"]) for r in statuses},
        "recent_failures": [{"reason": r["reason"] or "unknown", "count": int(r["n"])} for r in failures],
        "window_days": days,
        "sampled_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }
