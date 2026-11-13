"""实名门槛: 手机号 + 身份证二要素核验, 仅限成年人 (COMP 实名, 2026-11-12)。

依据《人工智能拟人化互动服务管理暂行办法》(2026-07-15 施行): 不得向未成年人提供
虚拟伴侣等虚拟亲密关系服务, 须采取有效措施识别未成年人。手机号实名拿不到年龄,
所以在手机号之上再做姓名 + 身份证号核验, 年龄从证件号本身算出。

开关在后台「系统设置 → 实名认证」实时修改 (system_config.realname_policy, 2s 内全进程
生效; 从未保存过时用环境变量 REALNAME_MODE 兜底):
  mode  off 不拦 / phone 只要求短信验证过的手机号 / id_card 手机号 + 核验通过且成年
  scope all 全部用户 / users 只对 user_ids 名单里的账号生效 (内测期先在新版 App 上试,
        旧版 App 用户不受影响)
管理员豁免。

存储 (2026-11-12 用户决定全部入库, 仅用于内部用户分析; 仅限核验通过的成年人,
判定未成年的账号只保留证件摘要与锁定状态): 证件号解析出的出生日期、
性别、户籍地区码与省份明文存; 姓名与证件号只存 AES-GCM 密文, 密钥由
REALNAME_HASH_KEY 派生, 库里没有明文。另存 HMAC 摘要保证一张证只认证一个账号 (防止
拿家长证件给多个账号用)。判定为未成年后账号锁定, 换一张证也不能重来, 只有管理员能重置。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
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
MODES = ("off", "phone", "id_card")
SCOPES = ("all", "users")
MAX_POLICY_USERS = 500
# GB/T 2260 省级码 (证件号前两位)。地区是办证时的户籍地, 不是现居地。
PROVINCES = {
    "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
    "21": "辽宁", "22": "吉林", "23": "黑龙江", "31": "上海", "32": "江苏",
    "33": "浙江", "34": "安徽", "35": "福建", "36": "江西", "37": "山东",
    "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
    "46": "海南", "50": "重庆", "51": "四川", "52": "贵州", "53": "云南",
    "54": "西藏", "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏",
    "65": "新疆", "71": "台湾", "81": "香港", "82": "澳门",
}


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


@dataclass(frozen=True)
class Policy:
    mode: str = "off"
    scope: str = "all"
    user_ids: frozenset[str] = field(default_factory=frozenset)
    source: str = "env"  # admin = 后台保存过; env = 环境变量兜底

    def applies_to(self, user_id: str) -> bool:
        return self.mode != "off" and (self.scope == "all" or user_id in self.user_ids)

    def public(self) -> dict:
        return {"mode": self.mode, "scope": self.scope,
                "user_ids": sorted(self.user_ids), "source": self.source}


def parse_policy(raw: object) -> Policy | None:
    """后台存的 {mode, scope, user_ids}; 结构不对返回 None (调用方回落环境变量)。"""
    if not isinstance(raw, dict):
        return None
    value_mode, scope, ids = raw.get("mode"), raw.get("scope", "all"), raw.get("user_ids", [])
    if value_mode not in MODES or scope not in SCOPES or not isinstance(ids, list):
        return None
    if not all(isinstance(i, str) and i for i in ids) or len(ids) > MAX_POLICY_USERS:
        return None
    return Policy(value_mode, scope, frozenset(ids), "admin")


def policy() -> Policy:
    from app.services.runtime_config import realname_policy_overrides

    stored = parse_policy(realname_policy_overrides())
    if stored is not None:
        return stored
    value = settings.realname_mode.strip().lower()
    return Policy(value if value in MODES else "off")


def mode() -> str:
    return policy().mode


def provider_configured() -> bool:
    return _mock_enabled() or bool(settings.tencent_realname_secret_id.strip()
                                   and settings.tencent_realname_secret_key.strip())


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


def mask_id_number(id_number: str) -> str:
    return id_number[0] + "*" * (len(id_number) - 2) + id_number[-1]


def _secret() -> bytes:
    key = settings.realname_hash_key.strip()
    if not key:
        if settings.is_production():
            raise RealnameError("not_configured")
        key = "development-only-realname-key"
    return key.encode()


def id_digest(id_number: str) -> str:
    return hmac.new(_secret(), id_number.encode(), hashlib.sha256).hexdigest()


def _cipher_key() -> bytes:
    # 与摘要用不同的派生键: 摘要泄露不等于能解密
    return hmac.new(_secret(), b"realname-encryption-v1", hashlib.sha256).digest()


def encrypt(value: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    return "v1:" + base64.b64encode(nonce + AESGCM(_cipher_key()).encrypt(nonce, value.encode(), None)).decode()


def decrypt(token: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not token.startswith("v1:"):
        raise ValueError("unknown ciphertext version")
    raw = base64.b64decode(token[3:])
    return AESGCM(_cipher_key()).decrypt(raw[:12], raw[12:], None).decode()


@dataclass(frozen=True)
class IdProfile:
    birth_date: date
    gender: str  # male / female
    region_code: str
    province: str | None


def profile_of(id_number: str) -> IdProfile:
    """证件号本身包含的信息: 出生日期、性别 (第 17 位奇男偶女)、户籍地区码。"""
    return IdProfile(birth_date(id_number), "male" if int(id_number[16]) % 2 else "female",
                     id_number[:6], PROVINCES.get(id_number[:2]))


def _today() -> date:
    return datetime.now(ZoneInfo(settings.schedule_timezone)).date()


async def load_state(user_id: str, *, role: str | None = None) -> RealnameState:
    """role=None (后台任务, 如主动消息) 时自己查角色。

    开关只对指定用户生效时, 名单外的用户按 off 处理 —— 旧版 App 用户不受影响。
    管理员在「全部用户」下豁免; 被明确加进名单的管理员照样要实名 (用于内部真机走查)。"""
    current_policy = policy()
    current = current_policy.mode
    if not current_policy.applies_to(user_id):
        return RealnameState("off", True, None)
    listed = current_policy.scope == "users"
    if role == "admin" and not listed:
        return RealnameState("off", True, None)
    if role is None and not listed:
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
                  adult: bool | None, reason: str | None, id_number: str | None = None) -> None:
    """未通过 (rejected) 时 name/id_number 为 None: 不一致的信息不入库。"""
    info = profile_of(id_number) if id_number else None
    await db.execute_raw(
        """INSERT INTO user_identity_verifications
             (id, user_id, status, masked_name, id_digest, is_adult, provider, failure_reason,
              birth_date, gender, region_code, province, name_encrypted, id_number_encrypted,
              attempts, verified_at, updated_at)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::date,$10,$11,$12,$13,$14,1,
                   CASE WHEN $3='verified' THEN now() END, now())
           ON CONFLICT (user_id) DO UPDATE SET
             status=EXCLUDED.status, masked_name=EXCLUDED.masked_name,
             id_digest=EXCLUDED.id_digest, is_adult=EXCLUDED.is_adult,
             provider=EXCLUDED.provider, failure_reason=EXCLUDED.failure_reason,
             birth_date=EXCLUDED.birth_date, gender=EXCLUDED.gender,
             region_code=EXCLUDED.region_code, province=EXCLUDED.province,
             name_encrypted=EXCLUDED.name_encrypted,
             id_number_encrypted=EXCLUDED.id_number_encrypted,
             attempts=user_identity_verifications.attempts+1,
             verified_at=EXCLUDED.verified_at, updated_at=now()""",
        str(uuid.uuid4()), user_id, status, mask_name(name) if name else None, digest, adult,
        "mock" if _mock_enabled() else "tencent", reason,
        info.birth_date.isoformat() if info else None, info.gender if info else None,
        info.region_code if info else None, info.province if info else None,
        encrypt(name) if name and id_number else None,
        encrypt(id_number) if id_number else None)


async def _record_claim(user_id: str, *, status: str, name: str | None, digest: str, adult: bool,
                        id_number: str | None) -> None:
    """写入占用证件的一行。两个账号同时用同一张证时, 唯一索引让后到的失败 → id_in_use。"""
    try:
        await _record(user_id, status=status, name=name, digest=digest, adult=adult, reason=None,
                      id_number=id_number)
    except Exception as exc:
        if "user_identity_verifications_digest_uq" not in str(exc):
            raise
        await _record(user_id, status=REJECTED, name=None, digest=None, adult=None,
                      reason="id_in_use")
        raise RealnameError("id_in_use") from None


async def verify(user_id: str, *, name: str, id_number: str) -> str:
    """跑一次核验, 返回最终状态 (verified / minor)。不通过抛 RealnameError。"""
    current = policy()
    if current.mode != "id_card" or not current.applies_to(user_id):
        raise RealnameError("not_required")
    if not provider_configured():
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
        # 未成年人的个人信息不保存 (不满 14 岁需监护人同意, 我们没有): 只留证件摘要
        # 与锁定状态, 用来防止换一个账号再用同一张证注册
        await _record_claim(user_id, status=MINOR, name=None, digest=digest, adult=False,
                            id_number=None)
        logger.info("Realname minor blocked", extra={"event": "realname_verified", "outcome": "minor"})
        return MINOR
    matched = True if _mock_enabled() else await _tencent_check(clean_name, clean_id)
    if not matched:
        await _record(user_id, status=REJECTED, name=None, digest=None, adult=None,
                      reason="mismatch")
        logger.info("Realname mismatch", extra={"event": "realname_verified", "outcome": "mismatch"})
        raise RealnameError("mismatch")
    await _record_claim(user_id, status=VERIFIED, name=clean_name, digest=digest, adult=True,
                        id_number=clean_id)
    logger.info("Realname verified", extra={"event": "realname_verified", "outcome": "verified"})
    return VERIFIED


async def record_of(user_id: str) -> dict | None:
    """用户自己的认证记录 (隐私中心展示), 与门槛是否对他生效无关。只给打码后的姓名和证件号。"""
    rows = await db.query_raw(
        "SELECT status, masked_name, id_number_encrypted, verified_at "
        "FROM user_identity_verifications WHERE user_id=$1", user_id)
    if not rows:
        return None
    row = rows[0]
    masked_id = None
    if row.get("id_number_encrypted"):
        try:
            masked_id = mask_id_number(decrypt(row["id_number_encrypted"]))
        except Exception:  # 密钥轮换等解不开时只是不展示证件号
            logger.warning("Realname record decrypt failed user=%s", user_id)
    verified_at = row.get("verified_at")
    return {"status": row["status"], "masked_name": row.get("masked_name"),
            "masked_id_number": masked_id,
            "verified_at": verified_at.isoformat() if hasattr(verified_at, "isoformat") else verified_at}


async def admin_reset(user_id: str) -> int:
    """管理员重置 (例如证件被冒用的申诉): 删掉这一行, 用户可重新认证。"""
    return int(await db.execute_raw(
        "DELETE FROM user_identity_verifications WHERE user_id=$1", user_id) or 0)


async def summary(*, days: int = 30) -> dict:
    """后台统计: 有界聚合, 不返回姓名、证件摘要或单个用户明细。

    管理员账号也计入: 内部账号会被加进名单走查, 认证了就要能在统计里看到。"""
    days = max(1, min(int(days), 90))
    users = await db.query_raw(
        """SELECT count(*) AS active_users,
             count(*) FILTER (WHERE EXISTS (SELECT 1 FROM auth_identities a
               WHERE a.user_id=u.id AND a.provider='phone')) AS phone_bound
           FROM users u WHERE u.status='active' AND u.archived_at IS NULL""")
    statuses = await db.query_raw(
        """SELECT v.status, count(*) AS n FROM user_identity_verifications v
           JOIN users u ON u.id=v.user_id WHERE u.status='active'
           GROUP BY v.status""")
    failures = await db.query_raw(
        """SELECT failure_reason AS reason, count(*) AS n FROM user_identity_verifications
           WHERE status='rejected' AND updated_at > now() - ($1 || ' days')::interval
           GROUP BY failure_reason ORDER BY n DESC LIMIT 10""", str(days))
    # 用户分析: 只统计核验通过的账号, 只给分布, 不给单个用户
    profile_rows = await db.query_raw(
        """SELECT v.gender, v.province,
             CASE WHEN v.birth_date IS NULL THEN NULL
                  ELSE date_part('year', age(CURRENT_DATE, v.birth_date))::int END AS age
           FROM user_identity_verifications v JOIN users u ON u.id=v.user_id
           WHERE v.status='verified' AND u.status='active'
           LIMIT 100000""")
    genders: dict[str, int] = {}
    provinces: dict[str, int] = {}
    ages: dict[str, int] = {}
    for r in profile_rows:
        if r.get("gender"):
            genders[r["gender"]] = genders.get(r["gender"], 0) + 1
        place = r.get("province") or "未知"
        provinces[place] = provinces.get(place, 0) + 1
        if r.get("age") is not None:
            bucket = _age_bucket(int(r["age"]))
            ages[bucket] = ages.get(bucket, 0) + 1
    row = users[0] if users else {}
    current = policy()
    return {
        "mode": current.mode,
        "policy": current.public(),
        "provider_configured": provider_configured(),
        "genders": genders,
        "age_buckets": ages,
        "provinces": [{"province": k, "count": v}
                      for k, v in sorted(provinces.items(), key=lambda kv: -kv[1])[:15]],
        "active_users": int(row.get("active_users") or 0),
        "phone_bound": int(row.get("phone_bound") or 0),
        "statuses": {r["status"]: int(r["n"]) for r in statuses},
        "recent_failures": [{"reason": r["reason"] or "unknown", "count": int(r["n"])} for r in failures],
        "window_days": days,
        "sampled_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }


def _age_bucket(age: int) -> str:
    for upper, label in ((24, "18-24"), (29, "25-29"), (34, "30-34"), (39, "35-39"), (49, "40-49")):
        if age <= upper:
            return label
    return "50+"
