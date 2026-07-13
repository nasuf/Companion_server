"""Short-lived, single-use QR redemption grants backed by Redis."""

from __future__ import annotations

import json
import re
import secrets
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.redis_client import get_redis

QR_PREFIX = "CPMEAL:1:"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{40,80}$")
_TOKEN_KEY_PREFIX = "meal:qr:token:"
_ACTIVE_KEY_PREFIX = "meal:qr:active:"

_ISSUE_LUA = """
local old_token = redis.call('get', KEYS[1])
if old_token then
  redis.call('del', ARGV[1] .. old_token)
end
redis.call('set', KEYS[2], ARGV[2], 'EX', ARGV[3])
redis.call('set', KEYS[1], ARGV[4], 'EX', ARGV[3])
return 1
"""

_CONSUME_LUA = """
local data = redis.call('get', KEYS[1])
if not data then
  return nil
end
redis.call('del', KEYS[1])
local current = redis.call('get', KEYS[2])
if current == ARGV[1] then
  redis.call('del', KEYS[2])
end
return data
"""


class MealQRError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _token_key(token: str) -> str:
    return f"{_TOKEN_KEY_PREFIX}{token}"


def _active_key(voucher_id: str) -> str:
    return f"{_ACTIVE_KEY_PREFIX}{voucher_id}"


async def issue(voucher_id: str, user_id: str) -> dict:
    """Create the only currently valid QR grant for a voucher."""
    token = secrets.token_urlsafe(32)
    ttl = max(20, min(int(settings.meal_qr_ttl_seconds), 300))
    now = datetime.now(UTC)
    data = json.dumps(
        {
            "voucher_id": voucher_id,
            "user_id": user_id,
            "issued_at": now.isoformat(),
        },
        separators=(",", ":"),
    )
    redis = await get_redis()
    await redis.eval(
        _ISSUE_LUA,
        2,
        _active_key(voucher_id),
        _token_key(token),
        _TOKEN_KEY_PREFIX,
        data,
        ttl,
        token,
    )
    return {
        "value": f"{QR_PREFIX}{token}",
        "expires_in": ttl,
        "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
    }


async def consume(value: str) -> dict:
    """Atomically consume a QR grant. A scan result can succeed at most once."""
    text = (value or "").strip()
    if not text.startswith(QR_PREFIX):
        raise MealQRError("invalid_qr", "不是有效的霸王餐核销二维码")
    token = text[len(QR_PREFIX) :]
    if not _TOKEN_RE.fullmatch(token):
        raise MealQRError("invalid_qr", "不是有效的霸王餐核销二维码")

    redis = await get_redis()
    raw = await redis.get(_token_key(token))
    if not raw:
        raise MealQRError("expired_qr", "二维码已过期，请让顾客刷新后重试")
    try:
        preview = json.loads(raw)
        voucher_id = str(preview["voucher_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MealQRError("invalid_qr", "二维码数据无效") from exc

    consumed = await redis.eval(
        _CONSUME_LUA,
        2,
        _token_key(token),
        _active_key(voucher_id),
        token,
    )
    if not consumed:
        raise MealQRError("expired_qr", "二维码已过期或已被使用")
    try:
        data = json.loads(consumed)
        return {
            "voucher_id": str(data["voucher_id"]),
            "user_id": str(data["user_id"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MealQRError("invalid_qr", "二维码数据无效") from exc
