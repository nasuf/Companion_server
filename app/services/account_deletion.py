"""用户自助注销, 7 天冷静期 (2026-11-13 用户决定)。

- 申请: 已绑手机号的账号要短信验证码确认; 写 deletion_requested_at / deletion_due_at,
  立即断开聊天连接并下线。客户端随即退出登录。
- 冷静期内: 不发主动消息、不发推送。
- 撤销: 冷静期内任何一次登录 (含恢复会话) 都自动撤销, 登录返回 deletion_cancelled=True。
- 到期: 定时任务调用 hard_delete_user_data 彻底删除全部数据 (与后台删除用户同一条路径)。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.db import db

logger = logging.getLogger(__name__)

COOLDOWN = timedelta(days=7)
_SCAN_LIMIT = 20


class DeletionError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _now() -> datetime:
    # 库里 timestamp 不带时区, 存 UTC naive
    return datetime.now(UTC).replace(tzinfo=None)


async def send_code(user_id: str, *, client_ip: str | None) -> bool:
    """给账号已绑定的手机号发注销验证码。没绑手机号返回 False (不需要验证码)。"""
    from app.services import sms
    from app.services.phone_auth import PHONE_PROVIDER

    phone = await db.authidentity.find_first(where={"userId": user_id, "provider": PHONE_PROVIDER})
    if phone is None:
        return False
    await sms.send_login_code(phone.providerAccountId, client_ip=client_ip)
    return True


async def request_deletion(user_id: str, *, code: str | None) -> datetime:
    """登记注销申请, 返回到期时间 (UTC)。"""
    from app.services import sms
    from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME
    from app.services.phone_auth import PHONE_PROVIDER

    user = await db.user.find_unique(where={"id": user_id})
    if user is None or user.status != "active":
        raise DeletionError("unavailable")
    if user.role == "admin" or user.username == TEMPLATE_SYSTEM_USERNAME:
        raise DeletionError("protected")
    phone = await db.authidentity.find_first(where={"userId": user_id, "provider": PHONE_PROVIDER})
    if phone is not None:
        if not code or not await sms.verify_code(phone.providerAccountId, code):
            raise DeletionError("bad_code")
    now = _now()
    due = now + COOLDOWN
    await db.execute_raw(
        "UPDATE users SET deletion_requested_at=$2::timestamp, deletion_due_at=$3::timestamp "
        "WHERE id=$1 AND status='active'",
        user_id, now.isoformat(), due.isoformat())
    await _disconnect(user_id)
    logger.info("Account deletion requested", extra={"event": "account_deletion", "outcome": "requested",
                                                     "user_id": user_id})
    return due


async def _disconnect(user_id: str) -> None:
    from app.redis_client import get_redis
    from app.services.notifications.presence import remove_online
    from app.services.runtime.ws_manager import manager

    try:
        await manager.revoke_user(user_id, redis=await get_redis())
    except Exception as exc:  # 断开失败不影响登记, 客户端本来就会退出登录
        logger.warning("Deletion disconnect failed: %s", type(exc).__name__)
    await remove_online(user_id)


async def cancel_on_login(user) -> bool:
    """登录时调用: 冷静期内的申请自动撤销。返回是否撤销了。"""
    if getattr(user, "deletionDueAt", None) is None:
        return False
    rows = await db.query_raw(
        "UPDATE users SET deletion_requested_at=NULL, deletion_due_at=NULL "
        "WHERE id=$1 AND deletion_due_at IS NOT NULL AND status='active' RETURNING id", user.id)
    if rows:
        logger.info("Account deletion cancelled by login",
                    extra={"event": "account_deletion", "outcome": "cancelled", "user_id": user.id})
    return bool(rows)


async def pending(user_id: str) -> bool:
    rows = await db.query_raw(
        "SELECT deletion_due_at IS NOT NULL AS pending FROM users WHERE id=$1", user_id)
    return bool(rows) and rows[0].get("pending") is True


async def purge_due(*, now: datetime | None = None) -> dict[str, int]:
    """到期的申请逐个彻底删除。删除失败的留到下一轮重试。"""
    from app.services.runtime.data_reset import hard_delete_user_data

    current = (now or _now()).replace(tzinfo=None)
    rows = await db.query_raw(
        "SELECT id FROM users WHERE deletion_due_at IS NOT NULL AND deletion_due_at <= $1::timestamp "
        "AND role<>'admin' ORDER BY deletion_due_at LIMIT $2",
        current.isoformat(), _SCAN_LIMIT)
    stats = {"due": len(rows), "deleted": 0, "failed": 0}
    for row in rows:
        try:
            await hard_delete_user_data(row["id"])
            stats["deleted"] += 1
            logger.info("Account deleted after cooldown",
                        extra={"event": "account_deletion", "outcome": "deleted", "user_id": row["id"]})
        except Exception as exc:
            stats["failed"] += 1
            logger.warning("Account deletion failed user=%s error=%s", row["id"], type(exc).__name__,
                           extra={"event": "account_deletion", "outcome": "failed", "user_id": row["id"]})
    return stats
