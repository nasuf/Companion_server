"""后台实名开关: 档位 + 全量/指定用户。保存后 2s 内全进程生效 (system_config 触发器递增配置版本)。"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.admin.proactive_tuning import _utc_iso
from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.identity import realname
from app.services.runtime_config import load_caches

router = APIRouter(prefix="/admin-api/realname-policy", tags=["admin"],
                   dependencies=[Depends(require_admin_jwt)])
CHANGES_LIMIT = 30


class PolicyPut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["off", "phone", "id_card"]
    scope: Literal["all", "users"]
    user_ids: list[str] = Field(default_factory=list, max_length=realname.MAX_POLICY_USERS)
    # 读取时拿到的当前值; 和库里不一致说明别人改过 → 409, 防止互相覆盖
    expected: dict | None = None


def _stored(raw) -> dict | None:
    parsed = realname.parse_policy(raw)
    return {k: v for k, v in parsed.public().items() if k != "source"} if parsed else None


async def _payload() -> dict:
    rows = await db.query_raw("SELECT realname_policy FROM system_config WHERE id=1")
    if not rows:
        raise HTTPException(503, "配置尚未初始化")
    stored = _stored(rows[0].get("realname_policy"))
    effective = realname.parse_policy(stored) or realname.Policy(
        realname.policy().mode if stored is None else "off")
    changes = await db.query_raw(
        """SELECT c.id, c.created_at, c.changes, c.admin_id, u.username AS admin_name
           FROM realname_policy_changes c LEFT JOIN users u ON u.id=c.admin_id
           ORDER BY c.created_at DESC, c.id DESC LIMIT $1""", CHANGES_LIMIT)
    users = []
    if effective.user_ids:
        users = await db.query_raw(
            """SELECT u.id, u.username, u.display_name, v.status AS realname_status
               FROM users u LEFT JOIN user_identity_verifications v ON v.user_id=u.id
               WHERE u.id = ANY($1::text[])""", sorted(effective.user_ids))
    return {
        "stored": stored,
        "effective": (realname.parse_policy(stored) or realname.policy()).public(),
        "provider_configured": realname.provider_configured(),
        "users": users,
        "refresh_seconds": 2,
        "sampled_at": datetime.now(UTC).isoformat(),
        "changes_limit": CHANGES_LIMIT,
        "changes": [{**row, "created_at": _utc_iso(row["created_at"]),
                     "changes": json.loads(row["changes"]) if isinstance(row["changes"], str) else row["changes"]}
                    for row in changes],
    }


@router.get("")
async def get_policy():
    return await _payload()


@router.put("")
async def put_policy(payload: PolicyPut, admin: dict = Depends(require_admin_jwt)):
    ids = sorted({i.strip() for i in payload.user_ids if i.strip()})
    if payload.mode == "id_card" and not realname.provider_configured():
        raise HTTPException(400, "服务端未配置实名核验密钥，不能开启身份证核验")
    if payload.scope == "users" and payload.mode != "off" and not ids:
        raise HTTPException(400, "指定用户生效时至少要填一个用户 ID")
    if ids:
        found = await db.query_raw("SELECT id FROM users WHERE id = ANY($1::text[])", ids)
        missing = sorted(set(ids) - {r["id"] for r in found})
        if missing:
            raise HTTPException(400, "找不到这些用户: " + ", ".join(missing[:5]))
    new = {"mode": payload.mode, "scope": payload.scope, "user_ids": ids}
    async with db.tx() as tx:
        rows = await tx.query_raw("SELECT realname_policy FROM system_config WHERE id=1 FOR UPDATE")
        if not rows:
            raise HTTPException(503, "配置尚未初始化")
        before = _stored(rows[0].get("realname_policy"))
        if payload.expected is not None and before != _stored(payload.expected) and not (
                before is None and payload.expected == {}):
            raise HTTPException(409, "配置已被修改，请刷新后重试")
        if before != new:
            await tx.execute_raw("UPDATE system_config SET realname_policy=$1::jsonb, updated_at=CURRENT_TIMESTAMP "
                                 "WHERE id=1", json.dumps(new))
            await tx.execute_raw(
                "INSERT INTO realname_policy_changes (id, admin_id, changes) VALUES ($1,$2,$3::jsonb)",
                str(uuid.uuid4()), str(admin.get("sub") or "") or None,
                json.dumps({"before": before, "after": new}))
            changed = True
        else:
            changed = False
    if changed:
        await load_caches()
    return await _payload()
