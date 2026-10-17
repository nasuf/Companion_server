"""Admin API: 后台「聊天管理 → 主动交流」的可调参数.

Endpoints:
  GET /admin-api/proactive-tuning   — 参数登记表 (分组 / 说明 / 默认 / 范围) + 当前生效值 + 最近修改记录
  PUT /admin-api/proactive-tuning   — 批量修改, body {"values": {key: 新值 | null}}; null = 恢复默认

参数定义见 app/services/proactive/tuning.py; 生效值存 system_config.proactive_policy
(只存与默认不同的项), 数据库触发器递增配置版本号, 各 worker ≤2s 内刷新。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.proactive import tuning
from app.services.runtime_config import load_caches

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/proactive-tuning",
    tags=["admin", "proactive-tuning"],
    dependencies=[Depends(require_admin_jwt)],
)

_RECENT_CHANGES = 30


class TuningPatch(BaseModel):
    values: dict[str, Any]


def _stored(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


async def _load_stored() -> dict[str, Any]:
    rows = await db.query_raw("SELECT proactive_policy FROM system_config WHERE id = 1")
    return _stored(rows[0].get("proactive_policy")) if rows else {}


async def _recent_changes() -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT c.id, c.admin_id, c.changes, c.created_at, u.username
        FROM proactive_tuning_changes c
        LEFT JOIN users u ON u.id = c.admin_id
        ORDER BY c.created_at DESC
        LIMIT $1
        """,
        _RECENT_CHANGES,
    )
    out = []
    for row in rows:
        items = row.get("changes")
        items = json.loads(items) if isinstance(items, str) else items
        out.append({
            "id": row["id"],
            "admin_id": row.get("admin_id"),
            "admin_name": row.get("username"),
            "created_at": _utc_iso(row.get("created_at")),
            "changes": [
                {**item, "label": _label(str(item.get("key")))}
                for item in (items or []) if isinstance(item, dict)
            ],
        })
    return out


def _utc_iso(value: Any) -> str | None:
    """库里是不带时区的 UTC; 输出带 +00:00, 前端按本地时区显示不会差 8 小时."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


def _label(key: str) -> str:
    """修改记录里的可读名称: 配比表成员带上所属表名."""
    param = tuning.PARAM_MAP.get(key)
    if param is None:
        return key  # 登记表里已删掉的旧参数, 记录照样显示
    if param.sum_group:
        group = next(g for g in tuning.SUM_GROUPS if g.key == param.sum_group)
        section = next(g for g in tuning.GROUPS if g.key == group.group)
        return f"{section.title} · {group.title} · {param.label}"
    return param.label


async def _payload(stored: dict[str, Any]) -> dict[str, Any]:
    values = tuning.resolve(stored)
    return {
        "groups": [g.__dict__ for g in tuning.GROUPS],
        "sum_groups": [g.__dict__ for g in tuning.SUM_GROUPS],
        "params": [
            {
                "key": p.key,
                "group": p.group,
                "label": p.label,
                "description": p.description,
                "kind": p.kind,
                "default": p.default,
                "minimum": p.minimum,
                "maximum": p.maximum,
                "sum_group": p.sum_group,
                "value": values[p.key],
                "overridden": values[p.key] != p.default,
            }
            for p in tuning.PARAMS
        ],
        "changes": await _recent_changes(),
    }


@router.get("")
async def get_proactive_tuning() -> dict[str, Any]:
    return await _payload(await _load_stored())


@router.put("")
async def put_proactive_tuning(
    payload: TuningPatch,
    admin: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    if not payload.values:
        raise HTTPException(status_code=400, detail="没有要保存的修改")
    if not await db.query_raw("SELECT 1 FROM system_config WHERE id = 1"):
        from app.services.runtime_config import _seed_system_config_with_env_defaults

        await _seed_system_config_with_env_defaults()

    async with db.tx() as tx:
        # 行锁: 两个管理员同时保存时串行合并, 不会互相覆盖对方刚改的项
        rows = await tx.query_raw("SELECT proactive_policy FROM system_config WHERE id = 1 FOR UPDATE")
        stored = _stored(rows[0].get("proactive_policy")) if rows else {}
        try:
            updated = tuning.apply_patch(stored, payload.values)
        except tuning.TuningError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        before, after = tuning.resolve(stored), tuning.resolve(updated)
        changes = [
            {"key": key, "before": before[key], "after": after[key]}
            for key in payload.values
            if key in after and before[key] != after[key]
        ]
        if changes:
            await tx.execute_raw(
                "UPDATE system_config SET proactive_policy = $1::jsonb, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = 1",
                json.dumps(updated) if updated else None,
            )
            await tx.execute_raw(
                "INSERT INTO proactive_tuning_changes (id, admin_id, changes) VALUES ($1, $2, $3::jsonb)",
                str(uuid.uuid4()), str(admin.get("sub") or "") or None, json.dumps(changes, ensure_ascii=False),
            )

    if changes:
        await load_caches()  # 本 worker 立即生效; 其他 worker 随版本号在 2s 内刷新
        logger.info(
            "[PROACTIVE-TUNING] updated by admin=%s: %s",
            admin.get("sub"), ", ".join(f"{c['key']}={c['after']}" for c in changes),
        )
    return await _payload(updated)
