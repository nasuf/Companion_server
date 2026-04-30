"""Admin API: 模型库 (admin "系统设置 → 模型库" 维护可选模型 + 单价).

Endpoints:
  GET    /admin-api/model-registry            列表 (含 disabled, admin 视图)
  POST   /admin-api/model-registry            新增 (identifier 唯一)
  PATCH  /admin-api/model-registry/{id}       更新 (identifier 不可改)

DELETE 暂不开放: 已被 system_config / agent_config_overrides 引用的模型 hard
delete 会让现存配置指 "幽灵 ID". admin 改 enabled=false 即可从 dropdown 隐藏.

任何写操作后 invalidate_caches + reload — runtime_config 的 _PRICING_CACHE
立即看到新价格, runtime-config/options 立即看到新模型.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.runtime_config import invalidate_caches, load_caches

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/model-registry",
    tags=["admin", "model-registry"],
    dependencies=[Depends(require_admin_jwt)],
)


_PROVIDERS = {"ollama", "dashscope", "claude"}


class ModelCreatePayload(BaseModel):
    identifier: str = Field(min_length=1, max_length=64)
    display_name: str | None = None
    provider: str
    enabled: bool = True
    context_window: int | None = Field(default=None, ge=1)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    notes: str | None = None


class ModelUpdatePayload(BaseModel):
    """identifier 不在更新范围 (创建后只读). 其他字段全部 optional."""
    display_name: str | None = None
    provider: str | None = None
    enabled: bool | None = None
    context_window: int | None = Field(default=None, ge=1)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    notes: str | None = None


def _row_to_dict(row) -> dict[str, Any]:
    return {
        "id": row.id,
        "identifier": row.identifier,
        "display_name": row.displayName,
        "provider": row.provider,
        "enabled": row.enabled,
        "context_window": row.contextWindow,
        "input_cost_per_million": row.inputCostPerMillion,
        "output_cost_per_million": row.outputCostPerMillion,
        "notes": row.notes,
        "created_at": row.createdAt.isoformat() if row.createdAt else None,
        "updated_at": row.updatedAt.isoformat() if row.updatedAt else None,
    }


def _validate_provider(provider: str) -> None:
    if provider not in _PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider 必须是 {sorted(_PROVIDERS)} 之一, 收到 {provider!r}",
        )


@router.get("")
async def list_models() -> dict[str, list[dict[str, Any]]]:
    """admin 视图: 含 disabled 全部模型, 按 provider + identifier 排序."""
    rows = await db.modelregistry.find_many(
        order=[{"provider": "asc"}, {"identifier": "asc"}],
    )
    return {"models": [_row_to_dict(r) for r in rows]}


@router.post("")
async def create_model(payload: ModelCreatePayload) -> dict[str, Any]:
    _validate_provider(payload.provider)
    try:
        row = await db.modelregistry.create(data={
            "identifier": payload.identifier,
            "displayName": payload.display_name,
            "provider": payload.provider,
            "enabled": payload.enabled,
            "contextWindow": payload.context_window,
            "inputCostPerMillion": payload.input_cost_per_million,
            "outputCostPerMillion": payload.output_cost_per_million,
            "notes": payload.notes,
        })
    except UniqueViolationError:
        raise HTTPException(
            status_code=409,
            detail=f"identifier {payload.identifier!r} 已存在",
        )
    await load_caches()
    invalidate_caches()
    logger.info(f"[MODEL-REGISTRY] created: {payload.identifier} ({payload.provider})")
    return _row_to_dict(row)


@router.patch("/{model_id}")
async def update_model(model_id: str, payload: ModelUpdatePayload) -> dict[str, Any]:
    if payload.provider is not None:
        _validate_provider(payload.provider)

    data: dict[str, Any] = {}
    # 严格按 set 语义: 字段不在 payload 里 = 不动 DB. payload 里 = None / 值都覆盖.
    # pydantic v2 model_fields_set 给 "客户端实际传了哪些 key", 区别于 default None.
    explicit = payload.model_fields_set
    if "display_name" in explicit:
        data["displayName"] = payload.display_name
    if "provider" in explicit:
        data["provider"] = payload.provider
    if "enabled" in explicit:
        data["enabled"] = payload.enabled
    if "context_window" in explicit:
        data["contextWindow"] = payload.context_window
    if "input_cost_per_million" in explicit:
        data["inputCostPerMillion"] = payload.input_cost_per_million
    if "output_cost_per_million" in explicit:
        data["outputCostPerMillion"] = payload.output_cost_per_million
    if "notes" in explicit:
        data["notes"] = payload.notes

    if not data:
        raise HTTPException(status_code=400, detail="payload 不含任何可更新字段")

    row = await db.modelregistry.find_unique(where={"id": model_id})
    if not row:
        raise HTTPException(status_code=404, detail="model not found")

    updated = await db.modelregistry.update(where={"id": model_id}, data=data)
    if updated is None:
        raise HTTPException(status_code=404, detail="model not found")
    await load_caches()
    invalidate_caches()
    logger.info(f"[MODEL-REGISTRY] updated: {row.identifier} fields={list(data.keys())}")
    return _row_to_dict(updated)
