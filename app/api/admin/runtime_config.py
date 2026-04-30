"""Admin API: 运行时模型配置 (admin "系统设置" + per-agent override).

Endpoints:
  GET    /admin-api/runtime-config            — 取全局 SystemConfig (缺省字段返 null = 走 env)
  PUT    /admin-api/runtime-config            — 更新全局 SystemConfig + invalidate caches
  GET    /admin-api/runtime-config/options    — 列出可选模型枚举 (前端 dropdown 用)
  GET    /admin-api/runtime-config/agents/{agent_id}     — 取该 agent override
  PUT    /admin-api/runtime-config/agents/{agent_id}     — 更新该 agent override + invalidate
  DELETE /admin-api/runtime-config/agents/{agent_id}     — 删除 override (回归全局)

字段范围: online_model / local_chat_model / local_small_model / remote_chat_model /
remote_small_model. 全部 nullable (null = 不设, fallback 上层).
embedding 不在此 — 跨 agent 共享 vector store 不能动态切.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import RecordNotFoundError
from pydantic import BaseModel

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.runtime_config import (
    ResolvedConfig, ensure_loaded, invalidate_caches, load_caches,
    resolve_config_sync,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/runtime-config",
    tags=["admin", "runtime-config"],
    dependencies=[Depends(require_admin_jwt)],
)


_LOCAL_PROVIDERS = {"ollama"}
_REMOTE_PROVIDERS = {"dashscope", "claude"}


class ConfigPayload(BaseModel):
    """所有字段 None = 不设/清除. PUT 接受这个用作 set/unset 单字段."""
    online_model: bool | None = None
    local_chat_model: str | None = None
    local_small_model: str | None = None
    remote_chat_model: str | None = None
    remote_small_model: str | None = None


def _row_to_payload(row) -> dict[str, Any]:
    if row is None:
        return {k: None for k in (
            "online_model", "local_chat_model", "local_small_model",
            "remote_chat_model", "remote_small_model",
        )}
    return {
        "online_model": row.onlineModel,
        "local_chat_model": row.localChatModel,
        "local_small_model": row.localSmallModel,
        "remote_chat_model": row.remoteChatModel,
        "remote_small_model": row.remoteSmallModel,
    }


def _resolved_to_dict(r: ResolvedConfig) -> dict[str, Any]:
    """ResolvedConfig → JSON dict (4 endpoints 共用)."""
    return {
        "online_model": r.online_model,
        "local_chat_model": r.local_chat_model,
        "local_small_model": r.local_small_model,
        "remote_chat_model": r.remote_chat_model,
        "remote_small_model": r.remote_small_model,
    }


def _payload_to_data(payload: ConfigPayload) -> dict[str, Any]:
    """payload → prisma 字段 dict. None 值保留 (清除该字段 override)."""
    return {
        "onlineModel": payload.online_model,
        "localChatModel": payload.local_chat_model,
        "localSmallModel": payload.local_small_model,
        "remoteChatModel": payload.remote_chat_model,
        "remoteSmallModel": payload.remote_small_model,
    }


@router.get("/options")
async def list_options() -> dict[str, list[str]]:
    """前端 dropdown 用. 来源 model_registry (admin "系统设置 → 模型库" 维护).

    按 provider 分桶: ollama → local_*, dashscope/claude → remote_*.
    chat/small 不分角色, 同 provider 模型在两个 dropdown 都出现 (admin 自由选).
    禁用模型 (enabled=false) 不出现.
    """
    rows = await db.modelregistry.find_many(
        where={"enabled": True}, order=[{"identifier": "asc"}],
    )
    local = [r.identifier for r in rows if r.provider in _LOCAL_PROVIDERS]
    remote = [r.identifier for r in rows if r.provider in _REMOTE_PROVIDERS]
    return {
        "local_chat": local,
        "local_small": local,
        "remote_chat": remote,
        "remote_small": remote,
    }


@router.get("")
async def get_system_config() -> dict[str, Any]:
    """全局 SystemConfig + 当前生效解析值 (null 字段已 fallback 到 env)."""
    await ensure_loaded()
    row = await db.systemconfig.find_unique(where={"id": 1})
    return {
        "config": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=None)),
    }


@router.put("")
async def put_system_config(payload: ConfigPayload) -> dict[str, Any]:
    """更新全局 SystemConfig + 重 load 缓存 + 清模型 lru_cache. 立即生效 (in-flight chain 仍旧)."""
    data = _payload_to_data(payload)
    row = await db.systemconfig.upsert(
        where={"id": 1},
        data={"create": {"id": 1, **data}, "update": data},
    )
    # 先 reload caches (原子赋值, 期间 sync 读者读旧值不阻塞), 再清模型 lru_cache.
    # 顺序: DB 写 → reload module-cache → clear lru_cache. 任何时刻读者拿到的
    # 都是有效配置 (旧 cache+旧 lru / 旧 cache+新 lru / 新 cache+新 lru),
    # 不会出现 "新 lru 实例用旧 cache 重 build 立刻又 evict" 抖动.
    await load_caches()
    invalidate_caches()
    logger.info(f"[RUNTIME-CONFIG] system updated: {data}")
    return {
        "config": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=None)),
    }


@router.get("/agents/{agent_id}")
async def get_agent_config(agent_id: str) -> dict[str, Any]:
    """该 agent 的 override + 当前生效解析值 (override → system → env 链路结果)."""
    await ensure_loaded()
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    row = await db.agentconfigoverride.find_unique(where={"agentId": agent_id})
    return {
        "agent_id": agent_id,
        "override": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=agent_id)),
    }


@router.put("/agents/{agent_id}")
async def put_agent_config(agent_id: str, payload: ConfigPayload) -> dict[str, Any]:
    """更新该 agent override. 改完 invalidate 让模型工厂下次 build 时按新 override 取."""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    data = _payload_to_data(payload)
    row = await db.agentconfigoverride.upsert(
        where={"agentId": agent_id},
        data={"create": {"agentId": agent_id, **data}, "update": data},
    )
    await load_caches()
    invalidate_caches()
    logger.info(f"[RUNTIME-CONFIG] agent={agent_id[:8]} override updated: {data}")
    return {
        "agent_id": agent_id,
        "override": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=agent_id)),
    }


@router.delete("/agents/{agent_id}")
async def delete_agent_config(agent_id: str) -> dict[str, str]:
    """删 override → 该 agent 回归 system / env 配置. RecordNotFound 视为 idempotent."""
    try:
        await db.agentconfigoverride.delete(where={"agentId": agent_id})
    except RecordNotFoundError:
        pass
    await load_caches()
    invalidate_caches()
    logger.info(f"[RUNTIME-CONFIG] agent={agent_id[:8]} override cleared")
    return {"status": "ok"}
