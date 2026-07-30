"""Admin API: 运行时模型配置 (admin "系统设置" + per-agent override).

Endpoints:
  GET    /admin-api/runtime-config            — 取全局 SystemConfig (缺省字段返 null = 走 env)
  PUT    /admin-api/runtime-config            — 更新全局 SystemConfig + invalidate caches
  GET    /admin-api/runtime-config/options    — 列出可选模型枚举 (前端 dropdown 用)
  GET    /admin-api/runtime-config/agents/{agent_id}     — 取该 agent override
  PUT    /admin-api/runtime-config/agents/{agent_id}     — 更新该 agent override + invalidate
  DELETE /admin-api/runtime-config/agents/{agent_id}     — 删除 override (回归全局)

字段范围: online_model / remote_chat_provider / remote_small_provider /
local_chat_model / local_small_model / remote_chat_model / remote_small_model
+ 多模态 vision_model / asr_model (仅全局, per-agent endpoints 忽略).
remote_provider 是旧客户端兼容字段. 全部 nullable (null = 不设, fallback 上层).
embedding 不在此 — 跨 agent 共享 vector store 不能动态切.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import RecordNotFoundError
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.config import settings
from app.db import db
from app.redis_client import get_redis
from app.services.llm.providers import provider_ids, public_provider_options
from app.services.memory.config import CALIBRATED_EMBEDDING_MODEL
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


_LOCAL_PROVIDERS = provider_ids(admin_only=True) - provider_ids(
    admin_only=True, remote_only=True,
)
_REMOTE_PROVIDERS = provider_ids(admin_only=True, remote_only=True)


async def _sync_tts_probability(probability: int) -> None:
    """Publish the global percentage for all uvicorn workers immediately."""
    try:
        redis = await get_redis()
        await redis.set("runtime:tts_output_probability", int(probability))
    except Exception as exc:
        logger.warning("TTS probability Redis sync failed: %s", exc)


class ConfigPayload(BaseModel):
    """所有字段 None = 不设/清除. PUT 接受这个用作 set/unset 单字段."""
    online_model: bool | None = None
    # Deprecated shared field. If a legacy client only sends this field, the
    # server mirrors it to both role-specific provider columns.
    remote_provider: str | None = None
    remote_chat_provider: str | None = None
    remote_small_provider: str | None = None
    local_chat_model: str | None = None
    local_small_model: str | None = None
    remote_chat_model: str | None = None
    remote_small_model: str | None = None
    # Global-only fields — ignored on the per-agent endpoints
    # (AgentConfigOverride has no such columns). vision/asr are free-text
    # identifiers (not part of model_registry, so no registry check).
    vision_model: str | None = None
    asr_model: str | None = None
    tts_model: str | None = None
    tts_output_probability: int | None = Field(default=None, ge=0, le=100)
    # Main-reply web search (Ark Responses API web_search tool, ark provider only).
    web_search_enabled: bool | None = None


def _row_to_payload(row) -> dict[str, Any]:
    if row is None:
        return {k: None for k in (
            "online_model", "remote_provider", "remote_chat_provider",
            "remote_small_provider", "local_chat_model", "local_small_model",
            "remote_chat_model", "remote_small_model",
            "vision_model", "asr_model", "tts_model",
            "tts_output_probability", "web_search_enabled",
        )}
    return {
        "online_model": row.onlineModel,
        "remote_provider": row.remoteProvider,
        "remote_chat_provider": row.remoteChatProvider,
        "remote_small_provider": row.remoteSmallProvider,
        "local_chat_model": row.localChatModel,
        "local_small_model": row.localSmallModel,
        "remote_chat_model": row.remoteChatModel,
        "remote_small_model": row.remoteSmallModel,
        # getattr: AgentConfigOverride rows share this helper but lack these
        # global-only columns → always null on the agent endpoints.
        "vision_model": getattr(row, "visionModel", None),
        "asr_model": getattr(row, "asrModel", None),
        "tts_model": getattr(row, "ttsModel", None),
        "tts_output_probability": getattr(row, "ttsOutputProbability", None),
        "web_search_enabled": getattr(row, "webSearchEnabled", None),
    }


def _resolved_to_dict(r: ResolvedConfig) -> dict[str, Any]:
    """ResolvedConfig → JSON dict (4 endpoints 共用)."""
    return {
        "online_model": r.online_model,
        "remote_provider": r.remote_provider,
        "remote_chat_provider": r.remote_chat_provider,
        "remote_small_provider": r.remote_small_provider,
        "local_chat_model": r.local_chat_model,
        "local_small_model": r.local_small_model,
        "remote_chat_model": r.remote_chat_model,
        "remote_small_model": r.remote_small_model,
        "vision_model": r.vision_model,
        "asr_model": r.asr_model,
        "tts_model": r.tts_model,
        "tts_output_probability": r.tts_output_probability,
        "web_search_enabled": r.web_search_enabled,
        # 只读. Embedding 模型不是运行时开关: 库里 8000+ 条向量就是当前模型的
        # 输出, 换掉而不重算等于让查询在陌生坐标系里检索 (同一段文本跨模型的
        # 余弦实测 -0.001, 比同模型内两段无关文本的 0.43 还低), 而且十一个相似
        # 度阈值是按该模型的分布标定的。改它是一次数据迁移, 不是改配置, 所以
        # 后台只展示不提供输入框 —— 流程见 scripts/reembed_memories.py。
        "embedding_model": settings.embedding_model,
        "embedding_model_editable": False,
        "embedding_model_calibrated": (
            settings.embedding_model == CALIBRATED_EMBEDDING_MODEL
        ),
    }


def _payload_to_data(
    payload: ConfigPayload, *, include_global_only: bool = False,
) -> dict[str, Any]:
    """payload → prisma 字段 dict. None 值保留 (清除该字段 override).

    include_global_only 仅全局 SystemConfig 为 True — AgentConfigOverride
    表没有 vision/asr/webSearch 列, 写入会直接报 prisma unknown column.
    """
    explicit = payload.model_fields_set
    legacy = payload.remote_provider.strip().lower() if payload.remote_provider else None
    chat_provider = (
        payload.remote_chat_provider.strip().lower()
        if payload.remote_chat_provider else None
    )
    small_provider = (
        payload.remote_small_provider.strip().lower()
        if payload.remote_small_provider else None
    )
    # Backward compatibility for clients that predate role-specific providers.
    if "remote_chat_provider" not in explicit and legacy:
        chat_provider = legacy
    if "remote_small_provider" not in explicit and legacy:
        small_provider = legacy
    data: dict[str, Any] = {
        "onlineModel": payload.online_model,
        "remoteProvider": legacy,
        "remoteChatProvider": chat_provider,
        "remoteSmallProvider": small_provider,
        "localChatModel": payload.local_chat_model,
        "localSmallModel": payload.local_small_model,
        "remoteChatModel": payload.remote_chat_model,
        "remoteSmallModel": payload.remote_small_model,
    }
    if include_global_only:
        # Empty string means "clear override" (fall back to env), same as null.
        data["visionModel"] = (payload.vision_model or "").strip() or None
        data["asrModel"] = (payload.asr_model or "").strip() or None
        data["ttsModel"] = (payload.tts_model or "").strip() or None
        data["ttsOutputProbability"] = payload.tts_output_probability
        data["webSearchEnabled"] = payload.web_search_enabled
    return data


async def _model_exists_for_provider(identifier: str, provider: str) -> bool:
    row = await db.modelregistry.find_first(
        where={"identifier": identifier, "provider": provider},
    )
    return row is not None


async def _tts_model_exists(identifier: str) -> bool:
    row = await db.modelregistry.find_first(
        where={
            "identifier": identifier,
            "provider": "dashscope",
            "modelKind": "tts",
            "enabled": True,
        },
    )
    return row is not None


def _normalize_remote_provider(value: str | None, fallback: str) -> str:
    provider = (value or fallback or "dashscope").strip().lower()
    if provider not in _REMOTE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"remote_provider 必须是 {sorted(_REMOTE_PROVIDERS)} 之一, 收到 {provider!r}",
        )
    return provider


async def _validate_payload_models(
    payload: ConfigPayload,
    *,
    fallback_remote_chat_provider: str,
    fallback_remote_small_provider: str,
    fallback_remote_chat_model: str,
    fallback_remote_small_model: str,
) -> None:
    legacy = payload.remote_provider
    explicit = payload.model_fields_set
    chat_value = (
        payload.remote_chat_provider
        if "remote_chat_provider" in explicit
        else legacy
    )
    small_value = (
        payload.remote_small_provider
        if "remote_small_provider" in explicit
        else legacy
    )
    chat_provider = _normalize_remote_provider(
        chat_value,
        fallback_remote_chat_provider,
    )
    small_provider = _normalize_remote_provider(
        small_value,
        fallback_remote_small_provider,
    )

    checks = [
        (payload.local_chat_model, "ollama", "local_chat_model"),
        (payload.local_small_model, "ollama", "local_small_model"),
        (
            payload.remote_chat_model or fallback_remote_chat_model,
            chat_provider,
            "remote_chat_model",
        ),
        (
            payload.remote_small_model or fallback_remote_small_model,
            small_provider,
            "remote_small_model",
        ),
    ]
    for identifier, expected_provider, field in checks:
        if not identifier:
            continue
        if not await _model_exists_for_provider(identifier, expected_provider):
            raise HTTPException(
                status_code=400,
                detail=f"{field}={identifier!r} 在 provider {expected_provider!r} 下不存在",
            )
    if "tts_model" in explicit and payload.tts_model:
        if not await _tts_model_exists(payload.tts_model):
            raise HTTPException(
                status_code=400,
                detail=f"tts_model={payload.tts_model!r} 不是已启用的 DashScope TTS 模型",
            )


@router.get("/options")
async def list_options() -> dict[str, Any]:
    """前端 dropdown 用. 来源 model_registry (admin "系统设置 → 模型库" 维护).

    按 provider 元数据动态分桶为 local_* / remote_*.
    chat/small 不分角色, 同 provider 模型在两个 dropdown 都出现 (admin 自由选).
    禁用模型 (enabled=false) 不出现.
    """
    rows = await db.modelregistry.find_many(
        where={"enabled": True}, order=[{"identifier": "asc"}],
    )
    by_provider: dict[str, list[str]] = {p: [] for p in sorted(_LOCAL_PROVIDERS | _REMOTE_PROVIDERS)}
    tts: list[str] = []
    for r in rows:
        if getattr(r, "modelKind", "llm") == "tts":
            tts.append(r.identifier)
            continue
        by_provider.setdefault(r.provider, []).append(r.identifier)
    local = [identifier for p in _LOCAL_PROVIDERS for identifier in by_provider.get(p, [])]
    remote = [identifier for p in _REMOTE_PROVIDERS for identifier in by_provider.get(p, [])]
    return {
        "local_chat": local,
        "local_small": local,
        "remote_chat": remote,
        "remote_small": remote,
        "tts": tts,
        "by_provider": by_provider,
        "providers": public_provider_options(),
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
    await _validate_payload_models(
        payload,
        fallback_remote_chat_provider=(
            settings.remote_chat_provider or settings.remote_provider
        ),
        fallback_remote_small_provider=(
            settings.remote_small_provider or settings.remote_provider
        ),
        fallback_remote_chat_model=settings.remote_chat_model,
        fallback_remote_small_model=settings.remote_small_model,
    )
    data = _payload_to_data(payload, include_global_only=True)
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
    if "tts_output_probability" in payload.model_fields_set:
        await _sync_tts_probability(
            resolve_config_sync(agent_id=None).tts_output_probability,
        )
    logger.info(f"[RUNTIME-CONFIG] system updated: {data}")
    return {
        "config": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=None)),
    }


class TtsProbabilityPayload(BaseModel):
    probability: int = Field(ge=0, le=100)


@router.put("/tts-output-probability")
async def put_tts_output_probability(
    payload: TtsProbabilityPayload,
) -> dict[str, Any]:
    """Atomically update only the global voice-output probability."""
    row = await db.systemconfig.upsert(
        where={"id": 1},
        data={
            "create": {
                "id": 1,
                "ttsOutputProbability": payload.probability,
            },
            "update": {"ttsOutputProbability": payload.probability},
        },
    )
    await load_caches()
    invalidate_caches()
    await _sync_tts_probability(payload.probability)
    logger.info(
        "[RUNTIME-CONFIG] TTS output probability updated: %s",
        payload.probability,
    )
    return {
        "probability": payload.probability,
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
    await ensure_loaded()
    system_config = resolve_config_sync(agent_id=None)
    await _validate_payload_models(
        payload,
        fallback_remote_chat_provider=system_config.remote_chat_provider,
        fallback_remote_small_provider=system_config.remote_small_provider,
        fallback_remote_chat_model=system_config.remote_chat_model,
        fallback_remote_small_model=system_config.remote_small_model,
    )
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
