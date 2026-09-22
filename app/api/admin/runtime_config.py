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
_MEDIA_MODEL_PROVIDERS = {
    "vision": {
        "ark": ("ark_api_key", "ARK_API_KEY"),
    },
    "asr": {
        "dashscope": ("dashscope_api_key", "DASHSCOPE_API_KEY"),
    },
    "tts": {
        "dashscope": ("dashscope_tts_api_key", "DASHSCOPE_TTS_API_KEY"),
    },
}


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
    # Global-only fields ignored on per-agent endpoints.
    vision_model: str | None = None
    asr_model: str | None = None
    tts_model: str | None = None
    tts_output_probability: int | None = Field(default=None, ge=0, le=100)
    # Main-reply web search (Ark Responses API web_search tool, ark provider only).
    web_search_enabled: bool | None = None
    # Proactive trending / hot-news (global-only).
    proactive_trending_enabled: bool | None = None
    proactive_trending_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    proactive_trending_link_probability: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )
    proactive_trending_cache_ttl_s: int | None = Field(default=None, ge=60, le=86400)
    # Chat management (global-only).
    reply_delay_enabled: bool | None = None
    reply_delay_max_seconds: int | None = Field(default=None, ge=1, le=3600)
    user_message_aggregation_enabled: bool | None = None


def _row_to_payload(row) -> dict[str, Any]:
    if row is None:
        return {k: None for k in (
            "online_model", "remote_provider", "remote_chat_provider",
            "remote_small_provider", "local_chat_model", "local_small_model",
            "remote_chat_model", "remote_small_model",
            "vision_model", "asr_model", "tts_model",
            "tts_output_probability", "web_search_enabled",
            "proactive_trending_enabled", "proactive_trending_probability",
            "proactive_trending_link_probability", "proactive_trending_cache_ttl_s",
            "reply_delay_enabled", "reply_delay_max_seconds",
            "user_message_aggregation_enabled",
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
        "proactive_trending_enabled": getattr(row, "proactiveTrendingEnabled", None),
        "proactive_trending_probability": getattr(row, "proactiveTrendingProbability", None),
        "proactive_trending_link_probability": getattr(
            row, "proactiveTrendingLinkProbability", None,
        ),
        "proactive_trending_cache_ttl_s": getattr(
            row, "proactiveTrendingCacheTtlS", None,
        ),
        "reply_delay_enabled": getattr(row, "replyDelayEnabled", None),
        "reply_delay_max_seconds": getattr(row, "replyDelayMaxSeconds", None),
        "user_message_aggregation_enabled": getattr(
            row, "userMessageAggregationEnabled", None,
        ),
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
        "proactive_trending_enabled": r.proactive_trending_enabled,
        "proactive_trending_probability": r.proactive_trending_probability,
        "proactive_trending_link_probability": r.proactive_trending_link_probability,
        "proactive_trending_cache_ttl_s": r.proactive_trending_cache_ttl_s,
        "reply_delay_enabled": r.reply_delay_enabled,
        "reply_delay_max_seconds": r.reply_delay_max_seconds,
        "user_message_aggregation_enabled": r.user_message_aggregation_enabled,
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


def _payload_to_update_data(
    payload: ConfigPayload, *, include_global_only: bool = False,
) -> dict[str, Any]:
    """Map only client-sent fields to prisma columns (partial PUT semantics).

    Unset fields are omitted so chat-management patches cannot wipe model config
    that another admin tab saved concurrently. Explicit null still clears override.
    """
    explicit = payload.model_fields_set
    data: dict[str, Any] = {}

    if "online_model" in explicit:
        data["onlineModel"] = payload.online_model

    legacy: str | None = None
    if "remote_provider" in explicit:
        legacy = (
            payload.remote_provider.strip().lower()
            if payload.remote_provider else None
        )
        data["remoteProvider"] = legacy

    chat_provider: str | None = None
    if "remote_chat_provider" in explicit:
        chat_provider = (
            payload.remote_chat_provider.strip().lower()
            if payload.remote_chat_provider else None
        )
        data["remoteChatProvider"] = chat_provider
    elif legacy:
        data["remoteChatProvider"] = legacy

    if "remote_small_provider" in explicit:
        data["remoteSmallProvider"] = (
            payload.remote_small_provider.strip().lower()
            if payload.remote_small_provider else None
        )
    elif legacy:
        data["remoteSmallProvider"] = legacy

    if "local_chat_model" in explicit:
        data["localChatModel"] = payload.local_chat_model
    if "local_small_model" in explicit:
        data["localSmallModel"] = payload.local_small_model
    if "remote_chat_model" in explicit:
        data["remoteChatModel"] = payload.remote_chat_model
    if "remote_small_model" in explicit:
        data["remoteSmallModel"] = payload.remote_small_model

    if include_global_only:
        if "vision_model" in explicit:
            data["visionModel"] = (payload.vision_model or "").strip() or None
        if "asr_model" in explicit:
            data["asrModel"] = (payload.asr_model or "").strip() or None
        if "tts_model" in explicit:
            data["ttsModel"] = (payload.tts_model or "").strip() or None
        if "tts_output_probability" in explicit:
            data["ttsOutputProbability"] = payload.tts_output_probability
        if "web_search_enabled" in explicit:
            data["webSearchEnabled"] = payload.web_search_enabled
        if "proactive_trending_enabled" in explicit:
            data["proactiveTrendingEnabled"] = payload.proactive_trending_enabled
        if "proactive_trending_probability" in explicit:
            data["proactiveTrendingProbability"] = payload.proactive_trending_probability
        if "proactive_trending_link_probability" in explicit:
            data["proactiveTrendingLinkProbability"] = (
                payload.proactive_trending_link_probability
            )
        if "proactive_trending_cache_ttl_s" in explicit:
            data["proactiveTrendingCacheTtlS"] = payload.proactive_trending_cache_ttl_s
        if "reply_delay_enabled" in explicit:
            data["replyDelayEnabled"] = payload.reply_delay_enabled
        if "reply_delay_max_seconds" in explicit:
            data["replyDelayMaxSeconds"] = payload.reply_delay_max_seconds
        if "user_message_aggregation_enabled" in explicit:
            data["userMessageAggregationEnabled"] = (
                payload.user_message_aggregation_enabled
            )
    return data


def _payload_to_data(
    payload: ConfigPayload, *, include_global_only: bool = False,
) -> dict[str, Any]:
    """Full-document mapping (tests / legacy callers). Prefer _payload_to_update_data."""
    return _payload_to_update_data(payload, include_global_only=include_global_only)


async def _model_exists_for_provider(identifier: str, provider: str) -> bool:
    row = await db.modelregistry.find_first(
        where={
            "identifier": identifier,
            "provider": provider,
            "modelKind": "llm",
            "enabled": True,
        },
    )
    return row is not None


async def _media_model_exists(identifier: str, model_kind: str) -> bool:
    providers = set(_MEDIA_MODEL_PROVIDERS.get(model_kind, {}))
    if not providers:
        return False
    row = await db.modelregistry.find_first(
        where={
            "identifier": identifier,
            "provider": {"in": sorted(providers)},
            "modelKind": model_kind,
            "enabled": True,
        },
    )
    return row is not None


def _media_provider_options(model_kind: str) -> list[dict[str, Any]]:
    providers = {
        option["id"]: option
        for option in public_provider_options(include_local=False)
    }
    result: list[dict[str, Any]] = []
    for provider_id, (credential_setting, credential_env) in (
        _MEDIA_MODEL_PROVIDERS.get(model_kind, {}).items()
    ):
        option = providers.get(provider_id)
        if option is None:
            continue
        media_option = dict(option)
        media_option["configured"] = bool(
            str(getattr(settings, credential_setting, "")).strip()
        )
        media_option["credential_env"] = credential_env
        result.append(media_option)
    return result


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

    checks: list[tuple[str | None, str, str]] = []
    if "local_chat_model" in explicit:
        checks.append((
            payload.local_chat_model,
            "ollama",
            "local_chat_model",
        ))
    if "local_small_model" in explicit:
        checks.append((
            payload.local_small_model,
            "ollama",
            "local_small_model",
        ))
    if explicit & {
        "remote_provider",
        "remote_chat_provider",
        "remote_chat_model",
    }:
        checks.append((
            payload.remote_chat_model or fallback_remote_chat_model,
            chat_provider,
            "remote_chat_model",
        ))
    if explicit & {
        "remote_provider",
        "remote_small_provider",
        "remote_small_model",
    }:
        checks.append((
            payload.remote_small_model or fallback_remote_small_model,
            small_provider,
            "remote_small_model",
        ))
    for identifier, expected_provider, field in checks:
        if not identifier:
            continue
        if not await _model_exists_for_provider(identifier, expected_provider):
            raise HTTPException(
                status_code=400,
                detail=f"{field}={identifier!r} 在 provider {expected_provider!r} 下不存在",
            )
    for field, model_kind in (
        ("vision_model", "vision"),
        ("asr_model", "asr"),
        ("tts_model", "tts"),
    ):
        if field not in explicit:
            continue
        value = (getattr(payload, field) or "").strip()
        if not value:
            continue
        if not await _media_model_exists(value, model_kind):
            raise HTTPException(
                status_code=400,
                detail=f"{field}={value!r} 不是已启用的 {model_kind} 模型",
            )


@router.get("/options")
async def list_options() -> dict[str, Any]:
    """前端 dropdown 用. 来源 model_registry (admin "系统设置 → 模型库" 维护).

    按 provider 元数据动态分桶为 local_* / remote_*.
    chat/small 不分角色, 同 provider 模型在两个 dropdown 都出现 (admin 自由选).
    vision/asr/tts are grouped independently by model kind and provider.
    禁用模型 (enabled=false) 不出现.
    """
    rows = await db.modelregistry.find_many(
        where={"enabled": True}, order=[{"identifier": "asc"}],
    )
    by_provider: dict[str, list[str]] = {p: [] for p in sorted(_LOCAL_PROVIDERS | _REMOTE_PROVIDERS)}
    media_by_kind: dict[str, dict[str, list[str]]] = {
        model_kind: {provider: [] for provider in providers}
        for model_kind, providers in _MEDIA_MODEL_PROVIDERS.items()
    }
    for r in rows:
        model_kind = getattr(r, "modelKind", "llm")
        if model_kind in media_by_kind:
            provider_models = media_by_kind[model_kind]
            if r.provider in provider_models:
                provider_models[r.provider].append(r.identifier)
            continue
        if model_kind != "llm":
            continue
        by_provider.setdefault(r.provider, []).append(r.identifier)
    local = [identifier for p in _LOCAL_PROVIDERS for identifier in by_provider.get(p, [])]
    remote = [identifier for p in _REMOTE_PROVIDERS for identifier in by_provider.get(p, [])]
    media = {
        model_kind: {
            "by_provider": by_media_provider,
            "providers": _media_provider_options(model_kind),
        }
        for model_kind, by_media_provider in media_by_kind.items()
    }
    return {
        "local_chat": local,
        "local_small": local,
        "remote_chat": remote,
        "remote_small": remote,
        "tts": [
            identifier
            for identifiers in media_by_kind["tts"].values()
            for identifier in identifiers
        ],
        "by_provider": by_provider,
        "providers": public_provider_options(),
        "media": media,
    }


@router.get("")
async def get_system_config() -> dict[str, Any]:
    """全局 SystemConfig + 当前生效解析值 (null 字段已 fallback 到 env)."""
    await ensure_loaded()
    row = await db.systemconfig.find_unique(where={"id": 1})
    return {
        "config": _row_to_payload(row),
        "resolved": _resolved_to_dict(resolve_config_sync(agent_id=None)),
        "env_gates": {
            "proactive_link_recommendation_enabled": (
                settings.proactive_link_recommendation_enabled
            ),
        },
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
    data = _payload_to_update_data(payload, include_global_only=True)
    if not data:
        raise HTTPException(status_code=400, detail="至少需要提供一个配置字段")
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
    data = _payload_to_update_data(payload)
    if not data:
        raise HTTPException(status_code=400, detail="至少需要提供一个配置字段")
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
