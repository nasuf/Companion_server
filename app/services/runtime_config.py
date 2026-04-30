"""Runtime config — 模型选择的动态配置 (admin "系统设置" + per-agent override).

设计:
  解析顺序: AgentConfigOverride.field → SystemConfig.field → settings.field (env)
  字段范围: ONLINE_MODEL / LOCAL_CHAT / LOCAL_SMALL / REMOTE_CHAT / REMOTE_SMALL
  embedding 不在此 — 跨 agent 共享 vector store, 改了已有向量失真.

热路径 (get_chat_model 等) 必须 sync 拿配置, 不能 await DB. 启动时把 system 行
+ 所有 agent overrides 一次性 load 到模块级 in-memory cache; 任何 PUT 后调
invalidate_caches() 重新 load + 清 lru_cache. ContextVar 携带当前 agent_id,
模型工厂据此挑 override.

ContextVar 在 orchestrator / proactive sender / memory pipeline 等请求入口设置
(set_current_agent), 没设时 fallback 到 None → 仅取 system / env, 不应用 override.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedConfig:
    """resolve_config 的返回. 每个字段都已穿透 override → system → env 链路."""
    online_model: bool
    local_chat_model: str
    local_small_model: str
    remote_chat_model: str
    remote_small_model: str


# Module-level caches, 启动时填充, 配置变更时 invalidate 重 load.
# 单进程内 dict 即可; 多 worker 时各 worker 自己 load (PUT API 也只更新本进程,
# 跨 worker 同步靠 Redis pub/sub, 暂不实现 — 单 worker 部署足够).
_GLOBAL_CACHE: dict | None = None
_AGENT_CACHE: dict[str, dict] = {}
# {identifier: {"input": float, "output": float}} — pricing.estimate_cost_cny 同步读取
_PRICING_CACHE: dict[str, dict[str, float]] = {}
_CACHE_LOADED = False
# 防并发 ensure_loaded 触发 N 次 load_caches (每次都打 DB)
_LOAD_LOCK = asyncio.Lock()

# 当前请求绑定的 agent_id, orchestrator 等入口 set, 模型工厂 read.
_current_agent: ContextVar[str | None] = ContextVar("_current_agent", default=None)


def set_current_agent(agent_id: str | None):
    """在请求入口处调用, 让后续 get_chat_model() 应用该 agent 的 override.

    返回 ContextVar Token, 调用方建议在 finally 里 reset_current_agent(token)
    防异常 leak (尤其共享 worker pool 场景). 普通 fire_background 通过
    copy_context 自己拿快照, 不受外层 reset 影响.
    """
    return _current_agent.set(agent_id)


def reset_current_agent(token) -> None:
    """配合 set_current_agent 的 try/finally 还原 ContextVar.

    token=None 时 no-op (调用方在 set 之前抛异常的 unbound 兜底).
    """
    if token is None:
        return
    try:
        _current_agent.reset(token)
    except (LookupError, ValueError):
        # 跨 task reset 抛 ValueError, 跨 ctx 抛 LookupError, 静默兜底.
        pass


def get_current_agent() -> str | None:
    return _current_agent.get()


async def bind_agent_context(agent_id: str | None):
    """请求入口的便捷封装: ensure_loaded + set_current_agent.

    返回 Token. 大多数 fire-and-forget 入口 (proactive/cron) 可忽略;
    长流式入口 (orchestrator) try/finally + reset_current_agent(token).
    """
    await ensure_loaded()
    return _current_agent.set(agent_id)


async def load_caches() -> None:
    """启动时调用. PUT API 改完后也调用 (invalidate + reload).

    先把 DB 行装进新 dict 再原子赋值给 module-level — 期间 sync 读者继续读
    旧值 (一致), 不会读到半填充状态. 同步路径不 await ensure_loaded, 这是
    避免 stale-read 的关键 (旧实现先 _CACHE_LOADED=False 再 reload, 中间窗口
    sync 读者拿到未清空的旧 dict, 与新 _CACHE_LOADED 状态语义割裂).

    DB 是模型配置真源, env 仅在 DB 整体不可用时兜底. 首次启动 / 现有 deploy
    没跑 seed migration 时 system_config 行可能缺 → 用当前 env 默认 auto-seed.
    """
    global _GLOBAL_CACHE, _AGENT_CACHE, _PRICING_CACHE, _CACHE_LOADED
    new_pricing: dict[str, dict[str, float]] = {}
    try:
        sys_row = await db.systemconfig.find_unique(where={"id": 1})
        if sys_row is None:
            sys_row = await _seed_system_config_with_env_defaults()
        new_global = _row_to_dict(sys_row)
        overrides = await db.agentconfigoverride.find_many()
        new_agent = {row.agentId: _row_to_dict(row) for row in overrides}
        # 装载 model_registry 的价格 (只 enabled, 因为 disabled 模型不会被新调用,
        # 但已有 llm_usage 行的归桶若指向它仍能匹配 → 保险起见 disabled 也装).
        registry = await db.modelregistry.find_many()
        for r in registry:
            new_pricing[r.identifier] = {
                "input": r.inputCostPerMillion or 0.0,
                "output": r.outputCostPerMillion or 0.0,
            }
    except Exception as e:
        # DB 不可用 (e.g. 测试无连接) → 缓存空, 全部 fallback 到 env / 价格当 0.
        logger.warning(f"[RUNTIME-CONFIG] load failed, falling back to env only: {e}")
        new_global = {}
        new_agent = {}
    _GLOBAL_CACHE = new_global
    _AGENT_CACHE = new_agent
    _PRICING_CACHE = new_pricing
    _CACHE_LOADED = True
    logger.info(
        f"[RUNTIME-CONFIG] loaded: global={bool(_GLOBAL_CACHE)} "
        f"overrides={len(_AGENT_CACHE)} pricing={len(_PRICING_CACHE)}"
    )


async def _seed_system_config_with_env_defaults():
    """system_config 行不存在时用 env 默认 seed 一行. 仅首次. 后续改靠 admin PUT."""
    logger.info("[RUNTIME-CONFIG] system_config row missing, seeding with env defaults")
    return await db.systemconfig.create(data={
        "id": 1,
        "onlineModel": settings.online_model,
        "localChatModel": settings.local_chat_model,
        "localSmallModel": settings.local_small_model,
        "remoteChatModel": settings.remote_chat_model,
        "remoteSmallModel": settings.remote_small_model,
    })


def _row_to_dict(row) -> dict:
    """SystemConfig / AgentConfigOverride row → 仅含非 None 字段的 dict."""
    out: dict = {}
    for key in ("onlineModel", "localChatModel", "localSmallModel",
                "remoteChatModel", "remoteSmallModel"):
        val = getattr(row, key, None)
        if val is not None:
            out[key] = val
    return out


def invalidate_caches() -> None:
    """清模型工厂 lru_cache. 调用方应先 await load_caches() 把 module cache
    更新到最新, 再调本函数 evict 旧模型实例 — 顺序保证下次 build 用新配置.
    """
    # 循环 import 防御: 延迟导入
    try:
        from app.services.llm.models import (
            get_chat_model, get_utility_model, get_embedding_model,
            get_fallback_chat_model,
        )
        get_chat_model.cache_clear()
        get_utility_model.cache_clear()
        get_embedding_model.cache_clear()
        get_fallback_chat_model.cache_clear()
    except Exception as e:
        logger.warning(f"[RUNTIME-CONFIG] cache_clear failed: {e}")


async def ensure_loaded() -> None:
    """resolve_config_sync 调用前确保 caches 已 load. 启动 lifespan 也可主动调.

    Lock 防并发: 多个请求同时穿透 _CACHE_LOADED=False 时只 load 一次.
    """
    if _CACHE_LOADED:
        return
    async with _LOAD_LOCK:
        if not _CACHE_LOADED:  # double-check
            await load_caches()


def resolve_config_sync(agent_id: str | None = None) -> ResolvedConfig:
    """同步取配置, 解析链 agent override → system (DB 真源) → env (兜底).

    正常情况 system_config 行总是存在 (migration seed + load_caches auto-seed),
    env fallback 仅在 DB 整体 load 失败时触发 (load_caches except 把 _GLOBAL_CACHE
    置 {}). 即: 跑得动 DB 就用 DB, 跑不动才用 env.

    agent_id 为 None 时跳过 override (用于无会话上下文场景: admin 建 agent /
    cron / global memory pipeline). 显式传 agent_id 时按该 agent 的 override 取.
    """
    ag_override = _AGENT_CACHE.get(agent_id) if agent_id else None
    glob = _GLOBAL_CACHE or {}

    def _pick(field: str, env_default):
        if ag_override and field in ag_override:
            return ag_override[field]
        if field in glob:
            return glob[field]
        return env_default

    return ResolvedConfig(
        online_model=_pick("onlineModel", settings.online_model),
        local_chat_model=_pick("localChatModel", settings.local_chat_model),
        local_small_model=_pick("localSmallModel", settings.local_small_model),
        remote_chat_model=_pick("remoteChatModel", settings.remote_chat_model),
        remote_small_model=_pick("remoteSmallModel", settings.remote_small_model),
    )


def resolve_for_current() -> ResolvedConfig:
    """convenience: 用 ContextVar 当前 agent 解析."""
    return resolve_config_sync(get_current_agent())


def get_pricing(model: str) -> dict[str, float] | None:
    """同步取该 model 的 (input, output) 单价 (元/1M tokens). 未知返 None.

    pricing.estimate_cost_cny 调本函数. _PRICING_CACHE 由 load_caches 装载;
    admin PUT model_registry 后 invalidate_caches → 下次 ensure_loaded 重 load.
    """
    return _PRICING_CACHE.get(model)
