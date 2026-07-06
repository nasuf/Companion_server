"""主回复系统提示词的段落顺序管理 — 唯一的"顺序汇总点".

背景: 各 section 的**文本**由 registry (defaults.py) 集中管理, 但它们在最终
system prompt 里的**排列顺序**此前只存在于 build_system_prompt 的代码调用
顺序里 (trace 面板拖拽只影响单次重跑). 本模块把顺序收敛为可管理配置:

- `CHAT_SECTION_SLOTS` 定义全部段位 (slot) — 代码默认顺序是唯一真理源;
- admin 可通过 PUT /admin-api/prompts/section-order 覆写顺序 (DB + Redis 持久),
  build_system_prompt 每次装配按 `get_chat_section_order()` 发射;
- 读取三级 fallback: 进程内缓存 (10s) → Redis → DB → 代码默认. 任何一层
  异常都退回默认顺序 — 顺序配置永远不能让聊天挂掉;
- 校验从严 (必须是全量 slot 的排列), 防止提示词结构混乱; 代码日后新增
  slot 时, 旧覆写在读取端自动把缺失 slot 按默认相对顺序补到末尾.

⚠️ 顺序影响 dashscope prefix cache: 默认顺序刻意把稳定段 (核心规则/身份/
一致性) 放前面, 变化段放后面, 命中前缀缓存省 ~60% input 成本. 把变化段
挪到稳定段之前会打穿缓存 — admin UI 侧有提示, 后端不做硬拦截 (调试自由).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from prisma import Json

from app.db import db
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# 目前唯一支持顺序管理的组合 prompt (boundary 组合 prompt / special_instruction
# appendix 有各自的装配点, 不在此管理).
CHAT_SECTION_ORDER_KEY = "chat.system_base"

_REDIS_KEY = f"prompt_section_order:{CHAT_SECTION_ORDER_KEY}"
_LOCAL_TTL_SECONDS = 10.0
_local_cache: tuple[list[str], float] | None = None


@dataclass(frozen=True)
class SectionSlot:
    """一个段位: slot id 稳定不变; prompt_keys 是该段位可能渲染出的模板 key
    (供前端把 trace 组件 span 映射回 slot)."""

    slot: str
    title: str
    prompt_keys: tuple[str, ...]


# 默认顺序 = build_system_prompt 的装配顺序 (cache 友好: 稳定前缀在前).
# ⚠️ 新增 section 时必须同步在这里登记 slot, 否则 build_system_prompt 的
# 装配循环不会发射它 (有守卫测试锁定).
CHAT_SECTION_SLOTS: tuple[SectionSlot, ...] = (
    SectionSlot("core_rules", "核心规则", ("chat.system_base",)),
    SectionSlot("anti_hallucination", "反幻觉硬约束", ("chat.anti_hallucination_hard_rule",)),
    SectionSlot("personality", "你的身份", ("chat.personality_section",)),
    SectionSlot("consistency", "对话一致性", ("chat.consistency_rules",)),
    SectionSlot("emotion", "当前情绪", ("chat.relationship_stage_section",)),
    SectionSlot("ai_mood", "你的心情", ("chat.ai_mood_section",)),
    SectionSlot("portrait", "用户画像", ("chat.portrait_section",)),
    SectionSlot("delay_context", "回复时机说明", ("chat.delay_context_section",)),
    SectionSlot(
        "reengagement", "重逢感知",
        ("chat.reengagement_short", "chat.reengagement_long", "chat.reengagement_day"),
    ),
    SectionSlot("session_recap", "上次聊到", ("chat.session_recap_section",)),
    SectionSlot(
        "memory", "你记得的事情",
        ("chat.memory_section_body", "chat.memory_empty_anchor"),
    ),
    SectionSlot("topic_context", "话题上下文", ("chat.topic_context_section",)),
    SectionSlot("expression_habits", "表达习惯参考", ("chat.expression_habits_section",)),
    SectionSlot("music_context", "一起听音乐", ("chat.music_context_section",)),
    SectionSlot("time_context", "时间", ("chat.time_context_section",)),
    SectionSlot("time_memories", "相关时间记忆", ("chat.time_memories_section",)),
    SectionSlot("l3_memories", "久远记忆（L3）", ("chat.l3_memory_section",)),
    SectionSlot(
        "patience", "情绪状态提醒",
        (
            "boundary.patience_instruction_medium",
            "boundary.patience_instruction_low",
            "boundary.patience_instruction_blocked",
        ),
    ),
    SectionSlot("ai_state_constraint", "你的隐性状态约束", ("chat.ai_state_constraint",)),
    SectionSlot("response_instruction", "回复要求", ("chat.response_instruction",)),
)

DEFAULT_CHAT_SECTION_ORDER: list[str] = [slot.slot for slot in CHAT_SECTION_SLOTS]
_KNOWN_SLOTS = set(DEFAULT_CHAT_SECTION_ORDER)


def _normalize(order: list[str]) -> list[str]:
    """读取端归一化: 丢弃未知 slot, 缺失 slot 按默认相对顺序补到末尾.

    覆写落库后代码若新增 slot, 不能让新 section 静默消失.
    """
    seen: set[str] = set()
    result: list[str] = []
    for slot in order:
        if slot in _KNOWN_SLOTS and slot not in seen:
            result.append(slot)
            seen.add(slot)
    for slot in DEFAULT_CHAT_SECTION_ORDER:
        if slot not in seen:
            result.append(slot)
    return result


def _cache_local(order: list[str]) -> None:
    global _local_cache
    _local_cache = (order, time.monotonic() + _LOCAL_TTL_SECONDS)


def invalidate_local_cache() -> None:
    """写入后清本进程缓存 (多 worker 靠 Redis, 最多 10s 收敛 — 与 prompt
    enabled 开关一致)."""
    global _local_cache
    _local_cache = None


async def _load_from_db() -> list[str] | None:
    row = await db.promptsectionorder.find_unique(where={"promptKey": CHAT_SECTION_ORDER_KEY})
    if row is None:
        return None
    raw = row.orderJson
    raw = getattr(raw, "data", raw)  # prisma Json 输入包装对象兜底 (真实读回是原生 list)
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list) and all(isinstance(s, str) for s in raw):
        return raw
    return None


async def get_chat_section_order() -> list[str]:
    """聊天热路径入口: 进程缓存 → Redis → DB → 代码默认. 全链路防炸 —
    顺序配置任何一层坏掉都退默认顺序, 绝不让主回复挂掉."""
    if _local_cache is not None and _local_cache[1] > time.monotonic():
        return list(_local_cache[0])
    try:
        redis = await get_redis()
        raw = await redis.get(_REDIS_KEY)
        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                order = _normalize([str(s) for s in parsed])
                _cache_local(order)
                return order
        stored = await _load_from_db()
        if stored is not None:
            order = _normalize(stored)
            await redis.set(_REDIS_KEY, json.dumps(order, ensure_ascii=False))
            _cache_local(order)
            return order
        # 无覆写 → 默认; 也缓存, 避免每条消息打一次 Redis miss + DB
        _cache_local(DEFAULT_CHAT_SECTION_ORDER)
        return list(DEFAULT_CHAT_SECTION_ORDER)
    except Exception as exc:  # noqa: BLE001 — 顺序配置故障不能影响聊天
        logger.warning(f"[SECTION-ORDER] load failed, fallback to default: {exc}")
        return list(DEFAULT_CHAT_SECTION_ORDER)


async def set_chat_section_order(order: list[str]) -> dict[str, Any]:
    """写入覆写. 校验从严: 必须是全部已知 slot 的一个排列 (不多不少不重),
    防止提示词结构混乱. 返回 API 响应 payload."""
    if not isinstance(order, list) or not all(isinstance(s, str) for s in order):
        raise ValueError("order must be a list of slot ids")
    unknown = [s for s in order if s not in _KNOWN_SLOTS]
    if unknown:
        raise ValueError(f"unknown slots: {', '.join(unknown)}")
    if len(set(order)) != len(order):
        raise ValueError("duplicate slots in order")
    missing = [s for s in DEFAULT_CHAT_SECTION_ORDER if s not in order]
    if missing:
        raise ValueError(f"missing slots: {', '.join(missing)}")

    row = await db.promptsectionorder.upsert(
        where={"promptKey": CHAT_SECTION_ORDER_KEY},
        data={
            "create": {"promptKey": CHAT_SECTION_ORDER_KEY, "orderJson": Json(order)},
            "update": {"orderJson": Json(order)},
        },
    )
    redis = await get_redis()
    await redis.set(_REDIS_KEY, json.dumps(order, ensure_ascii=False))
    invalidate_local_cache()
    logger.info(f"[SECTION-ORDER] updated: {' > '.join(order)}")
    return _info_payload(order, source="custom", updated_at=str(getattr(row, "updatedAt", "") or ""))


async def reset_chat_section_order() -> dict[str, Any]:
    """删除覆写, 回到代码默认顺序."""
    try:
        await db.promptsectionorder.delete(where={"promptKey": CHAT_SECTION_ORDER_KEY})
    except Exception:
        pass  # 不存在即视为已重置
    redis = await get_redis()
    await redis.delete(_REDIS_KEY)
    invalidate_local_cache()
    logger.info("[SECTION-ORDER] reset to default")
    return _info_payload(list(DEFAULT_CHAT_SECTION_ORDER), source="default", updated_at=None)


async def get_chat_section_order_info() -> dict[str, Any]:
    """Admin API 读取: 当前生效顺序 + 默认顺序 + slot 元数据."""
    updated_at: str | None = None
    source = "default"
    order = list(DEFAULT_CHAT_SECTION_ORDER)
    try:
        stored = await _load_from_db()
        if stored is not None:
            order = _normalize(stored)
            source = "custom"
            row = await db.promptsectionorder.find_unique(
                where={"promptKey": CHAT_SECTION_ORDER_KEY},
            )
            updated_at = str(getattr(row, "updatedAt", "") or "") if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[SECTION-ORDER] info load failed, showing default: {exc}")
    return _info_payload(order, source=source, updated_at=updated_at)


def _info_payload(order: list[str], *, source: str, updated_at: str | None) -> dict[str, Any]:
    return {
        "prompt_key": CHAT_SECTION_ORDER_KEY,
        "order": order,
        "default_order": list(DEFAULT_CHAT_SECTION_ORDER),
        "source": source,
        "updated_at": updated_at,
        "slots": [
            {"slot": s.slot, "title": s.title, "prompt_keys": list(s.prompt_keys)}
            for s in CHAT_SECTION_SLOTS
        ],
    }
