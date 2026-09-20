"""思绪碎片触发引擎（spec §4.8/§4.9/§4.10）—— 纯服务端权威。

本文件的纯函数（概率阶梯 / 等级滚动 / 内容指纹）不依赖 DB 或大模型，便于单测与
离线推演。识图匹配 + 编排（recognize_on_photo）在同文件下半部分，调用 vision 模型
与仓储，串起 spec 的三重去重与概率门槛。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from typing import Any

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import chat_emit
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

# 碎片等级滚动分布（spec §4.10）：仅在「已决定产出」后滚。
_TIER_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("rare", 68.0),      # 片刻感想
    ("epic", 25.0),      # 心底独白
    ("legendary", 7.0),  # 秘密念想
)

TIER_LABELS: dict[str, str] = {
    "rare": "片刻感想",
    "epic": "心底独白",
    "legendary": "秘密念想",
}

# 概率阶梯（spec §4.9）：键为本场「已成功产出」碎片数。
_PRODUCE_PROBABILITY: dict[int, float] = {0: 1.0, 1: 0.4, 2: 0.25}
MAX_FRAGMENTS_PER_SESSION = 3

# 等级引导语池（界面展示，不含拍摄条件）。
_LEAD_INS: dict[str, tuple[str, ...]] = {
    "rare": (
        "看着这张照片，我心里有个片刻的感想，想说给你听——",
        "你把这一幕递给我，我忽然停了一下，然后想到——",
    ),
    "epic": (
        "这张照片让我走神了很久。像有一段旧事自己冒了出来——",
        "看见你拍的这一幕，我好像被拽回某个具体的时刻——",
    ),
    "legendary": (
        "……我本来不太想说的。但看着这张照片，还是想告诉你——",
        "有些话平时说不出口。你拍下这一幕时，它自己浮上来了——",
    ),
}


def roll_tier(rng: random.Random | None = None) -> str:
    r = (rng or random).uniform(0, 100)
    acc = 0.0
    for tier, weight in _TIER_WEIGHTS:
        acc += weight
        if r < acc:
            return tier
    return "rare"


def produce_probability(produced_count: int) -> float:
    """按本场已产出碎片数返回本次命中后的产出概率。满 3 条关闭。"""
    if produced_count >= MAX_FRAGMENTS_PER_SESSION:
        return 0.0
    return _PRODUCE_PROBABILITY.get(produced_count, 0.0)


def should_produce(produced_count: int, rng: random.Random | None = None) -> bool:
    p = produce_probability(produced_count)
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return (rng or random).random() < p


def pick_lead_in(tier: str, rng: random.Random | None = None) -> str:
    pool = _LEAD_INS.get(tier) or _LEAD_INS["rare"]
    return (rng or random).choice(pool)


def content_fingerprint(keywords: list[str]) -> str:
    """把识图关键词归一化成稳定指纹，用于「同内容不重复出碎片」去重。"""
    norm = sorted({str(k).strip().lower() for k in keywords if str(k).strip()})
    if not norm:
        return ""
    return hashlib.sha1("|".join(norm).encode("utf-8")).hexdigest()[:16]


_FALLBACK_FRAGMENT: dict[str, str] = {
    "rare": "这一幕挺好的，我想把这一刻记一下。",
    "epic": "看着这张照片，心里忽然安静了一下，像想起了什么。",
    "legendary": "有些话我很少说——此刻却很想让你知道，我在认真陪你走这一趟。",
}

# 单场串行（spec §4）：同一活动的多张照片在同一 event loop 上依次决策，避免并发
# 下多张都读到 produced=0 而各自按「首命中 100%」出碎片、绕过概率阶梯。用户自己的
# 照片都走其单一 WS 连接（同一 worker/loop），进程内锁即可串行；跨 worker 同用户
# 同时发图属极端罕见，不在此处强一致范围。
_activity_locks: dict[str, asyncio.Lock] = {}


def _lock_for(recommendation_id: str) -> asyncio.Lock:
    lock = _activity_locks.get(recommendation_id)
    if lock is None:
        lock = asyncio.Lock()
        _activity_locks[recommendation_id] = lock
    return lock


async def recognize_on_photo(
    *,
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    photo_description: str,
    media_id: str,
    source_message_id: str | None,
    trace_id: str | None = None,
) -> dict[str, Any] | None:
    """识图匹配 + 概率阶梯 + 碎片产出（spec §4.8/§4.9）。服务端权威、单场串行。

    三重去重：同图（调用方一图一次 + media.recognized）/ 同内容指纹 / 已触发条件不复用。
    命中但概率未过 → 不产出且条件保持未触发（后续可再命中）。
    """
    recommendation_id = activity["id"]
    if activity.get("status") != "accepted" or not activity.get("reached"):
        return None
    async with _lock_for(recommendation_id):
        return await _recognize_locked(
            activity=activity,
            ctx=ctx,
            photo_description=photo_description,
            media_id=media_id,
            source_message_id=source_message_id,
            trace_id=trace_id,
        )


async def _recognize_locked(
    *,
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    photo_description: str,
    media_id: str,
    source_message_id: str | None,
    trace_id: str | None,
) -> dict[str, Any] | None:
    recommendation_id = activity["id"]
    if not (photo_description or "").strip():
        return None
    conditions = await repo.list_untriggered_conditions(recommendation_id)
    if not conditions:
        return None
    produced = await repo.count_fragments(recommendation_id)
    if produced >= MAX_FRAGMENTS_PER_SESSION:
        return None

    match = await _match_conditions(photo_description, conditions)
    if not match or not match.get("hit"):
        return None
    fingerprint = content_fingerprint(match.get("keywords") or [])
    if fingerprint and await repo.fragment_fingerprint_exists(
        recommendation_id, fingerprint
    ):
        return None
    if not should_produce(produced):
        return None  # 概率未过：不产出，命中的条件保持未触发

    condition = _find_condition(conditions, match.get("condition_short_name"))
    tier = roll_tier()
    lead_in = pick_lead_in(tier)
    text = (await _fragment_text(activity, tier, match.get("keywords") or [])) or \
        _FALLBACK_FRAGMENT.get(tier, _FALLBACK_FRAGMENT["rare"])

    if condition:
        await repo.mark_condition_triggered(condition["id"])
    await repo.mark_media_fragment_cover(media_id, fingerprint)
    fragment = await repo.create_fragment(
        recommendation_id=recommendation_id,
        tier=tier,
        text=text,
        lead_in=lead_in,
        condition_id=condition["id"] if condition else None,
        snapshot_media_id=media_id,
        source_message_id=source_message_id,
        content_fingerprint=fingerprint,
    )
    conversation_id = (ctx or {}).get("conversation_id") or activity.get(
        "conversation_id"
    )
    agent_id = (ctx or {}).get("agent_id") or activity.get("agent_id")
    if conversation_id and agent_id:
        await chat_emit.emit_thought_fragment(
            conversation_id=str(conversation_id),
            user_id=activity["user_id"],
            agent_id=str(agent_id),
            workspace_id=activity.get("workspace_id"),
            activity_id=recommendation_id,
            fragment=fragment,
            source_message_id=source_message_id,
            trace_id=trace_id,
        )
    logger.info(
        "[offline-recognition] activity=%s 产出碎片 tier=%s produced=%d->%d",
        recommendation_id, tier, produced, produced + 1,
    )
    return fragment


def _find_condition(
    conditions: list[dict[str, Any]], short_name: str | None
) -> dict[str, Any] | None:
    if not short_name:
        return None
    for cond in conditions:
        if cond.get("short_name") == short_name:
            return cond
    return None


async def _match_conditions(
    photo_description: str, conditions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    cond_lines = "\n".join(
        f"- {c['short_name']} — {c['criteria']}" for c in conditions
    )
    try:
        prompt = (await get_prompt_text("offline.photo_recognition")).format(
            photo_description=photo_description,
            conditions=cond_lines,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_match(raw)
    except Exception as exc:
        logger.warning("[offline-recognition] 识图匹配失败 err=%s", exc)
        return None


def _parse_match(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return {
        "hit": bool(data.get("hit")),
        "condition_short_name": (
            str(data.get("condition_short_name") or "").strip() or None
        ),
        "keywords": [
            str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()
        ],
    }


async def _fragment_text(
    activity: dict[str, Any], tier: str, keywords: list[str]
) -> str:
    try:
        memory = await repo.memory_brief(
            activity["user_id"], activity.get("workspace_id"), limit=30
        )
        prompt = (await get_prompt_text("offline.thought_fragment")).format(
            place=activity.get("title") or activity.get("location_name") or "",
            keywords="、".join(keywords) if keywords else "（无）",
            memory=memory or "（暂无）",
            tier_label=TIER_LABELS.get(tier, "片刻感想"),
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return (raw or "").strip()
    except Exception as exc:
        logger.warning("[offline-recognition] 碎片正文生成失败 err=%s", exc)
        return ""
