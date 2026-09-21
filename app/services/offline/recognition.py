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

from app.db import db
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import chat_emit
from app.services.offline import repository as repo
from app.services.offline.memory_hooks import remember_offline_fragment
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

    去重：① 同图——调用方对每张入站图只调一次（chat_capture 一图一 media_id 一次
    识别，无 media 层重入）；② 同内容指纹不重复产出；③ 已触发条件本活动内不复用。
    （产出时 media.recognized 标 TRUE，供画廊排除与 admin 追溯，非识别期闸门。）
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


# 口语化交付时的档位口吻（PM #11/#12/#13）。
_TIER_TONE: dict[str, str] = {
    "rare": "语气自然轻松，像随口一提。",
    "epic": "语气比普通回忆更认真、更向内一些。",
    "legendary": "语气带一点秘密感，像是终于说出口。",
}


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
    items = await repo.list_untriggered_conditions(recommendation_id)
    if not items:
        return None

    # 主体识图（复用真实视觉描述）→ 匹配未触发物品。
    subjects = await _detect_subjects(photo_description)
    matched_item, matched_subject = _match_subject_to_item(subjects, items)
    if not matched_item or not matched_subject:
        await _handle_miss(activity, ctx)  # 未命中：计数 + 视情况给方向暗示
        return None

    produced = await repo.count_fragments(recommendation_id)
    if produced >= MAX_FRAGMENTS_PER_SESSION:
        return None
    fingerprint = content_fingerprint(
        [str(matched_subject.get("type") or ""), str(matched_item.get("short_name") or "")]
    )
    if fingerprint and await repo.fragment_fingerprint_exists(
        recommendation_id, fingerprint
    ):
        return None
    if not should_produce(produced):
        return None  # 概率未过：不产出，物品保持未触发

    tier = roll_tier()
    lead_in = pick_lead_in(tier)
    prewritten = await repo.get_prewritten_fragment(matched_item["id"], tier)
    text = (
        (await _verbalize(activity, ctx, tier, prewritten, matched_item, matched_subject))
        or prewritten
        or _FALLBACK_FRAGMENT.get(tier, _FALLBACK_FRAGMENT["rare"])
    )

    await repo.mark_condition_triggered(matched_item["id"])
    await repo.mark_media_fragment_cover(media_id, fingerprint)
    # 持久化"已识图"标记到源消息 metadata：让金框/左上角标记/「偶遇一缕思绪」在前端
    # 重载聊天记录后仍保留（实时 WS 只改内存态，不落库）。失败不影响碎片交付。
    try:
        await repo.mark_message_recognized(source_message_id, tier)
    except Exception as exc:  # pragma: no cover - best-effort 持久化
        logger.warning("[offline-recognition] 标记源消息失败 err=%s", exc)
    fragment = await repo.create_fragment(
        recommendation_id=recommendation_id,
        tier=tier,
        text=text,
        lead_in=lead_in,
        condition_id=matched_item["id"],
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
    # 写回 AI 记忆（#2）：让这次旅途"想起"的思绪进入 AI 长期记忆，不游离于聊天系统之外。
    # 走保证写入（绕过"记/不记"预筛），否则碎片可能被会话式门控丢弃。
    remember_offline_fragment(
        user_id=activity["user_id"],
        workspace_id=activity.get("workspace_id"),
        text=text,
        location=activity.get("location_name") or activity.get("title"),
    )
    logger.info(
        "[offline-recognition] activity=%s 产出碎片 tier=%s produced=%d->%d",
        recommendation_id, tier, produced, produced + 1,
    )
    return fragment


def _match_subject_to_item(
    subjects: list[dict[str, Any]], items: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """按置信度从高到低，找第一个能对上某个未触发物品的主体。"""
    for subject in subjects:
        t = str(subject.get("type") or "").strip()
        if not t:
            continue
        for item in items:
            sn = str(item.get("short_name") or "").strip()
            cat = str(item.get("category") or "").strip()
            if (
                (sn and (sn == t or sn in t or t in sn))
                or (cat and (cat == t or cat in t or t in cat))
            ):
                return item, subject
    return None, None


async def _detect_subjects(photo_description: str) -> list[dict[str, Any]]:
    try:
        prompt = (await get_prompt_text("offline.photo_subjects")).format(
            photo_description=photo_description,
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_subjects(raw)
    except Exception as exc:
        logger.warning("[offline-recognition] 主体识图失败 err=%s", exc)
        return []


def _parse_subjects(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    subs = data.get("subjects") if isinstance(data, dict) else None
    if not isinstance(subs, list):
        return []
    out: list[dict[str, Any]] = []
    for s in subs:
        if not isinstance(s, dict):
            continue
        t = str(s.get("type") or "").strip()
        if not t:
            continue
        try:
            conf = float(s.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        out.append({"type": t, "confidence": conf})
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out


async def _verbalize(
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    tier: str,
    prewritten: str | None,
    item: dict[str, Any],
    subject: dict[str, Any],
) -> str:
    if not prewritten:
        return ""
    try:
        conversation_id = (ctx or {}).get("conversation_id") or activity.get(
            "conversation_id"
        )
        recent = await _recent_dialogue(conversation_id)
        prompt = (await get_prompt_text("offline.fragment_verbalize")).format(
            tier_tone=_TIER_TONE.get(tier, ""),
            prewritten_text=prewritten,
            matched_item=item.get("short_name") or "",
            photo_subject=subject.get("type") or "",
            recent_dialogue=recent or "（无）",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_text_field(raw)
    except Exception as exc:
        logger.warning("[offline-recognition] 口语化交付失败 err=%s", exc)
        return ""


async def _handle_miss(
    activity: dict[str, Any], ctx: dict[str, Any] | None
) -> None:
    """未命中：miss_count+1；每累计 2 次给一次方向暗示，最多 3 次（PM #9）。"""
    recommendation_id = activity["id"]
    try:
        miss = await repo.increment_activity_counter(recommendation_id, "miss_count")
    except Exception:
        return
    if miss % 2 != 0 or int(activity.get("hint_count") or 0) >= 3:
        return
    conversation_id = (ctx or {}).get("conversation_id") or activity.get(
        "conversation_id"
    )
    agent_id = (ctx or {}).get("agent_id") or activity.get("agent_id")
    if not (conversation_id and agent_id):
        return
    untriggered = await repo.list_untriggered_conditions(recommendation_id)
    all_items = await repo.list_all_conditions(recommendation_id)
    untriggered_ids = {i["id"] for i in untriggered}
    triggered = [
        str(i.get("short_name") or "")
        for i in all_items
        if i["id"] not in untriggered_ids
    ]
    hint = await _generate_miss_hint(
        activity, miss, int(activity.get("hint_count") or 0), untriggered, triggered
    )
    if not hint:
        return
    await repo.increment_activity_counter(recommendation_id, "hint_count")
    await chat_emit.emit_assistant(
        conversation_id=str(conversation_id),
        user_id=activity["user_id"],
        agent_id=str(agent_id),
        workspace_id=activity.get("workspace_id"),
        message=hint,
        real_world_type="activity",
        source_id=recommendation_id,
        trigger_type="offline_activity_miss_hint",
    )


async def _generate_miss_hint(
    activity: dict[str, Any],
    miss_count: int,
    hint_count: int,
    hintable_items: list[dict[str, Any]],
    triggered_items: list[str],
) -> str:
    try:
        prompt = (await get_prompt_text("offline.miss_hint")).format(
            miss_count=miss_count,
            hint_count=hint_count,
            hintable_items="、".join(
                str(i.get("short_name") or "") for i in hintable_items
            ) or "（无）",
            triggered_items="、".join(triggered_items) or "（无）",
            location_info=activity.get("title") or activity.get("location_name") or "",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_text_field(raw)
    except Exception as exc:
        logger.warning("[offline-recognition] 未命中暗示生成失败 err=%s", exc)
        return ""


async def _recent_dialogue(conversation_id: str | None, limit: int = 6) -> str:
    if not conversation_id:
        return ""
    try:
        rows = await db.message.find_many(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
            take=limit,
        )
        lines: list[str] = []
        for m in reversed(rows or []):
            content = str(getattr(m, "content", "") or "").strip()
            if not content:
                continue
            who = "我" if getattr(m, "role", "") == "assistant" else "用户"
            lines.append(f"{who}：{content}")
        return "\n".join(lines)[:1500]
    except Exception:
        return ""


def _parse_text_field(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("text"):
            return str(data["text"]).strip()
    except Exception:
        match = re.search(r'"text"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1).strip()
    return ""
