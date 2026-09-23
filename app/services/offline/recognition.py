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
import math
import random
import re
from typing import Any

from app.config import settings
from app.db import db
from app.observability.events import (
    EVT_OFFLINE_FOCUS_SWITCHED,
    EVT_OFFLINE_PHOTO_MATCH,
)
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import chat_emit
from app.services.offline import repository as repo
from app.services.offline.guidance import (
    contains_hidden_target,
    forbidden_terms,
    safe_guidance,
    sanitize_visible_text,
)
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
    allow_miss_hint: bool = True,
    allow_near_followup: bool = True,
) -> dict[str, Any] | None:
    """识图匹配 + 概率阶梯 + 碎片产出（spec §4.8/§4.9）。服务端权威、单场串行。

    去重：① 同图——调用方对每张入站图只调一次（chat_capture 一图一 media_id 一次
    识别，无 media 层重入）；② 同内容指纹不重复产出；③ 已触发条件本活动内不复用。
    （产出时 media.recognized 标 TRUE，供画廊排除与 admin 追溯，非识别期闸门。）
    命中但概率未过 → 不产出且条件保持未触发（后续可再命中）。
    """
    recommendation_id = activity["id"]
    if (
        activity.get("status") != "accepted"
        or not activity.get("reached")
        or not activity.get("conditions_ready_at")
    ):
        return None
    async with _lock_for(recommendation_id):
        return await _recognize_locked(
            activity=activity,
            ctx=ctx,
            photo_description=photo_description,
            media_id=media_id,
            source_message_id=source_message_id,
            trace_id=trace_id,
            allow_miss_hint=allow_miss_hint,
            allow_near_followup=allow_near_followup,
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
    allow_miss_hint: bool,
    allow_near_followup: bool,
) -> dict[str, Any] | None:
    recommendation_id = activity["id"]
    if not (photo_description or "").strip():
        return None
    items = await repo.list_untriggered_conditions(recommendation_id)
    if not items:
        return None

    match = await _classify_photo_match(
        photo_description,
        items,
        focus_condition_id=str(activity.get("focus_condition_id") or ""),
    )
    relation = str(match.get("relation") or "none")
    matched_id = str(match.get("condition_id") or "")
    matched_item = next(
        (item for item in items if str(item.get("id")) == matched_id),
        None,
    )
    logger.info(
        "[offline-recognition] relation=%s confidence=%.2f",
        relation,
        float(match.get("confidence") or 0.0),
        extra={
            "event": EVT_OFFLINE_PHOTO_MATCH,
            "activity_id": recommendation_id,
            "relation": relation,
            "confidence": float(match.get("confidence") or 0.0),
            "matched_focus": bool(matched_id and matched_id == str(
                activity.get("focus_condition_id") or ""
            )),
        },
    )
    if relation == "none" or not matched_item:
        await _handle_miss(
            activity,
            ctx,
            items,
            source_message_id=source_message_id,
            allow_hint=allow_miss_hint,
        )
        return None

    observed_subject = str(match.get("observed_subject") or "").strip()
    focus_id = str(activity.get("focus_condition_id") or "")
    switched = bool(focus_id and matched_id != focus_id)

    if relation == "near":
        if switched or not focus_id:
            await repo.set_activity_focus(
                recommendation_id,
                matched_id,
                reason="photo_near_other" if switched else "photo_near",
            )
            logger.info(
                "[offline-recognition] focus switched on near match",
                extra={
                    "event": EVT_OFFLINE_FOCUS_SWITCHED,
                    "activity_id": recommendation_id,
                    "reason": "photo_near_other" if switched else "photo_near",
                },
            )
        await repo.reset_activity_misses(recommendation_id)
        if not allow_near_followup:
            return None
        if not await repo.claim_activity_followup(source_message_id):
            return None
        await _emit_photo_followup(
            activity=activity,
            ctx=ctx,
            photo_description=photo_description,
            relation="near",
            next_item=matched_item,
            all_conditions=items,
            source_message_id=source_message_id,
            trace_id=trace_id,
        )
        return None

    # Exact matching completes the hidden condition independently of the
    # fragment reward probability.
    completed = await repo.mark_condition_triggered(matched_id)
    if not completed:
        return None
    await repo.reset_activity_misses(recommendation_id)
    remaining = [item for item in items if str(item.get("id")) != matched_id]
    next_item = random.choice(remaining) if remaining else None
    await repo.set_activity_focus(
        recommendation_id,
        str(next_item["id"]) if next_item else None,
        reason="photo_exact_next" if next_item else "all_conditions_completed",
    )
    logger.info(
        "[offline-recognition] focus advanced after exact match",
        extra={
            "event": EVT_OFFLINE_FOCUS_SWITCHED,
            "activity_id": recommendation_id,
            "reason": "photo_exact_next" if next_item else "free_roam",
        },
    )
    if not await repo.claim_activity_followup(source_message_id):
        return None

    fragment: dict[str, Any] | None = None
    produced = await repo.count_fragments(recommendation_id)
    fingerprint = content_fingerprint(
        [observed_subject, str(matched_item.get("short_name") or "")]
    )
    duplicate = bool(
        fingerprint
        and await repo.fragment_fingerprint_exists(recommendation_id, fingerprint)
    )
    if (
        produced < MAX_FRAGMENTS_PER_SESSION
        and not duplicate
        and should_produce(produced)
    ):
        tier = roll_tier()
        lead_in = pick_lead_in(tier)
        prewritten = await repo.get_prewritten_fragment(matched_item["id"], tier)
        text = (
            (
                await _verbalize(
                    activity,
                    ctx,
                    tier,
                    prewritten,
                    photo_description,
                )
            )
            or prewritten
            or _FALLBACK_FRAGMENT.get(tier, _FALLBACK_FRAGMENT["rare"])
        )
        text = await _guard_visible_message(
            text,
            items,
            safe_hint="",
            fallback=_FALLBACK_FRAGMENT.get(tier, _FALLBACK_FRAGMENT["rare"]),
        )

        await repo.mark_media_fragment_cover(media_id, fingerprint)
        try:
            await repo.mark_message_recognized(source_message_id, tier)
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            logger.warning("[offline-recognition] failed to mark source message: %s", exc)
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
    if conversation_id and source_message_id:
        await _wait_for_main_reply(str(conversation_id), source_message_id)
    if fragment and conversation_id and agent_id:
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
    if fragment:
        remember_offline_fragment(
            user_id=activity["user_id"],
            workspace_id=activity.get("workspace_id"),
            text=str(fragment.get("text") or ""),
            location=activity.get("location_name") or activity.get("title"),
        )
    await _emit_photo_followup(
        activity=activity,
        ctx=ctx,
        photo_description=photo_description,
        relation="exact",
        next_item=next_item,
        all_conditions=items,
        source_message_id=source_message_id,
        trace_id=trace_id,
    )
    logger.info(
        "[offline-recognition] activity=%s exact condition=%s fragment=%s",
        recommendation_id,
        matched_id,
        bool(fragment),
    )
    return fragment


async def _classify_photo_match(
    photo_description: str,
    items: list[dict[str, Any]],
    *,
    focus_condition_id: str,
) -> dict[str, Any]:
    """Classify exact/near/none; fall back to the legacy subject matcher."""
    candidates = []
    valid_ids: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        valid_ids.add(item_id)
        profile = item.get("guidance_profile")
        aliases = profile.get("aliases") if isinstance(profile, dict) else []
        candidates.append(
            {
                "condition_id": item_id,
                "short_name": item.get("short_name") or "",
                "category": item.get("category") or "",
                "criteria": item.get("criteria") or "",
                "aliases": aliases if isinstance(aliases, list) else [],
            }
        )
    try:
        prompt = (await get_prompt_text("offline.photo_match")).format(
            photo_description=photo_description,
            focus_condition_id=focus_condition_id or "（无）",
            conditions_json=json.dumps(candidates, ensure_ascii=False),
        )
        raw = await invoke_text(get_chat_model(), prompt)
        parsed = _parse_photo_match(raw)
        condition_id = str(parsed.get("condition_id") or "")
        relation = str(parsed.get("relation") or "none").lower()
        confidence = _normalized_confidence(parsed.get("confidence"))
        if condition_id not in valid_ids:
            relation = "none"
            condition_id = ""
        if relation == "exact" and confidence < settings.offline_activity_photo_exact_threshold:
            relation = (
                "near"
                if confidence >= settings.offline_activity_photo_near_threshold
                else "none"
            )
        elif (
            relation == "near"
            and confidence < settings.offline_activity_photo_near_threshold
        ) or relation not in {"exact", "near"}:
            relation = "none"
        if relation == "none":
            condition_id = ""
        return {
            **parsed,
            "relation": relation,
            "condition_id": condition_id,
            "confidence": confidence,
        }
    except Exception as exc:  # noqa: BLE001 - classifier failure uses fallback
        logger.warning("[offline-recognition] photo match classification failed: %s", exc)

    subjects = await _detect_subjects(photo_description)
    matched_item, matched_subject = _match_subject_to_item(subjects, items)
    if not matched_item or not matched_subject:
        return {"relation": "none", "condition_id": "", "confidence": 0.0}
    confidence = _normalized_confidence(matched_subject.get("confidence"))
    if confidence >= settings.offline_activity_photo_exact_threshold:
        relation = "exact"
    elif confidence >= settings.offline_activity_photo_near_threshold:
        relation = "near"
    else:
        return {"relation": "none", "condition_id": "", "confidence": confidence}
    return {
        "relation": relation,
        "condition_id": str(matched_item.get("id") or ""),
        "confidence": confidence,
        "observed_subject": str(matched_subject.get("type") or ""),
        "reason": "legacy_subject_fallback",
    }


def _normalized_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _parse_photo_match(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return {}
    return data if isinstance(data, dict) else {}


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
    except Exception as exc:  # noqa: BLE001 - user-facing fallback is mandatory
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
    photo_description: str,
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
            photo_description=photo_description,
            recent_dialogue=recent or "（无）",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_text_field(raw)
    except Exception as exc:  # noqa: BLE001 - safe fallback is mandatory
        logger.warning("[offline-recognition] 口语化交付失败 err=%s", exc)
        return ""


async def _handle_miss(
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    untriggered: list[dict[str, Any]],
    *,
    source_message_id: str | None,
    allow_hint: bool,
) -> None:
    """未命中：miss_count+1；每累计 2 次给一次方向暗示，最多 3 次（PM #9）。"""
    recommendation_id = activity["id"]
    try:
        miss = await repo.increment_activity_counter(recommendation_id, "miss_count")
    except Exception:
        return
    if not allow_hint:
        return
    current = await repo.get_activity(
        recommendation_id,
        activity["user_id"],
        reveal_task=True,
    )
    hint_count = int((current or activity).get("hint_count") or 0)
    if miss % 2 != 0 or hint_count >= 3:
        return
    conversation_id = (ctx or {}).get("conversation_id") or activity.get(
        "conversation_id"
    )
    agent_id = (ctx or {}).get("agent_id") or activity.get("agent_id")
    if not (conversation_id and agent_id):
        return
    if source_message_id:
        await _wait_for_main_reply(str(conversation_id), source_message_id)
    focus_id = str((current or activity).get("focus_condition_id") or "")
    focus_item = next(
        (item for item in untriggered if str(item.get("id")) == focus_id),
        untriggered[0] if untriggered else None,
    )
    if not focus_item:
        return
    hint = await _generate_miss_hint(
        activity,
        hint_count,
        focus_item,
        untriggered,
    )
    if not hint:
        return
    if not await repo.claim_activity_followup(source_message_id):
        return
    await repo.increment_activity_counter(recommendation_id, "hint_count")
    if source_message_id:
        await _wait_for_main_reply(str(conversation_id), source_message_id)
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
    hint_count: int,
    focus_item: dict[str, Any],
    all_conditions: list[dict[str, Any]],
) -> str:
    try:
        level = ("weak", "medium", "strong")[min(hint_count, 2)]
        fallback = safe_guidance(focus_item, level)
        prompt = (await get_prompt_text("offline.miss_hint")).format(
            safe_hint=fallback,
            location_info=activity.get("title") or activity.get("location_name") or "",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return await _guard_visible_message(
            _parse_text_field(raw),
            all_conditions,
            safe_hint=fallback,
            fallback=fallback,
        )
    except Exception as exc:  # noqa: BLE001 - user-facing fallback is mandatory
        logger.warning("[offline-recognition] 未命中暗示生成失败 err=%s", exc)
        return ""


_MECHANICAL_TERMS_RE = re.compile(
    r"(任务|目标|拍对|拍错|命中|完成|进度|进展|还剩|切换|隐藏|"
    r"解锁|下一关|收集|通关|拍摄要求)"
)


async def _emit_photo_followup(
    *,
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    photo_description: str,
    relation: str,
    next_item: dict[str, Any] | None,
    all_conditions: list[dict[str, Any]],
    source_message_id: str | None,
    trace_id: str | None,
) -> None:
    conversation_id = (ctx or {}).get("conversation_id") or activity.get(
        "conversation_id"
    )
    agent_id = (ctx or {}).get("agent_id") or activity.get("agent_id")
    if not (conversation_id and agent_id):
        return
    next_hint = safe_guidance(next_item, "weak") if next_item else ""
    if relation == "near":
        fallback = (
            f"这张里的感觉挺特别。{next_hint}"
            if next_hint
            else "这张挺有意思的，先顺着你刚才注意到的方向逛逛。"
        )
    elif next_item:
        fallback = f"这张挺有感觉的。{next_hint}"
    else:
        fallback = "这张我喜欢。接下来不用特意找什么，慢慢逛、看到喜欢的就发我。"
    try:
        recent = await _recent_dialogue(str(conversation_id))
        prompt_key = (
            "offline.photo_followup_near"
            if relation == "near"
            else "offline.photo_followup_transition"
            if next_item
            else "offline.photo_followup_free_roam"
        )
        prompt = (await get_prompt_text(prompt_key)).format(
            agent_name=(ctx or {}).get("agent_name") or "伴生",
            photo_description=photo_description,
            next_safe_hint=next_hint or "（无，进入随意闲逛）",
            recent_dialogue=recent or "（无）",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        draft = _parse_text_field(raw)
    except Exception as exc:  # noqa: BLE001 - user-facing fallback is mandatory
        logger.warning("[offline-recognition] photo follow-up failed: %s", exc)
        draft = ""
    message = await _guard_visible_message(
        draft,
        all_conditions,
        safe_hint=next_hint,
        fallback=fallback,
    )
    await chat_emit.emit_assistant(
        conversation_id=str(conversation_id),
        user_id=activity["user_id"],
        agent_id=str(agent_id),
        workspace_id=activity.get("workspace_id"),
        message=message,
        real_world_type="activity",
        source_id=activity["id"],
        trigger_type="offline_activity_photo_followup",
        trace_id=trace_id,
    )


async def _guard_visible_message(
    draft: str,
    conditions: list[dict[str, Any]],
    *,
    safe_hint: str,
    fallback: str,
) -> str:
    candidate = str(draft or "").strip()
    violates = (
        not candidate
        or contains_hidden_target(candidate, conditions)
        or bool(_MECHANICAL_TERMS_RE.search(candidate))
    )
    if not violates:
        return candidate
    terms = sorted(
        {
            term
            for condition in conditions
            for term in forbidden_terms(condition, include_category=True)
        }
    )
    try:
        prompt = (await get_prompt_text("offline.safe_rewrite")).format(
            draft=candidate or fallback,
            forbidden_terms="、".join(terms) or "（无）",
            safe_hint=safe_hint or "只聊照片本身",
        )
        rewritten = _parse_text_field(await invoke_text(get_chat_model(), prompt))
        if (
            rewritten
            and not contains_hidden_target(rewritten, conditions)
            and not _MECHANICAL_TERMS_RE.search(rewritten)
        ):
            return rewritten
    except Exception as exc:  # noqa: BLE001 - safe fallback is mandatory
        logger.warning("[offline-recognition] safe rewrite failed: %s", exc)
    return sanitize_visible_text(fallback, conditions, fallback="我看到啦，这张挺有感觉的。")


async def _wait_for_main_reply(
    conversation_id: str,
    source_message_id: str,
    *,
    attempts: int = 24,
    interval_seconds: float = 0.5,
) -> None:
    """Keep offline follow-ups behind the normal photo reply."""
    for _ in range(max(1, attempts)):
        try:
            rows = await db.query_raw(
                """
                SELECT 1
                FROM messages reply
                WHERE reply.conversation_id = $1
                  AND reply.role = 'assistant'
                  AND reply.created_at > (
                      SELECT created_at FROM messages WHERE id = $2
                  )
                  AND COALESCE(reply.metadata->>'proactive', 'false') <> 'true'
                LIMIT 1
                """,
                conversation_id,
                source_message_id,
            )
            if rows:
                return
        except Exception:  # noqa: BLE001 - ordering wait is best-effort
            return
        await asyncio.sleep(interval_seconds)


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
