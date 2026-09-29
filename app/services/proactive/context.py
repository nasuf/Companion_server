"""主动聊天上下文构建。"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.portrait import get_latest_portrait
from app.services.prompting.utils import render_prompt
from app.services.relationship.ai_mood import load_ai_mood
from app.services.relationship.intimacy import get_relationship_stage, get_topic_intimacy
from app.services.memory.storage import repo as memory_repo
from app.services.memory.core_memory import load_core_memory_strings
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status
from app.services.schedule_domain.time_service import _TZ

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 《主动交流提示词》提示词3「带时间感知历史对话搭话」的素材: 近两天带时间戳的聊天记录
RECENT_DIALOGUE_WINDOW = timedelta(hours=48)
RECENT_DIALOGUE_LIMIT = 60
# 只有"发生过的事"才能问后来怎样; 偏好/身份/思维类事实带时间反而像翻档案
# (「三周前你说你 28 岁」)
_TIME_HINT_CATEGORIES = frozenset({"生活", "情绪"})
# 超过这个时长的用户记忆不标时间 (太久远的时间感只会显得在翻档案)
USER_MEMORY_TIME_HINT_MAX_AGE = timedelta(days=60)
# 近两天没聊过时的兜底来源: 记忆主动抽不到会整次取消 (spec §4.2) → AI 人设记忆
# (建号即有); 沉默唤醒 → 打招呼
_SOURCE_FALLBACK = {
    ("memory_proactive", "recent_dialogue"): "ai_l1",
    ("silence_wakeup", "recent_dialogue"): "greeting",
}


async def build_proactive_context(
    *,
    workspace_id: str,
    user_id: str,
    agent_id: str,
    trigger_type: str,
    stage: str,
    exclude_memory_ids: set[str] | None = None,
    source: str | None = None,
    topic_theme: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    # 9 个独立 I/O 并发 (DB / Redis / LLM rerank). _load_proactive_memories
    # 含 utility LLM 调用是最长尾, 跟其余 DB 读并行可让 LLM 时间被吸收.
    (
        agent, schedule, core_memories,
        proactive_memories_pair, topic_intimacy,
        user_portrait, recent_context, ai_mood,
    ) = await asyncio.gather(
        db.aiagent.find_unique(where={"id": agent_id}),
        get_cached_schedule(agent_id),
        load_core_memory_strings(user_id=user_id, workspace_id=workspace_id, source="user"),
        _load_proactive_memories(
            user_id=user_id,
            workspace_id=workspace_id,
            source=source,
            exclude_memory_ids=exclude_memory_ids,
            topic_theme=topic_theme,
        ),
        get_topic_intimacy(agent_id, user_id),
        get_latest_portrait(user_id, agent_id),
        _load_recent_context(workspace_id),
        # AI 上一轮的残留情绪 (30min 半衰期衰减后). 主动消息的 current_mood 靠它 —
        # 没有它 ctx 里永远没 "emotion" 键, emotion_to_tone(None) 恒为中性语气.
        load_ai_mood(conversation_id),
    )
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    proactive_memories, used_memory_ids = proactive_memories_pair
    recent_dialogue = (
        await _load_recent_dialogue(workspace_id) if source == "recent_dialogue" else ""
    )
    fallback_source = _SOURCE_FALLBACK.get((trigger_type, source or ""))
    if source == "recent_dialogue" and not recent_dialogue and fallback_source:
        source = fallback_source
        proactive_memories, used_memory_ids = await _load_proactive_memories(
            user_id=user_id,
            workspace_id=workspace_id,
            source=source,
            exclude_memory_ids=exclude_memory_ids,
            topic_theme=topic_theme,
        )
    schedule_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "status": "idle", "type": "leisure"}
    relationship_stage = get_relationship_stage(topic_intimacy)
    scene_hint = _build_scene_hint(trigger_type, schedule_status)

    return {
        "agent": agent,
        "schedule_status": schedule_status,
        # core_memory now returns (category, text) tuples; extract text for
        # downstream consumers that expect plain strings.
        "core_memories": [t[1] if isinstance(t, tuple) else t for t in core_memories[:8]],
        "proactive_memories": proactive_memories[:6],
        "used_memory_ids": used_memory_ids,
        "recent_dialogue": recent_dialogue,
        "relationship_stage": relationship_stage,
        "topic_intimacy": topic_intimacy,
        # emotion_to_tone 读的就是这个形状 ({"emotion": 标签, ...}); None → 中性语气
        "emotion": ai_mood,
        "scene_hint": scene_hint,
        "trigger_type": trigger_type,
        "stage": stage,
        "source": source or "greeting",
        "topic_theme": topic_theme or "",
        "user_portrait": user_portrait or "",
        "recent_context": recent_context,
        # 2026-09-14 task#12: _generate_message 用它做 anti-repetition (查 Redis
        # 里 workspace 最近 24h 的主动消息, 一模一样 or 高度相似的就重生成一次)
        "workspace_id": workspace_id,
    }


async def _load_recent_context(workspace_id: str, limit: int = 6) -> str:
    """Spec §4.1 step 4 汇总参考信息里的"近期对话上下文"。

    取工作空间最近 N 条消息（用户+AI 混排），按时间正序拼成文本。
    """
    try:
        rows = await db.query_raw(
            """
            SELECT m.role, m.content, m.created_at
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.workspace_id = $1
              AND c.is_deleted = FALSE
            ORDER BY m.created_at DESC
            LIMIT $2
            """,
            workspace_id,
            limit,
        )
    except Exception:
        return ""
    if not rows:
        return ""
    # rows are newest-first; flip to chronological
    lines = []
    for r in reversed(rows):
        role = r.get("role") or "user"
        text = (r.get("content") or "").strip()
        if not text:
            continue
        prefix = "AI" if role == "assistant" else "用户"
        lines.append(f"{prefix}: {text[:80]}")
    return "\n".join(lines)


async def _rerank_memories_by_topic(
    rows: list, topic_theme: str
) -> list[str]:
    """Spec §3.2 + §4.2: utility model 从候选中挑出最贴 topic 的 ≤3 个 id.

    失败 / 空结果 → 返回 [], 调用方回退到 importance 倒排兜底.
    Caller (`_load_proactive_memories`) 已过滤空 content 行,
    rows 进来都有内容.
    """
    if not topic_theme or not rows:
        return []
    candidates = [{"id": r.id, "text": r.content[:80]} for r in rows]
    result = await render_prompt(
        "proactive.memory_topic_rerank",
        {
            "topic": topic_theme,
            "candidates": json.dumps(candidates, ensure_ascii=False),
        },
        lambda p: invoke_json(get_utility_model(), p),
    )
    ids = result.get("ids") if isinstance(result, dict) else None
    if not isinstance(ids, list) or not ids:
        # 监控 fallback 占比, 若长期居高说明 utility model 不稳, 体感"AI 老聊
        # 同样的事". grep `[REPLY-RERANK fallback=` 即可统计.
        reason = "render_failed" if result is None else "empty_ids"
        logger.info(
            f"[REPLY-RERANK fallback=importance] reason={reason} "
            f"topic={topic_theme!r} candidates={len(candidates)}"
        )
        return []
    valid = {c["id"] for c in candidates}
    return [str(i) for i in ids if str(i) in valid][:3]


async def _load_proactive_memories(
    *,
    user_id: str,
    workspace_id: str,
    source: str | None = None,
    exclude_memory_ids: set[str] | None = None,
    topic_theme: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load proactive memories with dedup support.

    spec §4.1/§4.2: 来源 (source) 决定从 A 库(ai)还是 B 库(user), 以及 L1 / L2 层级.
    - ai_l1 / ai_l2  → memories_ai, level=1 or 2
    - user_l1 / user_l2 → memories_user, level=1 or 2
    - relationship → memories_ai (生活, 交互) 共同经历 (Phase 2 关系记忆)
    - ai_schedule / greeting → 无记忆 (返回空)

    spec §3.2 + §4.2: 抽中 source 后, **先按 topic_theme 做 LLM rerank**, 再
    输出. 失败回退到 importance 倒排兜底, 保留原行为.

    Returns (texts, memory_ids) for tracking which memories were used.
    """
    # spec §4.1/§4.2: 非记忆来源直接返回空 (打招呼 / 作息走 prompt 模板自身)
    if source in ("ai_schedule", "greeting", None):
        return [], []

    # Resolve (owner, level) from source
    level: int | None
    sub_category: str | None = None
    if source == "ai_l1":
        owner, level = "ai", 1
    elif source == "ai_l2":
        owner, level = "ai", 2
    elif source == "user_l1":
        owner, level = "user", 1
    elif source == "user_l2":
        owner, level = "user", 2
    elif source == "relationship":
        # 我们之间的共同经历 — 不按层级筛 (交互 milestones 落 L1, 但历史数据
        # 可能散在其他层), 按子类筛.
        owner, level, sub_category = "ai", None, "交互"
    else:
        return [], []

    where: dict = {
        "userId": user_id,
        "workspaceId": workspace_id,
        "isArchived": False,
    }
    if level is not None:
        where["level"] = level
    if sub_category is not None:
        where["subCategory"] = sub_category
    rows = await memory_repo.find_many(
        source=owner,  # type: ignore[arg-type]
        where=where,
        order={"importance": "desc"},
        take=30,
    )

    exclude = exclude_memory_ids or set()
    eligible = [r for r in rows if r.id not in exclude and r.content]

    # spec §3.2 + §4.2: topic-aware rerank, 失败回退原 importance 顺序
    rerank_ids = await _rerank_memories_by_topic(eligible, topic_theme or "")
    if rerank_ids:
        order = {mid: idx for idx, mid in enumerate(rerank_ids)}
        ordered = sorted(
            (r for r in eligible if r.id in order),
            key=lambda r: order[r.id],
        )
    else:
        ordered = eligible

    now = datetime.now(UTC)
    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for row in ordered:
        text = row.content
        if not text or text in seen:
            continue
        seen.add(text)
        label = f"[{row.mainCategory or '生活'}/{row.subCategory or '其他'}]"
        # 用户记忆带上"几天前聊到" —— 真人想起朋友说过的事会带时间感;
        # AI 人设记忆是建号时生成的, 没有"聊到"的时间, 不标。
        hint = _user_memory_time_hint(row, now) if owner == "user" else ""
        texts.append(f"{label}{hint} {text}")
        ids.append(row.id)
    return texts[:6], ids[:6]


def _aware(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def relative_day_text(ts: datetime, now: datetime) -> str:
    """按 UTC+8 自然日算的粗粒度相对时间: 今天/昨天/前天/N天前/上周/N周前."""
    days = (now.astimezone(_TZ).date() - ts.astimezone(_TZ).date()).days
    if days <= 0:
        return "今天"
    if days == 1:
        return "昨天"
    if days == 2:
        return "前天"
    if days < 7:
        return f"{days}天前"
    if days < 14:
        return "上周"
    return f"{days // 7}周前"


def _user_memory_time_hint(row: Any, now: datetime) -> str:
    if getattr(row, "mainCategory", None) not in _TIME_HINT_CATEGORIES:
        return ""
    said_at = _aware(getattr(row, "statementTime", None) or getattr(row, "createdAt", None))
    if said_at is None or now - said_at > USER_MEMORY_TIME_HINT_MAX_AGE:
        return ""
    return f"[{relative_day_text(said_at, now)}聊到]"


async def _load_recent_dialogue(workspace_id: str, *, now: datetime | None = None) -> str:
    """近两天带时间戳的聊天记录 (提示词3 的输入). 用户一句都没说过就当没有."""
    from app.services.chat_media.prompt import render_message_content_for_prompt
    from app.services.interaction.topic_continuity import is_dialogue_noise, render_dialogue

    since = (now or datetime.now(UTC)) - RECENT_DIALOGUE_WINDOW
    try:
        rows = await db.query_raw(
            """
            SELECT m.role, m.content, m.metadata, m.created_at
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.workspace_id = $1
              AND c.is_deleted = FALSE
              AND m.role IN ('user', 'assistant')
              AND m.created_at >= $2::timestamp
            ORDER BY m.created_at DESC
            LIMIT $3
            """,
            workspace_id,
            since.astimezone(UTC).replace(tzinfo=None).isoformat(),
            RECENT_DIALOGUE_LIMIT,
        )
    except Exception as e:
        logger.warning(f"[PROACTIVE] recent dialogue load failed ws={workspace_id[:8]}: {e}")
        return ""
    entries = []
    for row in reversed(rows or []):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if is_dialogue_noise(metadata):
            continue
        text = render_message_content_for_prompt(str(row.get("content") or ""), metadata).strip()
        if text:
            entries.append((str(row.get("role")), text, _aware(_parse_ts(row.get("created_at")))))
    if not any(role == "user" for role, _, _ in entries):
        return ""
    return render_dialogue(entries, max_chars=80)


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _build_scene_hint(trigger_type: str, schedule_status: dict[str, Any]) -> str:
    activity = str(schedule_status.get("activity") or "自由时间")
    status = str(schedule_status.get("status") or "idle")
    if trigger_type == "scheduled_scene":
        return f"你当前处于{activity}（状态：{status}），适合从此刻生活情景自然发起聊天。"
    if trigger_type == "memory_proactive":
        return "优先从用户过往记忆里选一个具体点切入，不要泛泛问候。"
    return "优先用轻量、低打扰的方式重新建立联系。"
