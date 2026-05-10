"""主动聊天上下文构建。"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.portrait import get_latest_portrait
from app.services.prompting.utils import render_prompt
from app.services.relationship.intimacy import get_relationship_stage, get_topic_intimacy
from app.services.memory.storage import repo as memory_repo
from app.services.memory.core_memory import load_core_memory_strings
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)
UTC = timezone.utc


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
) -> dict[str, Any]:
    # 9 个独立 I/O 并发 (DB / Redis / LLM rerank). _load_proactive_memories
    # 含 utility LLM 调用是最长尾, 跟其余 DB 读并行可让 LLM 时间被吸收.
    (
        agent, schedule, core_memories,
        proactive_memories_pair, topic_intimacy, silence_hours,
        user_portrait, recent_context,
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
        _compute_silence_hours(workspace_id),
        get_latest_portrait(user_id, agent_id),
        _load_recent_context(workspace_id),
    )
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    proactive_memories, used_memory_ids = proactive_memories_pair
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
        "relationship_stage": relationship_stage,
        "topic_intimacy": topic_intimacy,
        "silence_hours": silence_hours,
        "scene_hint": scene_hint,
        "trigger_type": trigger_type,
        "stage": stage,
        "source": source or "greeting",
        "topic_theme": topic_theme or "",
        "user_portrait": user_portrait or "",
        "recent_context": recent_context,
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
    Caller (`_load_proactive_memories`) 已过滤空 summary/content 行,
    rows 进来都有内容.
    """
    if not topic_theme or not rows:
        return []
    candidates = [{"id": r.id, "text": (r.summary or r.content)[:80]} for r in rows]
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
    - ai_schedule / greeting → 无记忆 (返回空)

    spec §3.2 + §4.2: 抽中 source 后, **先按 topic_theme 做 LLM rerank**, 再
    输出. 失败回退到 importance 倒排兜底, 保留原行为.

    Returns (texts, memory_ids) for tracking which memories were used.
    """
    # spec §4.1/§4.2: 非记忆来源直接返回空 (打招呼 / 作息走 prompt 模板自身)
    if source in ("ai_schedule", "greeting", None):
        return [], []

    # Resolve (owner, level) from source
    if source == "ai_l1":
        owner, level = "ai", 1
    elif source == "ai_l2":
        owner, level = "ai", 2
    elif source == "user_l1":
        owner, level = "user", 1
    elif source == "user_l2":
        owner, level = "user", 2
    else:
        return [], []

    rows = await memory_repo.find_many(
        source=owner,  # type: ignore[arg-type]
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "isArchived": False,
            "level": level,
        },
        order={"importance": "desc"},
        take=30,
    )

    exclude = exclude_memory_ids or set()
    eligible = [r for r in rows if r.id not in exclude and (r.summary or r.content)]

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

    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for row in ordered:
        text = row.summary or row.content
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(f"[{row.mainCategory or '生活'}/{row.subCategory or '其他'}] {text}")
        ids.append(row.id)
    return texts[:6], ids[:6]


def _build_scene_hint(trigger_type: str, schedule_status: dict[str, Any]) -> str:
    activity = str(schedule_status.get("activity") or "自由时间")
    status = str(schedule_status.get("status") or "idle")
    if trigger_type == "scheduled_scene":
        return f"你当前处于{activity}（状态：{status}），适合从此刻生活情景自然发起聊天。"
    if trigger_type == "memory_proactive":
        return "优先从用户过往记忆里选一个具体点切入，不要泛泛问候。"
    return "优先用轻量、低打扰的方式重新建立联系。"


async def _compute_silence_hours(workspace_id: str) -> float:
    rows = await db.query_raw(
        """
        SELECT m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.workspace_id = $1
          AND c.is_deleted = FALSE
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        workspace_id,
    )
    if not rows:
        return 999.0
    last_created = rows[0].get("created_at")
    if not isinstance(last_created, datetime):
        try:
            last_created = datetime.fromisoformat(str(last_created))
        except ValueError:
            return 999.0
    if last_created.tzinfo is None:
        last_created = last_created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - last_created.astimezone(UTC)).total_seconds() / 3600.0)
