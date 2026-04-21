"""Yesterday-life summary for AI self-memory generation.

Per spec《产品手册·背景信息》§2.2:
    每日凌晨，系统汇总以下三类前一日数据，调用大模型生成一段第一人称的
    「昨日生活总结」（200 字以内）：
      1. 前一日预期作息表
      2. 前一日调整日志：用户要求 AI 调整作息的具体变动
      3. 前一日主动日志：AI 主动向用户发起对话的详情
                       （时间 / 触发原因 / 内容 / 是否被回复）

This summary is the input for §2.3 (memory split & scoring), so the
output is intentionally a single short paragraph in the AI's voice —
downstream `generate_daily_self_memories` will slice it into individual
memory rows.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.llm.models import get_utility_model, invoke_text

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = """你是 AI 角色「{ai_name}」，请用第一人称回顾昨天的生活，
写一段 150-200 字的自然口语化总结。

### 昨日预期作息表
{schedule}

### 用户调整记录（如有）
{adjusts}

### 主动发起对话记录（如有）
{proactives}

### 要求
- 第一人称（"我"）
- 150-200 字
- 自然口语，不要生硬罗列时间表
- 重点提及与作息差异、与用户的互动
- 如三类数据都为空，回顾"平静的一天"

直接输出总结文本，不要标题、不要 JSON。"""


def _format_schedule(slots: list[dict] | None) -> str:
    if not slots:
        return "（无作息记录）"
    lines = []
    for slot in slots[:20]:  # 上限 20 段，避免 prompt 膨胀
        start = slot.get("start_time") or slot.get("start") or ""
        end = slot.get("end_time") or slot.get("end") or ""
        # Spec §2.1 新 schema: event + status; 旧 schema: activity + type
        event = slot.get("event") or slot.get("activity") or slot.get("description") or ""
        status = slot.get("status") or slot.get("type") or ""
        lines.append(f"- {start}-{end} {event}（{status}）")
    return "\n".join(lines)


def _format_adjusts(rows: list) -> str:
    if not rows:
        return "（无）"
    lines = []
    for r in rows[:10]:
        lines.append(
            f"- 类型: {r.adjustType} | "
            f"原: {r.oldValue or '-'} → 新: {r.newValue or '-'} | "
            f"原因: {r.reason or '未说明'}"
        )
    return "\n".join(lines)


def _format_proactives(rows: list) -> str:
    if not rows:
        return "（无）"
    lines = []
    for r in rows[:10]:
        when = r.createdAt.strftime("%H:%M") if r.createdAt else "?"
        trigger = r.eventType or "未知触发"
        lines.append(f"- {when} [{trigger}] {r.message[:80]}")
    return "\n".join(lines)


async def build_yesterday_summary(
    *, agent_id: str, agent_name: str, user_id: str,
) -> str:
    """Aggregate the 3 sources for `agent` from yesterday and have the LLM
    produce a 150-200 word first-person summary. Returns "" on any failure
    so the caller can fall back to "no dialogue today" path.
    """
    now = datetime.now(UTC)
    yesterday = (now - timedelta(days=1)).date()
    day_start = datetime.combine(yesterday, datetime.min.time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)

    # ── 1) 前一日预期作息表 ──
    schedule_row = await db.aidailyschedule.find_first(
        where={"agentId": agent_id, "date": day_start},
    )
    schedule_slots: list[dict] | None = None
    if schedule_row and isinstance(schedule_row.scheduleData, list):
        schedule_slots = list(schedule_row.scheduleData)

    # ── 2) 前一日调整日志 ──
    adjusts = await db.scheduleadjustlog.find_many(
        where={
            "agentId": agent_id,
            "createdAt": {"gte": day_start, "lt": day_end},
        },
        order={"createdAt": "asc"},
    )

    # ── 3) 前一日主动日志 ──
    proactives = await db.proactivechatlog.find_many(
        where={
            "agentId": agent_id,
            "userId": user_id,
            "createdAt": {"gte": day_start, "lt": day_end},
        },
        order={"createdAt": "asc"},
    )

    # 三类全空 → 直接返回空字符串，让上游走"平静的一天"路径或跳过
    if not schedule_slots and not adjusts and not proactives:
        return ""

    prompt = _SUMMARY_PROMPT.format(
        ai_name=agent_name,
        schedule=_format_schedule(schedule_slots),
        adjusts=_format_adjusts(adjusts),
        proactives=_format_proactives(proactives),
    )

    try:
        text = await invoke_text(get_utility_model(), prompt)
        return text.strip()
    except Exception as e:
        logger.warning(f"Yesterday summary generation failed for {agent_id}: {e}")
        return ""
