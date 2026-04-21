"""特殊日期主动交流 (spec Part 4 §10 + Part 5 §5).

扫描当日是否命中以下任一条件, 并在「起床后第一个空闲时段」触发合并消息:
- 春节 (农历正月初一)
- 元旦 (公历 1-1)
- 用户生日 (从用户记忆 L1 身份:生日 抽取)
- 用户提醒日期 (生活:提醒 子类, occur_time 落在当日)
- AI 生日 / AI 重要日期 / AI 提醒日期 (A 库同类结构)

同日多个命中 → 一条合并消息, 使用 proactive.special_combined;
单一命中 → 节日类 / 生日类 / 提醒类 三个独立 prompt;
所有特殊日期消息带 metadata.skip_post_process=True, 走独立发送路径.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from app.db import db
from app.services.llm.models import get_chat_model, invoke_text
from app.services.memory.storage import repo as memory_repo
from app.services.prompting.store import get_prompt_text
from app.services.proactive.emit import emit_proactive_message
from app.services.proactive.history import (
    can_send_proactive,
    increment_proactive_count,
    increment_proactive_2day_count,
)
from app.services.proactive.sender import _build_personality_brief  # type: ignore[import-private]
from app.services.proactive.state import (
    get_active_workspace_context,
    ensure_proactive_state_for_workspace,
)
from app.services.schedule_domain.schedule import get_cached_schedule
from app.services.schedule_domain.time_service import _TZ
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

UTC = timezone.utc

OccasionType = Literal["holiday", "birthday", "reminder"]


@dataclass(frozen=True)
class Occasion:
    type: OccasionType
    name: str  # 节日名称 / "生日" / 提醒事项内容
    owner: str  # "user" or "ai"


# ── 起床后第一个空闲时段 ──

def find_first_idle_after_wakeup(
    schedule: list[dict] | None,
    the_date: date,
) -> datetime:
    """spec §10.2 + Part 5 §5.2: 找到作息表中 '起床' 事件结束后的第一个空闲时段开始时刻.

    找不到时 fallback 到 08:00 (对齐原硬编码兜底).
    """
    fallback = datetime(the_date.year, the_date.month, the_date.day, 8, 0, tzinfo=_TZ)
    if not schedule:
        return fallback

    # 找起床事件
    wake_end: str | None = None
    for slot in schedule:
        activity = str(slot.get("activity") or "")
        if "起床" in activity or slot.get("type") == "wake":
            wake_end = str(slot.get("end") or "")
            if wake_end:
                break

    if not wake_end:
        return fallback

    # 从起床结束时刻开始, 找第一个 status != busy/very_busy/sleep 的时段
    for slot in schedule:
        start = str(slot.get("start") or "")
        if start < wake_end:
            continue
        slot_type = slot.get("type") or "leisure"
        if slot_type in ("leisure",) or slot.get("status") == "idle":
            h, m = start.split(":")
            return datetime(the_date.year, the_date.month, the_date.day, int(h), int(m), tzinfo=_TZ)

    # 没找到明确 idle slot → 就用起床结束时刻作为触发点
    h, m = wake_end.split(":")
    return datetime(the_date.year, the_date.month, the_date.day, int(h), int(m), tzinfo=_TZ)


# ── 特殊日期收集 ──

_BIRTHDAY_RE = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")


async def _extract_birthday_from_memories(user_id: str, owner: str) -> tuple[int, int] | None:
    """从 owner(user/ai) 的 L1 身份记忆中提取生日 (month, day)."""
    try:
        rows = await memory_repo.find_many(
            source=owner,  # type: ignore[arg-type]
            where={
                "userId": user_id,
                "mainCategory": "身份",
                "subCategory": "生日",
                "isArchived": False,
                "level": 1,
            },
            take=3,
        )
        for row in rows:
            text = (row.content or "") + " " + (row.summary or "")
            m = _BIRTHDAY_RE.search(text)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception as e:
        logger.debug(f"Birthday extraction ({owner}) failed for user={user_id}: {e}")
    return None


async def _extract_reminders_for_date(
    user_id: str,
    owner: str,
    the_date: date,
) -> list[str]:
    """从 owner 的 生活-提醒 子类中抽取 occur_time 落在当日的提醒.

    优先级: occur_time 字段精确匹配 → fallback 到 content 里的日期串粗匹配.
    """
    try:
        rows = await memory_repo.find_many(
            source=owner,  # type: ignore[arg-type]
            where={
                "userId": user_id,
                "mainCategory": "生活",
                "subCategory": "提醒",
                "isArchived": False,
            },
            take=50,
        )
        contents: list[str] = []
        date_hints = [
            f"{the_date.month}月{the_date.day}日",
            f"{the_date.month}月{the_date.day}号",
            f"{the_date.year}年{the_date.month}月{the_date.day}",
            the_date.isoformat(),
        ]
        for row in rows:
            text = (row.summary or row.content or "").strip()
            if not text:
                continue
            # 优先: occur_time 精确匹配当日 (spec §6.1 落库映射: parser event_time → occur_time)
            occur = getattr(row, "occurTime", None)
            if occur is not None:
                try:
                    if occur.date() == the_date:
                        contents.append(text[:60])
                        continue
                except Exception:
                    pass
            # Fallback: content 里的日期串粗匹配
            if any(hint in text for hint in date_hints):
                contents.append(text[:60])
        return contents
    except Exception as e:
        logger.debug(f"Reminder extraction ({owner}) failed user={user_id}: {e}")
        return []


async def _extract_important_dates_for_date(
    user_id: str,
    owner: str,
    the_date: date,
) -> list[str]:
    """Part 5 §4.1 "用户/AI 重要日期" 子类:
    生活记忆中 occur_time 命中当日的非提醒条目 (纪念/考试/面试/体检等).
    spec §5.1 不把"重要日期"列为主动触发场景, 但 §4.1 要求日历库记录,
    用作合并消息时的话题素材 (例如生日恰逢面试日, 一句话带上).
    """
    try:
        rows = await memory_repo.find_many(
            source=owner,  # type: ignore[arg-type]
            where={
                "userId": user_id,
                "mainCategory": "生活",
                "isArchived": False,
                "occurTime": {"not": None},
            },
            take=50,
        )
        contents: list[str] = []
        for row in rows:
            if row.subCategory == "提醒":
                continue  # 提醒走 _extract_reminders_for_date
            occur = getattr(row, "occurTime", None)
            if occur is None:
                continue
            try:
                if occur.date() == the_date:
                    text = (row.summary or row.content or "").strip()
                    if text:
                        contents.append(text[:60])
            except Exception:
                continue
        return contents
    except Exception as e:
        logger.debug(f"Important date extraction ({owner}) failed user={user_id}: {e}")
        return []


async def collect_special_dates_today(
    *,
    user_id: str,
    the_date: date | None = None,
) -> list[Occasion]:
    """spec Part 5 §4.3: 凌晨扫当日特殊日期, 命中则返回 Occasion 列表.

    spec §5.1 主动触发场景 (4 类):
    - 公共节假日 (春节/元旦)
    - 用户生日 / AI 生日
    - 用户提醒 / AI 提醒 (occur_time 落在当日)

    Part 5 §4.1 日历库还登记"重要日期"(纪念/考试/面试)但不独立触发.
    若同日有主动触发命中 + 重要日期, 合并消息会带上重要日期素材.
    """
    d = the_date or datetime.now(_TZ).date()
    occasions: list[Occasion] = []

    # 公共节假日: 只主动触发春节 / 元旦
    try:
        from app.services.schedule_domain.time_service import is_holiday

        info = is_holiday(d)
        if info and info.name in ("春节", "元旦"):
            occasions.append(Occasion(type="holiday", name=info.name, owner="user"))
    except Exception as e:
        logger.debug(f"Holiday check failed: {e}")

    # 用户生日
    ub = await _extract_birthday_from_memories(user_id, "user")
    if ub and ub == (d.month, d.day):
        occasions.append(Occasion(type="birthday", name="用户生日", owner="user"))

    # AI 生日
    ab = await _extract_birthday_from_memories(user_id, "ai")
    if ab and ab == (d.month, d.day):
        occasions.append(Occasion(type="birthday", name="AI生日", owner="ai"))

    # 用户提醒
    for text in await _extract_reminders_for_date(user_id, "user", d):
        occasions.append(Occasion(type="reminder", name=text, owner="user"))
    # AI 提醒
    for text in await _extract_reminders_for_date(user_id, "ai", d):
        occasions.append(Occasion(type="reminder", name=text, owner="ai"))

    # spec §5.1 重要日期不独立触发 — 但若已有其他 4 类命中, 把当日重要日期作为
    # 素材附加 (空列表则跳过, 不影响触发判定).
    if occasions:
        for text in await _extract_important_dates_for_date(user_id, "user", d):
            occasions.append(Occasion(type="reminder", name=f"今日重要事项: {text}", owner="user"))
        for text in await _extract_important_dates_for_date(user_id, "ai", d):
            occasions.append(Occasion(type="reminder", name=f"今日(我的)重要事项: {text}", owner="ai"))

    return occasions


# ── 发送 ──

async def _pick_prompt_key_and_fields(
    occasions: list[Occasion],
    personality_brief: str,
) -> tuple[str, dict[str, Any]]:
    """spec §10.3: 单一命中用类型 prompt, 多命中用合并类."""
    if len(occasions) >= 2:
        return (
            "proactive.special_combined",
            {
                "personality_brief": personality_brief,
                "user_portrait": "",  # 由调用方填充
                "occasions": ", ".join(f"{o.type}:{o.name}" for o in occasions),
            },
        )

    o = occasions[0]
    if o.type == "holiday":
        return (
            "proactive.special_holiday",
            {"personality_brief": personality_brief, "holiday_name": o.name},
        )
    if o.type == "birthday":
        return (
            "proactive.special_birthday",
            {"personality_brief": personality_brief},
        )
    if o.type == "reminder":
        return (
            "proactive.special_reminder",
            {
                "personality_brief": personality_brief,
                "reminder_content": o.name,
            },
        )
    return ("proactive.special_holiday", {"personality_brief": personality_brief, "holiday_name": o.name})


async def send_special_date_proactive(
    *,
    agent_id: str,
    user_id: str,
    occasions: list[Occasion],
    now: datetime | None = None,
) -> bool:
    """spec §10 特殊日期主动消息.

    - 不经过时间窗概率
    - 计入每日 3 次上限
    - 消息 metadata skip_post_process=True
    - 衰减 final stop 状态下不触发
    """
    if not occasions:
        return False

    now_ts = now or datetime.now(UTC)

    # 每日上限: 特殊日期也计入 (spec §10.2)
    if not await can_send_proactive(agent_id, user_id):
        logger.debug(f"Special date skipped: daily limit agent={agent_id[:8]}")
        return False

    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    if not workspace_id:
        return False

    # 衰减最终停止检查: 通过 ensure_proactive_state_for_workspace + 读 status
    state = await ensure_proactive_state_for_workspace(
        workspace_id, now=now_ts, reason="special_date",
    )
    if state and state.status == "stopped_permanent":
        logger.debug(f"Special date skipped: decay final stop workspace={workspace_id[:8]}")
        return False

    workspace_context = await get_active_workspace_context(workspace_id)
    if not workspace_context:
        return False
    conversation_id = str(workspace_context.get("conversation_id") or "")
    if not conversation_id:
        return False

    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    personality_brief = _build_personality_brief(agent)
    prompt_key, fields = await _pick_prompt_key_and_fields(occasions, personality_brief)
    try:
        tpl = await get_prompt_text(prompt_key)
        prompt = tpl.format(**fields)
    except (KeyError, ValueError) as e:
        logger.warning(f"Special date prompt format failed key={prompt_key}: {e}")
        return False

    model = get_chat_model()
    message = (await invoke_text(model, prompt)).strip()
    if not message or len(message) < 4:
        return False

    await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=message,
        trigger_type="special_date",
        skip_post_process=True,  # spec §10.4 不经过回复加工
        extra_metadata={
            "occasions": [
                {"type": o.type, "name": o.name, "owner": o.owner}
                for o in occasions
            ],
        },
        ws_payload_extra={"occasions": [o.type for o in occasions]},
    )
    await increment_proactive_count(agent_id, user_id)
    await increment_proactive_2day_count(agent_id, user_id)
    return True


# ── 每日扫描入口 ──

async def scan_special_dates_today(now: datetime | None = None) -> None:
    """spec Part 5 §4.3: 每日凌晨扫所有 active agent 的特殊日期.

    - 命中 → 计算「起床后第一个空闲时段」作为触发时刻
    - 写入 timetrigger (actionType=special_date, actionData=occasions 列表)
    - 到点由 scan_triggers 捞出并实际发送
    """
    now_ts = now or datetime.now(_TZ)
    today = now_ts.date()

    agents = await db.aiagent.find_many(where={"status": "active"})
    for agent in agents:
        try:
            occasions = await collect_special_dates_today(user_id=agent.userId, the_date=today)
            if not occasions:
                continue

            schedule = await get_cached_schedule(agent.id)
            trigger_time = find_first_idle_after_wakeup(schedule, today)

            # 幂等: 若同日同 agent 已创建, 跳过
            day_start = datetime(today.year, today.month, today.day, 0, 0, tzinfo=_TZ)
            day_end = day_start + timedelta(days=1)
            existing = await db.timetrigger.find_first(
                where={
                    "aiAgentId": agent.id,
                    "userId": agent.userId,
                    "actionType": "special_date",
                    "triggerTime": {"gte": day_start, "lt": day_end},
                }
            )
            if existing:
                continue

            await db.timetrigger.create(data={
                "agent": {"connect": {"id": agent.id}},
                "user": {"connect": {"id": agent.userId}},
                "triggerTime": trigger_time,
                "actionType": "special_date",
                "actionData": {
                    "occasions": [
                        {"type": o.type, "name": o.name, "owner": o.owner}
                        for o in occasions
                    ],
                    "scheduled_reason": "first_idle_after_wakeup",
                },
            })
            logger.info(
                f"Special date trigger created agent={agent.id} "
                f"date={today} occasions={[o.type for o in occasions]} at={trigger_time}"
            )
        except Exception as e:
            logger.warning(f"scan_special_dates failed for agent {agent.id}: {e}")
