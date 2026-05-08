"""特殊日期主动交流 (spec Part 4 §10 + Part 5 §5).

扫描当日是否命中以下任一条件, 并在「起床后第一个空闲时段」触发合并消息:
- 春节 (农历正月初一)
- 元旦 (公历 1-1)
- 用户生日 (从用户记忆 L1 身份:生日 抽取)
- AI 生日 / AI 重要日期 (A 库同类结构)

⚠️ 提醒类 (生活:提醒) 自 Phase 4 起**不再走本路径**——pipeline 录入提醒时
直接建 timetrigger.actionType="reminder" 按精确 occur_time 触发, 由
`triggers._handle_reminder_trigger` 处理 (含 pre-check / 续期 / 隐式完成).
旧 `_extract_reminders_for_date` 已删除. proactive.special_reminder prompt
保留供其他场景或回滚使用. 详见 CLAUDE.md §6 偏离表.

同日多个命中 → 一条合并消息, 使用 proactive.special_combined;
单一命中 → 节日类 / 生日类 两个独立 prompt;
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
from app.services.prompting.utils import render_template
from app.services.proactive.emit import emit_proactive_message
from app.services.proactive.history import (
    can_send_proactive,
    increment_proactive_count,
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

OccasionType = Literal["holiday", "birthday", "reminder", "important_date"]


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

    # 找起床事件（兼容新 schema `event` + 旧 schema `activity`/`type`）
    wake_end: str | None = None
    for slot in schedule:
        event = str(slot.get("event") or slot.get("activity") or "")
        if "起床" in event or slot.get("type") == "wake":
            wake_end = str(slot.get("end") or "")
            if wake_end:
                break

    if not wake_end:
        return fallback

    # 从起床结束时刻开始, 找第一个 status == idle 的时段
    for slot in schedule:
        start = str(slot.get("start") or "")
        if start < wake_end:
            continue
        # Spec §2.1: status ∈ {idle, busy, very_busy, sleep}
        if slot.get("status") == "idle":
            h, m = start.split(":")
            return datetime(the_date.year, the_date.month, the_date.day, int(h), int(m), tzinfo=_TZ)

    # 没找到明确 idle slot → 就用起床结束时刻作为触发点
    h, m = wake_end.split(":")
    return datetime(the_date.year, the_date.month, the_date.day, int(h), int(m), tzinfo=_TZ)


# ── 特殊日期收集 ──

_BIRTHDAY_RE = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")
_BIRTHDAY_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")


async def _extract_birthday_from_memories(user_id: str, owner: str) -> tuple[int, int] | None:
    """从 owner(user/ai) 的 L1 身份记忆中提取生日 (month, day).

    三层优先级 (spec §4.1 + Part 5 §6.1: parser event_time → occur_time):
    1. row.occurTime (parser 抽取出的权威源)
    2. content/summary 中文格式 "3月20日/号"
    3. content/summary ISO 格式 "1995-03-20" / "2026/3/20"
    """
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
            occur = getattr(row, "occurTime", None)
            if occur is not None:
                try:
                    return occur.month, occur.day
                except Exception:
                    pass
            text = (row.content or "") + " " + (row.summary or "")
            m = _BIRTHDAY_RE.search(text)
            if m:
                return int(m.group(1)), int(m.group(2))
            m = _BIRTHDAY_ISO_RE.search(text)
            if m:
                return int(m.group(2)), int(m.group(3))
    except Exception as e:
        logger.debug(f"Birthday extraction ({owner}) failed for user={user_id}: {e}")
    return None


# Phase 4.2: removed `_reminder_matches_date` + `_extract_reminders_for_date`.
# 提醒不再走本日扫描——pipeline 录入即建 timetrigger.actionType="reminder",
# triggers._handle_reminder_trigger 直接按 occur_time 精确调度.


async def _extract_important_dates_for_date(
    user_id: str,
    owner: str,
    the_date: date,
) -> list[str]:
    """Part 5 §4.1 "用户/AI 重要日期" 子类: 直接 SELECT subCategory='重要日期'
    AND occurTime.date()==today.

    spec §5.1 不把"重要日期"列为独立主动触发场景, 但 §4.1 要求日历库记录,
    用作合并消息时的话题素材 (例如生日恰逢面试日, 一句话带上).

    历史从 `生活/非提醒/有 occur_time` 反向推, 不可靠 (依赖 LLM 抽取时碰巧
    填了 occur_time) — 改为正向查询子类, 配合 extraction prompt 把"日历事件"
    显式归到"重要日期"子类.
    """
    try:
        rows = await memory_repo.find_many(
            source=owner,  # type: ignore[arg-type]
            where={
                "userId": user_id,
                "mainCategory": "生活",
                "subCategory": "重要日期",
                "isArchived": False,
                "occurTime": {"not": None},
            },
            take=50,
        )
        contents: list[str] = []
        for row in rows:
            # SQL `WHERE occurTime IS NOT NULL` 已过滤; Python `if occur is None`
            # 仅防 mock/repo drift (test 直接构 SimpleNamespace 时绕开 WHERE).
            occur = getattr(row, "occurTime", None)
            if occur is None:
                continue
            if occur.date() == the_date:
                text = (row.summary or row.content or "").strip()
                if text:
                    contents.append(text[:60])
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

    spec §5.1 主动触发场景剩 3 类 (提醒已迁移到 timetrigger 直接调度, Phase 4.2):
    - 公共节假日 (春节/元旦)
    - 用户生日 / AI 生日

    Part 5 §4.1 日历库还登记"重要日期"(纪念/考试/面试)但不独立触发.
    若同日有主动触发命中 + 重要日期, 合并消息会带上重要日期素材.
    """
    import asyncio
    d = the_date or datetime.now(_TZ).date()
    occasions: list[Occasion] = []

    # 公共节假日 (sync, 不进 gather)
    try:
        from app.services.schedule_domain.time_service import is_holiday
        info = is_holiday(d)
        if info and info.name in ("春节", "元旦"):
            occasions.append(Occasion(type="holiday", name=info.name, owner="user"))
    except Exception as e:
        logger.debug(f"Holiday check failed: {e}")

    # 2 个独立 I/O (生日 user/ai) 并发拉取
    ub, ab = await asyncio.gather(
        _extract_birthday_from_memories(user_id, "user"),
        _extract_birthday_from_memories(user_id, "ai"),
    )

    if ub and ub == (d.month, d.day):
        occasions.append(Occasion(type="birthday", name="用户生日", owner="user"))
    if ab and ab == (d.month, d.day):
        occasions.append(Occasion(type="birthday", name="AI生日", owner="ai"))

    # spec §5.1 重要日期不独立触发 — 但若已有其他 4 类命中, 把当日重要日期作为
    # 素材附加. 独立 type="important_date" 防回归: 若以后去掉 `if occasions` gate,
    # important_date 也不会被 _pick_prompt_key_and_fields 误归到 reminder 单一触发分支.
    if occasions:
        user_important, ai_important = await asyncio.gather(
            _extract_important_dates_for_date(user_id, "user", d),
            _extract_important_dates_for_date(user_id, "ai", d),
        )
        for text in user_important:
            occasions.append(Occasion(type="important_date", name=f"今日重要事项: {text}", owner="user"))
        for text in ai_important:
            occasions.append(Occasion(type="important_date", name=f"今日(我的)重要事项: {text}", owner="ai"))

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
        # Spec 指令模版 P28 节日类无【参考信息】，仅需 holiday_name 占位符
        return ("proactive.special_holiday", {"holiday_name": o.name})
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
    return ("proactive.special_holiday", {"holiday_name": o.name})


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

    # spec §10.2 衰减最终停止检查. stop_proactive_state 把永久停止 state 写
    # status='idle' + stop_reason='silence_exhausted' (无 stopped_permanent 状态值),
    # 必须按 stop_reason 字段判别.
    state = await ensure_proactive_state_for_workspace(
        workspace_id, now=now_ts, reason="special_date",
    )
    if state and state.stop_reason == "silence_exhausted":
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
        prompt = render_template(
            tpl,
            fields,
            optional_keys={"user_portrait"},
            safe=False,
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Special date prompt format failed key={prompt_key}: {e}")
        return False

    from app.services.llm.usage_tracker import traced_usage_session
    async with traced_usage_session(
        name="[proactive:special_date]",
        scope="proactive", conversation_id=conversation_id,
        agent_id=agent_id, user_id=user_id,
    ) as tracer:
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
            trace_id=tracer.safe_trace_id,
        )
        await increment_proactive_count(agent_id, user_id)
        return True


# ── 每日扫描入口 ──

async def _scan_one_agent(agent, today: date, day_start: datetime, day_end: datetime) -> None:
    """单个 agent 的特殊日期扫描. 主流程错误本地处理, 不影响其他 agent."""
    try:
        occasions = await collect_special_dates_today(user_id=agent.userId, the_date=today)
        if not occasions:
            return

        schedule = await get_cached_schedule(agent.id)
        trigger_time = find_first_idle_after_wakeup(schedule, today)

        # 幂等: 同日同 agent 已建则跳
        existing = await db.timetrigger.find_first(
            where={
                "aiAgentId": agent.id,
                "userId": agent.userId,
                "actionType": "special_date",
                "triggerTime": {"gte": day_start, "lt": day_end},
            }
        )
        if existing:
            return

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


async def scan_special_dates_today(now: datetime | None = None, concurrency: int = 8) -> None:
    """spec Part 5 §4.3: 每日凌晨扫所有 active agent 的特殊日期.

    - 命中 → 计算「起床后第一个空闲时段」作为触发时刻
    - 写入 timetrigger (actionType=special_date, actionData=occasions 列表)
    - 到点由 scan_triggers 捞出并实际发送

    并发上限 = `concurrency` (默认 8) — 跟主动消息生成模式一致, 避免 N=1000+
    agent 时 6+ 次串行 DB 查询拖垮 daily_schedule 链路总时长.
    """
    import asyncio
    now_ts = now or datetime.now(_TZ)
    today = now_ts.date()
    day_start = datetime(today.year, today.month, today.day, 0, 0, tzinfo=_TZ)
    day_end = day_start + timedelta(days=1)

    agents = await db.aiagent.find_many(where={"status": "active"})
    if not agents:
        return

    sem = asyncio.Semaphore(concurrency)

    async def _bound(agent):
        async with sem:
            await _scan_one_agent(agent, today, day_start, day_end)

    # 用 gather 并发, exception 不会冒上来 (each task 内部已 try/except)
    await asyncio.gather(*(_bound(a) for a in agents))
