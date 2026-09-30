"""话题完结判定 + B 模式追问 (《主动聊天机制（新增）》).

AI 说完最后一句满 5 分钟用户没回, 判定窗 (state window 0) 到期后进入本模块:

  1. 读最近对话。最后一条已经不是 AI 的普通回复 (用户其实回了 / 刚发过提醒等
     主动消息) → 不判定, 直接进 A 模式窗口。
  2. 判定话题是否完结: 告别按规则判完结; 危机照护 / 系统确认类问句不判; 其余
     交给小模型 (proactive.topic_completion_judge, 输出 已完结 / 未完结)。
  3. 结论写入 topic_continuity, 供聊天侧做被动承接。
  4. 未完结 + 本会话没用过 B + 过门槛 → 生成并发送 B 追问, A 模式窗口从这句
     重新起算; 其余一律进入 A 模式 (window 1, t0 仍是 AI 最后一句的时刻)。

B 只追问一次, 之后本会话不论话题状态都走 A —— 单会话一次的记录在
topic_continuity, 跨越用户回复存活。判定在冷却期 / 夜间照样跑 (便宜的小模型),
因为被动承接要用它的结论; 门槛只拦 B 的发送。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.observability.events import (
    EVT_PROACTIVE_FOLLOWUP_SENT,
    EVT_PROACTIVE_FOLLOWUP_SKIPPED,
    EVT_PROACTIVE_TOPIC_JUDGED,
)
from app.services.interaction import topic_continuity
from app.services.interaction.topic_continuity import (
    FOLLOWUP_TRIGGER_TYPE,
    TopicVerdict,
    clean_single_line,
    is_dialogue_noise,
    render_dialogue,
)
from app.services.llm.models import get_chat_model, get_utility_model, invoke_text
from app.services.proactive.emit import emit_proactive_message
from app.services.proactive.gates import check_followup_gates, check_workspace
from app.services.proactive.state import (
    ARM_REASON_FAREWELL,
    NO_FOLLOWUP_REASONS,
    ProactiveStateRecord,
    advance_to_next_window,
    log_proactive_event,
    stop_proactive_state,
)
from app.services.prompting.utils import render_prompt
from app.services.schedule_domain.time_expression import describe_time_scene, format_clock

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 提示词文档: "前20轮对话上下文" —— 一轮 = 用户一句 + AI 一句
_RECENT_LIMIT = 40
_JUDGE_TIMEOUT_S = 10.0
_GENERATE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class _Turn:
    id: str
    role: str
    text: str
    proactive: bool
    created_at: datetime | None = None
    # 边界 / 危机 / 系统确认类回复: 无论 arm 原因是什么, 这之后都不追问
    no_followup: bool = False


# ────────────────────────────────────────────────────────────────────
# Recent conversation
# ────────────────────────────────────────────────────────────────────

_NO_FOLLOWUP_KIND_PREFIXES = ("crisis", "deletion", "record_request")


def _blocks_followup(metadata: dict[str, Any]) -> bool:
    if metadata.get("boundary"):
        return True
    diagnostics = metadata.get("response_diagnostics")
    kind = str((diagnostics or {}).get("short_circuit_kind") or "") if isinstance(diagnostics, dict) else ""
    return kind.startswith(_NO_FOLLOWUP_KIND_PREFIXES)


def _aware(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


async def _load_recent_turns(conversation_id: str) -> list[_Turn]:
    from app.services.chat_media.prompt import render_message_content_for_prompt

    rows = await db.query_raw(
        """
        SELECT id, role, content, metadata, created_at
        FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        conversation_id,
        _RECENT_LIMIT,
    )
    turns: list[_Turn] = []
    for row in reversed(rows or []):
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if is_dialogue_noise(metadata):
            continue
        text = render_message_content_for_prompt(str(row.get("content") or ""), metadata).strip()
        if not text:
            continue
        turns.append(_Turn(
            id=str(row["id"]),
            role=str(row.get("role") or "user"),
            text=text,
            proactive=bool(metadata.get("proactive")),
            created_at=_aware(row.get("created_at")),
            no_followup=_blocks_followup(metadata),
        ))
    return turns


def format_turns(turns: list[_Turn]) -> str:
    return render_dialogue((t.role, t.text, t.created_at) for t in turns)


def _last_time(turns: list[_Turn], role: str) -> str:
    for t in reversed(turns):
        if t.role == role and t.created_at:
            return format_clock(t.created_at)
    return "（无）"


# ────────────────────────────────────────────────────────────────────
# Topic completion judgement
# ────────────────────────────────────────────────────────────────────

def _parse_verdict(raw: Any) -> TopicVerdict | None:
    """提示词输出 已完结 / 未完结 二选一; 说不清 (含把两个选项照抄一遍) 当作判不出."""
    text = str(raw or "")
    unfinished, finished = "未完结" in text, "已完结" in text
    if unfinished == finished:
        return None
    return TopicVerdict("unfinished" if unfinished else "finished", "llm")


async def judge_topic_completion(
    turns: list[_Turn],
    *,
    arm_reason: str,
    now: datetime,
) -> TopicVerdict | None:
    """None = 不适用 / 判不出 (不写结论, 聊天侧维持原有逻辑)."""
    if arm_reason == ARM_REASON_FAREWELL:
        return TopicVerdict("finished", "farewell")
    if arm_reason in NO_FOLLOWUP_REASONS:
        return None
    try:
        raw = await asyncio.wait_for(
            render_prompt(
                "proactive.topic_completion_judge",
                {
                    "conversation": format_turns(turns),
                    "user_last_time": _last_time(turns, "user"),
                    "current_time": format_clock(now),
                },
                lambda p: invoke_text(get_utility_model(), p),
            ),
            timeout=_JUDGE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.info("[FOLLOWUP] topic judge timed out")
        return None
    return _parse_verdict(raw)


# ────────────────────────────────────────────────────────────────────
# B-mode follow-up message
# ────────────────────────────────────────────────────────────────────

def _clean_followup(text: str | None) -> str | None:
    return clean_single_line(text, min_len=4)


async def generate_followup_message(
    state: ProactiveStateRecord,
    turns: list[_Turn],
    now: datetime,
) -> str | None:
    from app.services.mbti import build_personality_brief

    agent = await db.aiagent.find_unique(where={"id": state.agent_id})
    if not agent:
        return None
    try:
        raw = await asyncio.wait_for(
            render_prompt(
                "proactive.followup_unfinished",
                {
                    "conversation": format_turns(turns),
                    "ai_last_time": _last_time(turns, "assistant"),
                    "current_time": format_clock(now),
                    "time_scene": describe_time_scene(now),
                    "personality_brief": build_personality_brief(agent),
                },
                lambda p: invoke_text(get_chat_model(), p),
                strip_split=False,
            ),
            timeout=_GENERATE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.info("[FOLLOWUP] generation timed out")
        return None
    return _clean_followup(raw)


# ────────────────────────────────────────────────────────────────────
# Judge window entry
# ────────────────────────────────────────────────────────────────────

async def _log(state: ProactiveStateRecord, event_type: str, payload: dict[str, Any]) -> None:
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type=event_type,
        window_index=state.current_window_index,
        trigger_type=FOLLOWUP_TRIGGER_TYPE,
        payload=payload,
    )


async def _enter_a_mode(state: ProactiveStateRecord, now: datetime, reason: str) -> None:
    await advance_to_next_window(
        state, now=now, event_type="followup_skipped", payload={"reason": reason},
    )


async def _send_followup(
    state: ProactiveStateRecord,
    turns: list[_Turn],
    verdict: TopicVerdict,
    now: datetime,
    *,
    trace_id: str | None,
) -> str | None:
    """生成并发送; 返回 assistant message id, 没发出去返回 None."""
    message = await generate_followup_message(state, turns, now)
    if not message:
        return None
    conversation_id = str(state.conversation_id)
    if not await topic_continuity.reserve_followup(conversation_id, now=now):
        return None
    message_id = await emit_proactive_message(
        conversation_id=str(state.conversation_id),
        user_id=state.user_id,
        agent_id=state.agent_id,
        workspace_id=state.workspace_id,
        message=message,
        trigger_type=FOLLOWUP_TRIGGER_TYPE,
        extra_metadata={"followup_reason": verdict.reason},
        trace_id=trace_id,
        # 生成期间用户回来了 → 不插入 (spec 任务互斥: 优先响应用户)
        abort_if_user_replied_since=now,
    )
    if not message_id:
        await topic_continuity.release_followup(conversation_id)
        return None
    from app.services.interaction.reply_context import save_last_reply_timestamp
    from app.services.proactive.sender import schedule_proactive_ai_memory

    await save_last_reply_timestamp(state.agent_id, state.user_id, when=now)
    schedule_proactive_ai_memory(
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=str(state.conversation_id),
        message=message,
    )
    logger.info(
        f"[FOLLOWUP] sent reason={verdict.reason}",
        extra={
            "event": EVT_PROACTIVE_FOLLOWUP_SENT,
            "followup_reason": verdict.reason,
            "message_len": len(message),
        },
    )
    return message_id


async def process_followup_window(
    state: ProactiveStateRecord,
    *,
    now: datetime | None = None,
) -> None:
    """判定窗到期 (orchestrator 已 claim 为 processing)。任何路径都会离开 processing."""
    now_ts = now or datetime.now(UTC)
    gate = await check_workspace(state)
    if gate:
        await stop_proactive_state(state, reason=gate.reason, now=now)
        return
    # 与 A 模式同: 绑 agent 让 per-agent 模型覆盖生效; trace + usage 让判定/追问的
    # LLM 调用按后台 scope 计入并发配额 (不占前台槽位), B 消息的 Trace 按钮可点。
    from app.services.llm.usage_tracker import traced_usage_session
    from app.services.runtime_config import bind_agent_context

    try:
        await bind_agent_context(state.agent_id)
        async with traced_usage_session(
            name=f"[proactive:{FOLLOWUP_TRIGGER_TYPE}]",
            scope="proactive",
            conversation_id=state.conversation_id,
            agent_id=state.agent_id,
            user_id=state.user_id,
        ) as tracer:
            await _judge_and_maybe_follow_up(state, now_ts, trace_id=tracer.safe_trace_id)
    except Exception as exc:  # noqa: BLE001 — 判定窗失败不能卡住 A 模式
        logger.warning(f"[FOLLOWUP] judge window failed: {exc!r}")
        await _enter_a_mode(state, now_ts, "error")


async def _judge_and_maybe_follow_up(
    state: ProactiveStateRecord,
    now: datetime,
    *,
    trace_id: str | None,
) -> None:
    conversation_id = state.conversation_id
    turns = await _load_recent_turns(conversation_id) if conversation_id else []
    last = turns[-1] if turns else None
    if last is None or last.role != "assistant" or last.proactive:
        await _enter_a_mode(state, now, "not_awaiting_user")
        return

    arm_reason = str((state.metadata or {}).get("reason") or "")
    if last.no_followup:
        # 兜底: 即使 arm 原因过期 (比如被更早一轮的 arm 覆盖), 也认最后那条回复本身
        await _enter_a_mode(state, now, "no_followup_reply")
        return
    verdict = await judge_topic_completion(turns, arm_reason=arm_reason, now=now)
    if verdict is None:
        await _enter_a_mode(state, now, "no_verdict")
        return
    await topic_continuity.record_verdict(
        str(conversation_id), verdict, anchor_message_id=last.id, now=now,
    )
    logger.info(
        f"[FOLLOWUP] judged status={verdict.status} reason={verdict.reason}",
        extra={
            "event": EVT_PROACTIVE_TOPIC_JUDGED,
            "verdict_status": verdict.status,
            "verdict_reason": verdict.reason,
        },
    )
    await _log(state, "topic_judged", {
        "status": verdict.status,
        "reason": verdict.reason,
        "anchor_message_id": last.id,
    })
    if not verdict.unfinished:
        await _enter_a_mode(state, now, "topic_finished")
        return

    skip_reason = await _followup_skip_reason(state, now)
    if skip_reason is None:
        message_id = await _send_followup(state, turns, verdict, now, trace_id=trace_id)
        if message_id:
            await advance_to_next_window(
                state,
                now=now,
                event_type="followup_sent",
                payload={"assistant_message_id": message_id, "reason": verdict.reason},
                restart_cycle=True,
            )
            return
        skip_reason = "not_generated"
    logger.info(
        f"[FOLLOWUP] skipped reason={skip_reason}",
        extra={"event": EVT_PROACTIVE_FOLLOWUP_SKIPPED, "skip_reason": skip_reason},
    )
    await _enter_a_mode(state, now, skip_reason)


async def _followup_skip_reason(state: ProactiveStateRecord, now: datetime) -> str | None:
    continuity = await topic_continuity.load_continuity(state.conversation_id)
    if continuity is None:
        return "continuity_unavailable"
    if not continuity.followup_available:
        return "followup_used_this_session"
    return await check_followup_gates(state, now=now)
