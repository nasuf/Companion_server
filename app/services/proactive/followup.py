"""话题完结判定 + B 模式追问 (《主动聊天机制（新增）》).

AI 说完最后一句满 5 分钟用户没回, 判定窗 (state window 0) 到期后进入本模块:

  1. 读最近对话。最后一条已经不是 AI 的普通回复 (用户其实回了 / 刚发过提醒等
     主动消息) → 不判定, 直接进 A 模式窗口。
  2. 判定话题是否完结: 告别按规则判完结; 危机照护 / 系统确认类问句不判; 其余
     交给小模型。拿不准一律判完结 —— 误判完结只是少一句追问, 误判未完结就是打扰。
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
import re
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
    SESSION_GAP_SECONDS,
    TopicVerdict,
)
from app.services.llm.models import get_chat_model, get_utility_model, invoke_json, invoke_text
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
from app.services.prompting.utils import is_skip_output, render_prompt
from app.services.schedule_domain.time_service import _TZ

logger = logging.getLogger(__name__)

UTC = timezone.utc

_RECENT_LIMIT = 12
_JUDGE_TIMEOUT_S = 10.0
_GENERATE_TIMEOUT_S = 20.0
_JUDGE_REASONS = frozenset({"ai_question", "user_story", "interrupted", "natural_end"})
# 兜底过滤: prompt 已要求别提"对方没回", LLM 仍写出催促/查岗口吻时宁可不发。
# 自然追问不能误杀: "后来那个人呢""现在吗""然后呢？后来你去哪儿了" 都得放行 ——
# 所以"在吗"要求前面是句首/标点/你/宝, "人呢"要求句首或标点, "去哪了/睡着了"
# 只在整句就是它时才算查岗。
_NEEDY_PATTERN = re.compile(
    r"(?:^|[，。！？!?,.~～…\s]|你|宝)(?:还)?(?:在吗|在不在|在嘛)"
    r"|(?:^|[，。！？!?,.~～…\s])人呢"
    r"|还在忙|是不是(?:还)?(?:在|去)?忙|忙完了?[吗没]"
    r"|怎么(?:不|没)(?:回|理|说话|动静)|不理我|没回我|理理我|回我一下"
    r"|^\W*你?(?:去哪(?:了|啦|儿了)|睡着了)[吗呀啊？?！!~～。]*$"
    r"|消失了|打扰(?:到)?你|不好意思打扰|等你(?:回|好久)"
)


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

def _is_context_noise(metadata: dict[str, Any]) -> bool:
    """游戏播报 / 收礼小灰条不是"聊天内容", 不参与话题判定."""
    kind = str(metadata.get("kind") or "")
    return kind.startswith("game") or bool(metadata.get("offering_received"))


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


def _current_segment(turns: list[_Turn]) -> list[_Turn]:
    """只留最后一段连续对话: 隔了 ≥3h 的旧内容是另一个会话, 不该影响这次判定."""
    start = 0
    for i in range(1, len(turns)):
        prev, cur = turns[i - 1].created_at, turns[i].created_at
        if prev and cur and (cur - prev).total_seconds() >= SESSION_GAP_SECONDS:
            start = i
    return turns[start:]


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
        if _is_context_noise(metadata):
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
    return _current_segment(turns)


def format_turns(turns: list[_Turn], *, max_chars: int = 120) -> str:
    lines = []
    for t in turns:
        stamp = f"[{t.created_at.astimezone(_TZ).strftime('%H:%M')}] " if t.created_at else ""
        speaker = "AI" if t.role == "assistant" else "用户"
        lines.append(f"{stamp}{speaker}: {t.text[:max_chars]}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Topic completion judgement
# ────────────────────────────────────────────────────────────────────

def _clean_pending_topic(raw: str) -> str:
    """注入回复 prompt 时是「你们上次聊到「…」」, 称呼要换成那段 prompt 的视角.

    只换作主语/宾语的称呼 ("AI问用户周末去哪" → "你问对方周末去哪"),
    "学AI绘画" 这类话题本身里的 AI 不动。
    """
    text = re.sub(r"(?<![A-Za-z])AI(?=问|说|想|提|让|跟|和)", "你", raw)
    text = re.sub(r"(?:^|(?<=问|跟|和|给|让))用户", "对方", text)
    return text.strip()[:15]


def _parse_verdict(raw: Any) -> TopicVerdict | None:
    if not isinstance(raw, dict):
        return None
    status = str(raw.get("status") or "").strip().lower()
    if status not in ("finished", "unfinished"):
        return None
    reason = str(raw.get("reason") or "").strip()
    if reason not in _JUDGE_REASONS:
        reason = "natural_end" if status == "finished" else "interrupted"
    pending = _clean_pending_topic(str(raw.get("pending_topic") or ""))
    if status == "unfinished" and not pending:
        # 说不出"还没聊完什么"的未完结, 多半是模型在猜 —— 按完结处理
        return TopicVerdict("finished", "natural_end")
    return TopicVerdict(status, reason, pending if status == "unfinished" else "")  # type: ignore[arg-type]


async def judge_topic_completion(turns: list[_Turn], *, arm_reason: str) -> TopicVerdict | None:
    """None = 不适用 / 判不出 (不写结论, 聊天侧维持原有逻辑)."""
    if arm_reason == ARM_REASON_FAREWELL:
        return TopicVerdict("finished", "farewell")
    if arm_reason in NO_FOLLOWUP_REASONS:
        return None
    try:
        raw = await asyncio.wait_for(
            render_prompt(
                "proactive.topic_completion_judge",
                {"conversation": format_turns(turns)},
                lambda p: invoke_json(get_utility_model(), p),
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

_REASON_HINTS = {
    "ai_question": "你刚问了对方一个问题",
    "user_story": "对方的事才讲到一半",
    "interrupted": "你们在商量的事还没定下来",
}


def _clean_followup(text: str | None) -> str | None:
    if not text or is_skip_output(text):
        return None
    # 通用回复规则允许 "||" 分条, 主动消息是单条投递: 拼成一句, 不能只发前半句
    segments = [seg.strip() for seg in text.split("||") if seg.strip()]
    text = "，".join(segments).strip().strip("\"'“”「」")
    if len(text) < 4:
        return None
    if _NEEDY_PATTERN.search(text):
        logger.info(f"[FOLLOWUP] dropped needy phrasing: {text[:30]!r}")
        return None
    return text


async def generate_followup_message(
    state: ProactiveStateRecord,
    turns: list[_Turn],
    verdict: TopicVerdict,
) -> str | None:
    from app.services.proactive.sender import build_personality_brief
    from app.services.relationship.ai_mood import load_ai_mood
    from app.services.relationship.emotion import emotion_to_tone

    agent = await db.aiagent.find_unique(where={"id": state.agent_id})
    if not agent:
        return None
    mood = await load_ai_mood(state.conversation_id)
    try:
        raw = await asyncio.wait_for(
            render_prompt(
                "proactive.followup_unfinished",
                {
                    "personality_brief": build_personality_brief(agent),
                    "current_mood": emotion_to_tone(mood),
                    "pending_topic": verdict.pending_topic,
                    "situation": _REASON_HINTS.get(verdict.reason, "话题还没聊完"),
                    "recent_context": format_turns(turns),
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
    message = await generate_followup_message(state, turns, verdict)
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
        f"[FOLLOWUP] sent reason={verdict.reason} pending={verdict.pending_topic!r}",
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
    verdict = await judge_topic_completion(turns, arm_reason=arm_reason)
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
        "pending_topic": verdict.pending_topic,
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
