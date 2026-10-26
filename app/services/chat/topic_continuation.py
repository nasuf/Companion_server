"""被动回复高级承接 (《主动聊天机制（新增）》+《主动交流提示词（新增）》).

用户这条消息到达时, 若上一轮被判定「话题未完结」(proactive/followup.py 写的结论):

  - 隔 >10min 且本会话没追问过 → 一句承接时间差的开场短句 (chat.topic_continuation_return)
  - 用户换了个完全无关的新话题 (chat.topic_jump_detect 判) → 一句轻柔过渡 (chat.topic_continuation_jump)

两句都作为独立气泡排在正常回复之前 —— 提示词文档的输出格式是「回归承接短句 +
旧话题过渡句 + 新话题正常回复」; 主回复 prompt 再注入「话题接续」段告诉模型这
两句已经说了, 别重复。

在意图识别 / 记忆检索的同时并行生成 (orchestrator 早早 create_task), 不拉长回复
链路; 任何一步失败或超时只是少一句, 不影响正常回复。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.observability.events import EVT_CHAT_TOPIC_CONTINUATION
from app.services.interaction.topic_continuity import (
    FOLLOWUP_TRIGGER_TYPE,
    ContinuationCue,
    clean_single_line,
    explain_cue,
    is_dialogue_noise,
    load_continuity,
)
from app.services.llm.models import get_chat_model, get_utility_model, invoke_text
from app.services.prompting.utils import render_prompt
from app.services.schedule_domain.time_expression import format_clock

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 被动承接使用统一模板窗口；主动完结判断和 B 追问仍由各自的 20 轮窗口负责。
_LLM_TIMEOUT_S = 6.0
# 主回复已经准备好时最多再等这么久: 承接句是锦上添花, 不能把整轮回复拖慢
_AWAIT_BUDGET_S = 3.0


@dataclass(frozen=True)
class TopicContinuation:
    cue: ContinuationCue
    # 排在正常回复之前的独立气泡 (承接短句 / 过渡句), 可能为空
    lines: list[str] = field(default_factory=list)
    # explain_cue 的结论: followup_reply / finished / unfinished
    reason: str = "unfinished"
    # 生成失败的句子数 (失败只是少一句, 但要看得见)
    failed_lines: int = 0


@dataclass(frozen=True)
class ContinuationSkip:
    """Why this turn carries no continuation; written to the reply diagnostics."""

    status: str  # patience_low / offering_turn / nothing_to_continue
    detail: str | None = None
    gap_seconds: float | None = None


def record_topic_continuation(diagnostics: dict[str, Any], status: str, **fields: Any) -> None:
    """One bounded record per reply; never contains message or generated text.

    status: applied / not_spawned / user_low / patience_low / offering_turn /
    timeout / error / nothing_to_continue.
    """
    diagnostics["topic_continuation"] = {"version": 1, "status": status, **fields}


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _history_text(history: list[Any], current_turn_ids: set[str]) -> str:
    """history 是 orchestrator 取出的消息行 (带 metadata, 才能滤掉游戏播报/收礼小灰条)."""
    from app.services.chat_media.prompt import render_message_content_for_prompt

    from app.services.chat.dialogue_context import render_dialogue as render_recent
    entries = []
    for m in history:
        metadata = m.metadata if isinstance(getattr(m, "metadata", None), dict) else None
        if is_dialogue_noise(metadata):
            continue
        text = render_message_content_for_prompt(str(m.content or ""), metadata).strip()
        if text:
            entries.append({"id": getattr(m, "id", None), "role": m.role, "content": text,
                            "metadata": metadata, "createdAt": getattr(m, "createdAt", None)})
    return render_recent(entries, exclude_ids=current_turn_ids, purpose="continuation").text



def _clean_line(raw: str | None) -> str | None:
    """单条气泡: 对方此刻在场 (查岗口吻不算冒犯), 且守住单条字数上限."""
    from app.services.prompts.system_prompts import MAX_PER_REPLY
    from app.services.chat.reply_formatting import truncate_at_sentence

    line = clean_single_line(raw, user_present=True)
    return truncate_at_sentence(line, MAX_PER_REPLY) if line else None


async def _generate(key: str, params: dict[str, Any], *, utility: bool = False) -> str | None:
    model = get_utility_model if utility else get_chat_model
    try:
        return await asyncio.wait_for(
            render_prompt(key, params, lambda p: invoke_text(model(), p), strip_split=False),
            timeout=_LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.info(f"[CONTINUITY] {key} timed out")
        return None


async def _return_line(conversation: str, *, ai_last_at: datetime, replied_at: datetime,
                       gap_text: str, user_message: str, personality: str) -> str | None:
    raw = await _generate("chat.topic_continuation_return", {
        "conversation": conversation,
        "ai_last_send_time": format_clock(ai_last_at),
        "current_time": format_clock(replied_at),
        "time_gap": gap_text,
        "user_msg": user_message,
        "personality_brief": personality,
    })
    return _clean_line(raw)


async def _transition_line(conversation: str, *, user_message: str, personality: str) -> str | None:
    verdict = await _generate("chat.topic_jump_detect", {
        "conversation": conversation,
        "user_msg": user_message,
    }, utility=True)
    if "新话题" not in str(verdict or ""):
        return None  # 接着聊 / 判不出 → 不提旧话题
    raw = await _generate("chat.topic_continuation_jump", {
        "conversation": conversation,
        "user_new_msg": user_message,
        "personality_brief": personality,
    })
    return _clean_line(raw)


async def build_topic_continuation(
    *,
    conversation_id: str,
    previous_assistant: Any,
    replied_at: datetime | None,
    user_message: str,
    history: list[Any],
    current_turn_ids: set[str],
    agent: Any,
    offering_turn: bool,
    patience_low: bool,
) -> TopicContinuation | ContinuationSkip:
    """ContinuationSkip = 不适用 (维持原有重逢感知逻辑), 并说明原因."""
    # 红包/礼物本身就是新话题; AI 还在生气时语气交给「情绪状态提醒」段
    if offering_turn:
        return ContinuationSkip("offering_turn")
    if patience_low:
        return ContinuationSkip("patience_low")
    if previous_assistant is None:
        return ContinuationSkip("nothing_to_continue", "no_previous_turn")
    ai_last_at = _as_datetime(getattr(previous_assistant, "createdAt", None))
    if ai_last_at is None:
        return ContinuationSkip("nothing_to_continue", "no_previous_reply_time")
    reply_at = replied_at or datetime.now(UTC)
    gap = max(0.0, (reply_at - ai_last_at).total_seconds())
    metadata = getattr(previous_assistant, "metadata", None)
    cue, reason = explain_cue(
        await load_continuity(conversation_id),
        previous_assistant_id=getattr(previous_assistant, "id", None),
        gap_seconds=gap,
        previous_assistant_is_followup=(
            isinstance(metadata, dict) and metadata.get("trigger_type") == FOLLOWUP_TRIGGER_TYPE
        ),
    )
    if cue is None:
        return ContinuationSkip("nothing_to_continue", reason, gap)
    if not cue.unfinished:
        return TopicContinuation(cue, reason=reason)

    from app.services.chat.prompt_builder import format_gap_text
    from app.services.mbti import build_personality_brief

    conversation = _history_text(history, current_turn_ids)
    personality = build_personality_brief(agent)
    tasks = []
    if cue.return_line:
        tasks.append(_return_line(
            conversation, ai_last_at=ai_last_at, replied_at=reply_at,
            gap_text=format_gap_text(gap), user_message=user_message, personality=personality,
        ))
    tasks.append(_transition_line(conversation, user_message=user_message, personality=personality))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    lines = [r for r in results if isinstance(r, str) and r]
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in failed:
        logger.warning(f"[CONTINUITY] line generation failed: {r!r}")
    logger.info(
        f"[CONTINUITY] return_line={cue.return_line} lines={len(lines)} gap={int(gap)}s",
        extra={
            "event": EVT_CHAT_TOPIC_CONTINUATION,
            "return_line": cue.return_line,
            "n_lines": len(lines),
            "gap_seconds": int(gap),
        },
    )
    return TopicContinuation(cue, lines, reason=reason, failed_lines=len(failed))


async def await_topic_continuation(
    task: asyncio.Task | None,
    diagnostics: dict[str, Any],
    *,
    not_spawned_detail: str | None = None,
) -> TopicContinuation | None:
    """主回复生成前取结果; 已被短路路径取消 / 出错都按"没有"处理, 但记下原因."""
    if task is None:
        record_topic_continuation(diagnostics, "not_spawned", detail=not_spawned_detail)
        return None
    # 已被取消的任务不能 await (CancelledError 不是 Exception)
    if task.cancelled():
        record_topic_continuation(diagnostics, "error", detail="cancelled")
        return None
    try:
        result = await asyncio.wait_for(task, timeout=_AWAIT_BUDGET_S)
    except asyncio.TimeoutError:
        logger.info("[CONTINUITY] skipped: not ready within budget")
        record_topic_continuation(diagnostics, "timeout", budget_seconds=_AWAIT_BUDGET_S)
        return None
    except Exception as e:  # noqa: BLE001 — 锦上添花, 失败就按没有判定处理
        logger.warning(f"[CONTINUITY] skipped: {type(e).__name__}")
        record_topic_continuation(diagnostics, "error", detail=type(e).__name__)
        return None
    if isinstance(result, TopicContinuation):
        record_topic_continuation(
            diagnostics, "applied", detail=result.reason,
            unfinished=result.cue.unfinished, n_lines=len(result.lines),
            failed_lines=result.failed_lines, gap_seconds=int(result.cue.gap_seconds),
        )
        return result
    if isinstance(result, ContinuationSkip):
        record_topic_continuation(
            diagnostics, result.status, detail=result.detail,
            gap_seconds=None if result.gap_seconds is None else int(result.gap_seconds),
        )
    else:
        record_topic_continuation(diagnostics, "nothing_to_continue")
    return None
