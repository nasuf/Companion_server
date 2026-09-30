"""A 模式主动消息 (原主动回复) 的生成与发送.

generate_and_send_proactive 按步骤编排:
  _check_send_eligibility  日限 / 疲劳分 / workspace / conversation
  _resolve_topic           spec §3.2 话题方向 + §4.1/§4.2 来源抽签 → 上下文
  _attach_trending         A 模式新来源「全网热点」(4-1/4-2 筛选 → 4-3 生成, 默认关)
  _generate_message        按 (trigger_type, source, decay_final) 分发 prompt
  _prepare_attachments     音乐卡 / 链接卡
  emit_proactive_message   落库 + WS (生成期间用户回来则不插入)
  _after_emit / _persist_proactive_state   记账 + 状态机 → waiting_user

B 模式 (话题未完结追问) 在 followup.py; 公共持久化与 WS 广播在 emit.py;
发送门槛在 gates.py。
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.runtime.distributed_lock import distributed_lock
from app.services.runtime.tasks import fire_background
from app.observability import bind_context
from app.observability.events import EVT_PROACTIVE_SENT, EVT_PROACTIVE_SKIPPED
from app.redis_client import get_redis
from app.services.llm.models import get_chat_model, invoke_text
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.proactive.emit import emit_proactive_message
from app.services.proactive.history import (
    can_send_proactive,
    get_proactive_fatigue_score,
    increment_proactive_count,
)
from app.services.proactive.context import build_proactive_context
from app.services.schedule_domain.time_expression import format_clock
from app.services.schedule_domain.time_service import _now_corrected
from app.services.proactive.policy import select_topic_source, select_topic_theme
from app.services.mbti import build_personality_brief
from app.services.relationship.emotion import emotion_to_tone
from app.services.workspace.workspaces import (
    get_active_workspace,
    get_workspace_by_id,
    resolve_workspace_id,
)
from app.services.proactive.state import (
    STATUS_IDLE,
    STATUS_PROCESSING,
    STATUS_PROCESSING_TIMEOUT,
    STATUS_RUNNING,
    STATUS_WAITING_USER,
    ProactiveStateRecord,
    determine_proactive_stage,
    ensure_proactive_state_for_workspace,
    get_active_workspace_context,
    log_proactive_event,
    mark_proactive_sent,
)
from app.services.prompting.store import PromptDisabledError, get_prompt_text
from app.services.prompting.utils import is_skip_output, render_template
from app.services.interaction import topic_continuity
from app.services.interaction.reply_context import save_last_reply_timestamp

logger = logging.getLogger(__name__)

UTC = timezone.utc
SENDABLE_PROACTIVE_STATUSES = {STATUS_IDLE}
# Admin QA (skip_limits): force-send by resetting transient blockers to idle
# without touching decay counters (silence_level_n / followup_plan_type).
_ADMIN_UNLOCKABLE_STATUSES = frozenset({
    STATUS_WAITING_USER,
    STATUS_RUNNING,
    STATUS_PROCESSING,
    STATUS_PROCESSING_TIMEOUT,
})

_MEMORY_SOURCES = frozenset({"ai_l1", "ai_l2", "user_l1", "user_l2", "relationship"})
_DIALOGUE_APPROACHES = ("关心追问后续", "话题轻度延伸")


# ────────────────────────────────────────────────────────────────────
# Eligibility checks
# ────────────────────────────────────────────────────────────────────

def _mark_proactive_skip(
    send_outcome: "AdminProactiveSendOutcome | None",
    reason: str,
) -> None:
    if send_outcome is not None:
        send_outcome.skip_reason = reason


async def _log_skip(
    state: ProactiveStateRecord,
    trigger_type: str,
    reason: str,
    *,
    conversation_id: str | None = None,
    extra: dict[str, Any] | None = None,
    send_outcome: "AdminProactiveSendOutcome | None" = None,
) -> None:
    _mark_proactive_skip(send_outcome, reason)
    payload: dict[str, Any] = {"reason": reason}
    if extra:
        payload.update(extra)
    logger.info(
        f"proactive skipped: trigger={trigger_type} reason={reason}",
        extra={
            "event": EVT_PROACTIVE_SKIPPED,
            "trigger_type": trigger_type,
            "skip_reason": reason,
            "stage": state.stage,
        },
    )
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=conversation_id or state.conversation_id,
        event_type="send_skipped",
        window_index=state.current_window_index,
        trigger_type=trigger_type,
        payload=payload,
    )


@dataclass
class _SendPrep:
    conversation_id: str
    cooldown: dict[str, int]
    exclude_memory_ids: set[str]


async def _ensure_conversation_for_admin_test(
    *,
    workspace_id: str,
    user_id: str,
    agent_id: str,
) -> str | None:
    """Admin QA: create a conversation on the target workspace when missing."""
    try:
        rows = await db.query_raw(
            """
            SELECT id
            FROM conversations
            WHERE workspace_id = $1
              AND is_deleted = FALSE
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT 1
            """,
            workspace_id,
        )
    except Exception as e:
        logger.warning(
            f"admin proactive list conversations failed ws={workspace_id[:8]}: {e}"
        )
        return None
    if rows:
        return str(rows[0]["id"])

    workspace = await get_workspace_by_id(workspace_id)
    if workspace is None or getattr(workspace, "status", None) != "active":
        return None

    try:
        conv = await db.conversation.create(
            data={
                "user": {"connect": {"id": user_id}},
                "agent": {"connect": {"id": agent_id}},
                "workspace": {"connect": {"id": workspace_id}},
                "title": None,
            }
        )
        return conv.id
    except Exception as e:
        logger.warning(
            f"admin proactive create conversation failed ws={workspace_id[:8]}: {e}"
        )
        try:
            existing = await db.conversation.find_first(
                where={
                    "workspaceId": workspace_id,
                    "agentId": agent_id,
                    "userId": user_id,
                    "isDeleted": False,
                },
                order={"updatedAt": "desc"},
            )
        except Exception:
            existing = None
        return existing.id if existing else None


async def _check_send_eligibility(
    state: ProactiveStateRecord,
    trigger_type: str,
    *,
    skip_limits: bool = False,
    send_outcome: "AdminProactiveSendOutcome | None" = None,
) -> _SendPrep | None:
    """spec §9 互斥: 检查日限/workspace/conversation. 失败返回 None."""
    if not skip_limits:
        if not await can_send_proactive(state.agent_id, state.user_id):
            await _log_skip(
                state, trigger_type, "daily_limit", send_outcome=send_outcome,
            )
            return None
        fatigue = await get_proactive_fatigue_score(
            state.agent_id,
            state.user_id,
            workspace_id=state.workspace_id,
        )
        if fatigue.get("block"):
            await _log_skip(
                state,
                trigger_type,
                "fatigue_score",
                extra=fatigue,
                send_outcome=send_outcome,
            )
            return None

    workspace_context = await get_active_workspace_context(state.workspace_id)
    if not workspace_context:
        await _log_skip(
            state, trigger_type, "workspace_missing", send_outcome=send_outcome,
        )
        return None

    conversation_id = str(
        workspace_context.get("conversation_id") or state.conversation_id or ""
    )
    if not conversation_id and skip_limits:
        conversation_id = await _ensure_conversation_for_admin_test(
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
        ) or ""
        if conversation_id and conversation_id != (state.conversation_id or ""):
            try:
                await db.execute_raw(
                    """
                    UPDATE proactive_states
                    SET conversation_id = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    state.id,
                    conversation_id,
                )
            except Exception as e:
                logger.warning(
                    f"admin proactive bind conversation failed state={state.id[:8]}: {e}"
                )
    if not conversation_id:
        await _log_skip(
            state, trigger_type, "conversation_missing", send_outcome=send_outcome,
        )
        return None

    cooldown, exclude = _apply_memory_cooldown(state, trigger_type)
    return _SendPrep(
        conversation_id=conversation_id,
        cooldown=cooldown,
        exclude_memory_ids=exclude,
    )


# ────────────────────────────────────────────────────────────────────
# spec §9 记忆冷却 (-1 / +50)
# ────────────────────────────────────────────────────────────────────

def _apply_memory_cooldown(
    state: ProactiveStateRecord,
    trigger_type: str,
) -> tuple[dict[str, int], set[str]]:
    """spec §9 记忆去重规则.

    - metadata["memory_cooldown"] = {memory_id: int}
    - 只在 memory_proactive 候选检索时 -1
    - 抽中后置 50 (在 _persist_proactive_state 处理)
    - 兼容旧 used_memory_ids 列表 → 一次性迁移为冷却 50
    """
    metadata = state.metadata or {}
    cooldown: dict[str, int] = dict(metadata.get("memory_cooldown") or {})
    if not cooldown and metadata.get("used_memory_ids"):
        cooldown = {mid: 50 for mid in (metadata.get("used_memory_ids") or [])}
    if trigger_type == "memory_proactive":
        cooldown = {mid: cd - 1 for mid, cd in cooldown.items() if cd - 1 > 0}
    exclude = {mid for mid, cd in cooldown.items() if cd > 0}
    return cooldown, exclude


# ────────────────────────────────────────────────────────────────────
# Personality brief & prompt dispatch
# ────────────────────────────────────────────────────────────────────

# (trigger_type, source) → prompt key
_PROMPT_KEY_BY_SOURCE: dict[tuple[str, str], str] = {
    ("silence_wakeup", "ai_l1"): "proactive.silence_ai_memory",
    ("silence_wakeup", "ai_l2"): "proactive.silence_ai_memory",
    ("silence_wakeup", "user_l1"): "proactive.silence_user_memory",
    ("silence_wakeup", "user_l2"): "proactive.silence_user_memory",
    ("silence_wakeup", "ai_schedule"): "proactive.silence_schedule",
    ("silence_wakeup", "greeting"): "proactive.silence_plain",
    ("silence_wakeup", "music"): "music.proactive_recommend",
    ("silence_wakeup", "recent_dialogue"): "proactive.recent_dialogue",
    ("memory_proactive", "ai_l1"): "proactive.memory_ai",
    ("memory_proactive", "ai_l2"): "proactive.memory_ai",
    ("memory_proactive", "user_l1"): "proactive.memory_user",
    ("memory_proactive", "user_l2"): "proactive.memory_user",
    # Phase 2 关系记忆: 共同经历 (memories_ai 生活/交互) 走 AI 记忆模板 —
    # 素材本来就是 AI 第一人称叙述的"我和用户…", memory_ai 模板语气吻合.
    ("memory_proactive", "relationship"): "proactive.memory_ai",
    # 《主动交流提示词》提示词3: A 模式「带时间感知历史对话搭话」
    ("memory_proactive", "recent_dialogue"): "proactive.recent_dialogue",
    ("scheduled_scene", "ai_schedule"): "proactive.scheduled_scene",
}

_OPTIONAL_REFERENCE_KEYS = frozenset({
    "ai_memory",
    "user_memory",
    "user_portrait",
    "recent_context",
})


def _format_prompt(key: str, ctx: dict, personality_brief: str) -> str | None:
    """按 prompt key 选定填充字段."""
    topic = ctx.get("topic_theme") or "日常"
    memories = ctx.get("proactive_memories") or []
    schedule_status = ctx.get("schedule_status") or {}
    activity = str(schedule_status.get("activity") or "自由时间")
    status = str(schedule_status.get("status") or "idle")
    memory_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无）"

    # 主动消息保留 current_mood 字段；无运行时 AI 情绪向量时使用标签情绪助手的中性语气。
    user_portrait = ctx.get("user_portrait") or "(未知)"
    recent_context = ctx.get("recent_context") or "(无)"
    current_mood = emotion_to_tone(ctx.get("emotion"))
    silence_shared = {
        "topic": topic,
        "user_portrait": user_portrait,
        "recent_context": recent_context,
        "current_mood": current_mood,
    }
    fields_by_key: dict[str, dict[str, Any]] = {
        "proactive.silence_plain": {
            "personality_brief": personality_brief,
            **silence_shared,
        },
        "proactive.silence_ai_memory": {
            "personality_brief": personality_brief,
            "ai_memory": memory_text,
            **silence_shared,
        },
        "proactive.silence_user_memory": {
            "personality_brief": personality_brief,
            "user_memory": memory_text,
            **silence_shared,
        },
        "proactive.silence_schedule": {
            "personality_brief": personality_brief,
            "current_activity": f"{activity}({status})",
            **silence_shared,
        },
        # Spec §4.2 + 指令模版 P24-25: 性格 / 当前心境 / 记忆 / 话题主题
        "proactive.memory_ai": {
            "personality_brief": personality_brief,
            "current_mood": current_mood,
            "ai_memory": memory_text,
            "topic": topic,
        },
        "proactive.memory_user": {
            "personality_brief": personality_brief,
            "current_mood": current_mood,
            "user_memory": memory_text,
            "topic": topic,
        },
        "proactive.recent_dialogue": {
            "personality_brief": personality_brief,
            "recent_dialogue": ctx.get("recent_dialogue") or "（无）",
            "current_time": format_clock(_now_corrected()),
            # 提示词规定"随机二选一": 由代码掷骰, 让模型自己选并不随机
            "approach": random.choice(_DIALOGUE_APPROACHES),
        },
        "proactive.scheduled_scene": {
            "personality_brief": personality_brief,
            # 必须用项目时区 _TZ (Asia/Shanghai), 不能 datetime.now().astimezone() —
            # 后者会跟服务器系统时区走, 容器跑在 UTC 里就会让 LLM 看到"现在是 00:51"
            # 然后回"夜深了" (生产 bug 2026-05-03 trace: UTC 00:51 = 上海 08:51,
            # 用户在吃早饭收到"夜深了"). 同时复用 _now_corrected 保留 NTP drift 修正.
            "time": _now_corrected().strftime("%H:%M"),
            "activity": activity,
            "status": status,
        },
        "music.proactive_recommend": {
            "personality_brief": personality_brief,
            "song_name": getattr(ctx.get("music_track"), "title", "这首歌"),
            "artist": getattr(ctx.get("music_track"), "artist", "Jamendo"),
            "scene_hint": ctx.get("scene_hint") or "轻量分享一首适合此刻的歌。",
        },
    }
    fields = fields_by_key.get(key)
    if fields is None:
        return None
    try:
        # tpl is fetched async by caller (this is a sync helper for clarity)
        tpl = ctx["__tpl"]
        return render_template(
            tpl,
            fields,
            optional_keys=_OPTIONAL_REFERENCE_KEYS,
            safe=False,
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"Prompt format failed key={key}: {e}")
        return None


async def _generate_message(ctx: dict) -> str | None:
    """spec §4 按 (trigger_type, source) 分发到 7 个专属 prompt;
    spec §8.5 衰减最后一次优先 decay_final.
    """
    agent = ctx["agent"]
    trigger_type = ctx["trigger_type"]
    source = ctx.get("source") or "greeting"
    personality_brief = build_personality_brief(agent)
    # 《主动交流提示词》4-3: 热点已由 4-1/4-2 筛好并摘要 (_attach_trending)
    trending_pick = ctx.get("trending_pick")

    try:
        if ctx.get("is_decay_final"):
            tpl = await get_prompt_text("proactive.decay_final")
            prompt = tpl.format(personality_brief=personality_brief)
        elif trending_pick is not None:
            tpl = await get_prompt_text("proactive.trending_chat")
            prompt = tpl.format(
                personality_brief=personality_brief,
                hot_summary=trending_pick.summary,
            )
        else:
            key = _PROMPT_KEY_BY_SOURCE.get(
                (trigger_type, source), "proactive.silence_plain"
            )
            tpl = await get_prompt_text(key)
            ctx["__tpl"] = tpl
            prompt = _format_prompt(key, ctx, personality_brief)
            if not prompt:
                return None
    except PromptDisabledError as e:
        logger.info(f"[proactive-gen] prompt disabled key={e}")
        ctx["_skip_reason_detail"] = f"prompt_disabled:{e}"
        return None

    try:
        response = (await invoke_text(get_chat_model(), prompt)).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[proactive-gen] LLM invoke failed: {exc!r}")
        ctx["_skip_reason_detail"] = f"llm_error:{type(exc).__name__}"
        return None
    if is_skip_output(response):
        logger.info("[proactive-gen] LLM returned literal SKIP")
        ctx["_skip_reason_detail"] = "llm_skip_literal"
        return None
    if len(response) < 4:
        logger.info(f"[proactive-gen] LLM response too short len={len(response)} raw={response!r}")
        ctx["_skip_reason_detail"] = f"llm_response_too_short:len={len(response)}"
        return None

    # anti-repetition (2026-09-14 task#12): 生产路径 retry 一次 diversity hint;
    # 若 retry 仍相似, **仍然发送** —— 一条重复的消息比 empty_or_skip 静默失败好.
    #
    # admin_test 完全绕过: admin 反复触发同 topic 会兜死, 用户看到 "LLM 未生成"
    # 假象 (其实是守卫). 测试路径优先"每次都可见输出", anti-repeat 是生产路径的
    # 弱守卫, 不该在 QA 面里阻断.
    workspace_id = ctx.get("workspace_id")
    admin_test = bool(ctx.get("_admin_test"))
    if workspace_id and not admin_test:
        from app.services.proactive.recent_messages import is_repeat_of_recent
        if await is_repeat_of_recent(str(workspace_id), response):
            logger.info("[proactive-repeat] first attempt matched recent, retrying with diversity hint")
            diversity_prompt = (
                prompt
                + "\n\n【重要】最近你说过类似的话了, 换个开头、换个说法, "
                "不要跟上次一样. 保持同样自然, 但表达不同."
            )
            try:
                retry = (await invoke_text(get_chat_model(), diversity_prompt)).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[proactive-repeat] retry LLM failed: {exc!r}; shipping first attempt")
                return response
            if retry and not is_skip_output(retry) and len(retry) >= 4:
                response = retry
                if await is_repeat_of_recent(str(workspace_id), response):
                    # 重复即重复, 发出去 —— 不再 return None. 详见函数上方注释.
                    logger.info("[proactive-repeat] retry still similar, shipping anyway")
    return response


# ────────────────────────────────────────────────────────────────────
# Persistence wrapping (state + cooldown commit)
# ────────────────────────────────────────────────────────────────────

async def _persist_proactive_state(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    message: str,
    assistant_message_id: str,
    cooldown: dict[str, int],
    new_used_ids: set[str],
    now_ts: datetime,
) -> None:
    """spec §9 抽中的 mid 置 50 + mark_proactive_sent + last_reply_timestamp."""
    for mid in new_used_ids:
        cooldown[mid] = 50
    await mark_proactive_sent(
        state,
        trigger_type=trigger_type,
        message=message,
        assistant_message_id=assistant_message_id,
        now=now_ts,
        mark_daily_scene=(trigger_type == "scheduled_scene"),
        extra_metadata={
            "memory_cooldown": cooldown,
            "used_memory_ids": list(cooldown.keys()),
        },
    )
    await save_last_reply_timestamp(state.agent_id, state.user_id, when=now_ts)


def _should_use_music_source(trigger_type: str) -> bool:
    if trigger_type != "silence_wakeup":
        return False
    return random.random() < 0.04


async def _prepare_music_recommendation_source(
    ctx: dict[str, Any],
    *,
    conversation_id: str,
) -> str:
    from app.services import music

    schedule_status = ctx.get("schedule_status") or {}
    if str(schedule_status.get("status") or "idle") != "idle":
        return "music_skip_not_idle"
    open_session = await music.get_open_co_listening(conversation_id=conversation_id)
    if open_session is not None:
        return "greeting"
    library = music.default_libraries()[0]
    track = await music.fetch_random_track(library, index=0, use_cache=True)
    ctx["music_track"] = track
    return "music"


# ────────────────────────────────────────────────────────────────────
# Main entry: generate_and_send_proactive
# ────────────────────────────────────────────────────────────────────

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _generation_started_at(state: ProactiveStateRecord, fallback: datetime) -> datetime:
    """取消基准: 扫描路径用认领时刻 (门槛查"用户近期活跃"就在那之后), 而不是
    跑完门槛和 DB 往返之后才取的 now —— 中间落库的用户消息否则两头都漏掉。"""
    if state.status == STATUS_PROCESSING and state.last_attempt_at:
        claimed = state.last_attempt_at
        return claimed if claimed.tzinfo else claimed.replace(tzinfo=timezone.utc)
    return fallback


def schedule_proactive_ai_memory(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    message: str,
) -> None:
    """Spec §2.2: 每条 AI 主动消息也进 AI 自我记忆录入管线 (后台, 不阻塞发送).

    fire_background 持有任务引用 —— 裸 asyncio.create_task 的任务可能被中途 GC。
    """
    fire_background(_bg_proactive_ai_memory(
        user_id, message,
        conversation_id=conversation_id,
        agent_id=agent_id,
    ))


async def _resolve_topic(
    state: ProactiveStateRecord,
    trigger_type: str,
    prep: _SendPrep,
    *,
    admin_test: bool,
    send_outcome: "AdminProactiveSendOutcome | None",
) -> dict[str, Any] | None:
    """spec §3.2 话题方向 + §4.1/§4.2 来源抽签 → 组装上下文; 该取消本次发送时返回 None."""
    # spec §2.2: 话题亲密度可变, 触发前实时算; state.stage 仅在 session
    # start/restart 时持久化, 不追踪中途 intimacy 升级.
    stage = await determine_proactive_stage(state.agent_id, state.user_id)
    topic_theme = select_topic_theme(stage)
    source = select_topic_source(stage, trigger_type)
    if _should_use_music_source(trigger_type):
        source = "music"

    ctx = await build_proactive_context(
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        trigger_type=trigger_type,
        stage=stage,
        exclude_memory_ids=prep.exclude_memory_ids,
        source=source,
        topic_theme=topic_theme,
        conversation_id=prep.conversation_id,
    )
    # admin QA 反复触发时, anti-repetition 会兜死同 topic 的连测 → empty_or_skip 假象.
    # 标记后 _generate_message 会绕过 recent 相似度守卫, 保证每次测试都出可见结果.
    ctx["_admin_test"] = admin_test
    # context 可能改写来源 (近两天没聊过 → ai_l1 / greeting), 以它为准
    source = ctx.get("source") or source
    if source == "music":
        source = await _prepare_music_recommendation_source(
            ctx,
            conversation_id=prep.conversation_id,
        )
        if source == "music_skip_not_idle":
            await _log_skip(
                state, trigger_type, "music_source_not_idle",
                conversation_id=prep.conversation_id, send_outcome=send_outcome,
            )
            return None

    # spec §4.1 沉默唤醒兜底; §4.2 记忆主动失败时取消
    if source in _MEMORY_SOURCES and not ctx.get("proactive_memories"):
        if trigger_type != "silence_wakeup":
            await _log_skip(
                state, trigger_type, "memory_source_empty",
                conversation_id=prep.conversation_id,
                extra={"source": source}, send_outcome=send_outcome,
            )
            return None
        source = "greeting"
        ctx["scene_hint"] = "优先用轻量、低打扰的方式重新建立联系。"
    ctx["source"] = source
    ctx["stage"] = stage
    # spec §8.5 衰减最后一次
    ctx["is_decay_final"] = state.followup_plan_type == "thirty_day_final"
    return ctx


async def _attach_trending(
    ctx: dict[str, Any],
    state: ProactiveStateRecord,
    trigger_type: str,
    admin_test_options: "AdminProactiveTestOptions | None",
) -> bool:
    """A 模式新来源「全网热点内容搭话」(《主动交流提示词》4-1 / 4-2 → 4-3).

    命中 trending 概率门时抓 48 小时热榜 → trending_pick 筛一条能聊的 (爱好匹配 /
    随机) 并摘要 → ctx["trending_pick"], _generate_message 走 4-3 生成; 一条都
    不合适就按原来源正常发。整体开关: SystemConfig.proactive_trending_enabled (默认关)。
    返回是否抓了热榜 (admin QA 的 web_search_used)。
    """
    if ctx.get("is_decay_final") or ctx.get("source") == "music":
        # 衰减最后一次走专属 prompt / 音乐推荐挂的是音乐卡: 都不说热点, 否则消息与
        # 卡片不同源
        return False

    from app.services.proactive.trending_context import resolve_trending_context

    _text, trending_attached, trending_meta = await resolve_trending_context(
        trigger_type,
        topic=ctx.get("topic_theme"),
        admin_test_options=admin_test_options,
    )
    if not (trending_attached and trending_meta is not None and trending_meta.candidates):
        return trending_attached

    from app.services.proactive.featured_topics import get_recent_featured
    from app.services.proactive.trending_pick import pick_trending

    # 最近 featured 过的话题 (6h, 按 workspace 隔离) 排除, 连续触发会轮到别的
    pick = await pick_trending(
        list(trending_meta.candidates),
        user_id=state.user_id,
        workspace_id=state.workspace_id,
        exclude_titles=await get_recent_featured(str(state.workspace_id) if state.workspace_id else None),
    )
    if pick is not None:
        ctx["trending_pick"] = pick
        logger.info(f"[TRENDING] picked mode={pick.mode} title={str(pick.item.get('title'))[:30]!r}")
    return trending_attached


@dataclass
class _Attachments:
    extra_metadata: dict[str, Any]
    ws_payload_extra: dict[str, Any] | None = None
    link: Any = None
    link_skip_reason: str | None = None


async def _prepare_attachments(
    ctx: dict[str, Any],
    state: ProactiveStateRecord,
    prep: _SendPrep,
    *,
    trigger_type: str,
    message: str,
    trending_attached: bool,
    admin_test_options: "AdminProactiveTestOptions | None",
) -> _Attachments:
    """音乐卡 / 链接卡 (二选一)。"""
    source = ctx["source"]
    attachments = _Attachments(extra_metadata={"stage": ctx["stage"]})
    if source == "music" and ctx.get("music_track") is not None:
        from app.services.music_chat import card_from_track

        card = card_from_track(ctx["music_track"], intent="recommend", source="proactive")
        attachments.extra_metadata.update({
            "component_card": card,
            "music_proactive": True,
            "topic_source": "music",
        })
        attachments.ws_payload_extra = {"component_card": card}
        return attachments

    from app.services.chat_links import maybe_prepare_proactive_link_recommendation
    from app.services.proactive.trending_gate import should_attach_trending_link_card

    force_link = False
    skip_link = False
    if admin_test_options is not None:
        force_link = admin_test_options.use_link_card
        skip_link = not admin_test_options.use_link_card
    # 消息-卡片硬耦合: 热点筛好的那条, 卡片必须挂同一条, 不许独立再搜
    # (修"文本说银锁骨链 + 卡说机场"这种语义脱钩)。抓了热榜但一条没选中时消息
    # 与热点无关, 不强挂卡片。
    trending_pick = ctx.get("trending_pick")
    preselected = trending_pick.item if trending_pick is not None else None
    if admin_test_options is None and trending_attached and preselected is not None:
        force_link = should_attach_trending_link_card(trending_attached=True)
    link, attachments.link_skip_reason = await maybe_prepare_proactive_link_recommendation(
        user_id=state.user_id,
        conversation_id=prep.conversation_id,
        trigger_type=trigger_type,
        source=source,
        topic=ctx.get("topic_theme"),
        stage=ctx["stage"],
        message=message,
        force=force_link,
        skip=skip_link,
        preselected_item=preselected,
    )
    if link is not None:
        attachments.link = link
        attachments.extra_metadata.update({
            "component_card": link.component_card,
            "link_card": link.link_card_metadata,
            "link_proactive": True,
            "topic_source": "link",
        })
        attachments.ws_payload_extra = {"component_card": link.component_card}
    return attachments


async def _after_emit(
    ctx: dict[str, Any],
    state: ProactiveStateRecord,
    prep: _SendPrep,
    *,
    message: str,
    assistant_message_id: str,
    attachments: _Attachments,
) -> None:
    """发出后的记账: 防重复池 / 已推热点 / 链接卡绑定 / 音乐一起听."""
    from app.services.proactive.recent_messages import remember_recent

    # 只在真发出去后记, 免得各种 skip 路径污染防重复池
    await remember_recent(state.workspace_id, message)
    trending_pick = ctx.get("trending_pick")
    if trending_pick is not None:
        featured_title = str(trending_pick.item.get("title") or "").strip()
        if featured_title:
            from app.services.proactive.featured_topics import remember_featured

            await remember_featured(
                str(state.workspace_id) if state.workspace_id else None, featured_title,
            )
    if attachments.link is not None:
        from app.services.chat_links import bind_link_card_to_message

        await bind_link_card_to_message(
            link_id=attachments.link.link.id,
            message_id=assistant_message_id,
            user_id=state.user_id,
            conversation_id=prep.conversation_id,
        )
    if ctx["source"] == "music" and ctx.get("music_track") is not None:
        await _start_proactive_co_listening(ctx, state, prep)


async def _start_proactive_co_listening(
    ctx: dict[str, Any],
    state: ProactiveStateRecord,
    prep: _SendPrep,
) -> None:
    from app.models.music import MusicTrackPayload
    from app.services import music
    from app.services.music_status import persist_and_emit_music_status

    track = ctx["music_track"]
    await music.start_co_listening(
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=prep.conversation_id,
        workspace_id=state.workspace_id,
        payload=MusicTrackPayload(
            id=track.id,
            title=track.title,
            artist=track.artist,
            album=track.album,
            library=track.library,
            url=track.url,
            duration_sec=track.duration_sec,
            cover_key=track.cover_key,
            accent_a=track.accent_a,
            accent_b=track.accent_b,
            source=track.source,
            metadata=track.metadata,
        ),
        initiated_by="agent",
        status="active",
        position_seconds=0,
        is_playing=False,
    )
    await persist_and_emit_music_status(
        conversation_id=prep.conversation_id,
        status="started",
        track=track,
        actor="agent",
        actor_name=getattr(ctx.get("agent"), "name", None) or "我",
    )


async def generate_and_send_proactive(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    now: datetime | None = None,
    skip_limits: bool = False,
    admin_test_options: "AdminProactiveTestOptions | None" = None,
    send_outcome: "AdminProactiveSendOutcome | None" = None,
) -> bool:
    """A 模式 (原主动回复) 发送主流程: 资格 → 话题 → 热点 → 生成 → 附件 → 发送 → 记账."""
    # 绑 ContextVar 让本调用栈的 LLM 工厂应用该 agent 的模型 override.
    # 不绑的话主动消息生成 / AI 自我记忆抽取都会用 system 全局, 跟 chat 路径
    # 的 per-agent 行为不一致, 同时 token stats 会把这些 LLM 调用归到全局模型名.
    from app.services.runtime_config import bind_agent_context
    await bind_agent_context(state.agent_id)

    now_ts = now or datetime.now(UTC)
    prep = await _check_send_eligibility(
        state, trigger_type, skip_limits=skip_limits, send_outcome=send_outcome,
    )
    if prep is None:
        return False
    ctx = await _resolve_topic(
        state, trigger_type, prep,
        admin_test=admin_test_options is not None,
        send_outcome=send_outcome,
    )
    if ctx is None:
        return False
    source = ctx["source"]
    stage = ctx["stage"]
    trending_attached = await _attach_trending(ctx, state, trigger_type, admin_test_options)

    # 主动消息也开 trace + usage_session, 名字 [proactive:trigger_type],
    # 方便看板与统计 dashboard 区分被动回复.
    from app.services.llm.usage_tracker import traced_usage_session
    async with traced_usage_session(
        name=f"[proactive:{trigger_type}]",
        scope="proactive", conversation_id=prep.conversation_id,
        agent_id=state.agent_id, user_id=state.user_id,
    ) as tracer:
        message = await _generate_message(ctx)
        if not message:
            # 细分 reason: prompt_disabled:X / llm_error:X / llm_skip_literal /
            # llm_response_too_short:len=N. admin QA 直接看到 "为什么没消息".
            detail = str(ctx.get("_skip_reason_detail") or "empty_or_skip")
            await _log_skip(
                state, trigger_type, detail,
                conversation_id=prep.conversation_id, send_outcome=send_outcome,
            )
            return False

        attachments = await _prepare_attachments(
            ctx, state, prep,
            trigger_type=trigger_type,
            message=message,
            trending_attached=trending_attached,
            admin_test_options=admin_test_options,
        )
        if send_outcome is not None:
            send_outcome.web_search_used = trending_attached
            send_outcome.link_card_used = attachments.link is not None
            # 音乐卡走自己的路径, 那时 link_skip_reason 为 None 是合理的
            if attachments.link is None:
                send_outcome.link_card_skip_reason = attachments.link_skip_reason
            send_outcome.extra.update({"trigger_type": trigger_type, "source": source})

        assistant_message_id = await emit_proactive_message(
            conversation_id=prep.conversation_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            workspace_id=state.workspace_id,
            message=message,
            trigger_type=trigger_type,
            extra_metadata=attachments.extra_metadata,
            ws_payload_extra=attachments.ws_payload_extra,
            trace_id=tracer.safe_trace_id,
            # 生成期间用户回来了 → 不插入, 优先响应用户 (spec 任务互斥)
            abort_if_user_replied_since=_generation_started_at(state, now_ts),
        )
        if not assistant_message_id:
            await _log_skip(
                state, trigger_type, "user_replied_during_generation",
                conversation_id=prep.conversation_id, send_outcome=send_outcome,
            )
            return False
        await _after_emit(
            ctx, state, prep,
            message=message,
            assistant_message_id=assistant_message_id,
            attachments=attachments,
        )
    logger.info(
        f"proactive sent: trigger={trigger_type} source={source} stage={stage}",
        extra={
            "event": EVT_PROACTIVE_SENT,
            "trigger_type": trigger_type,
            "topic_source": source,
            "stage": stage,
            "is_decay_final": ctx.get("is_decay_final", False),
            "message_len": len(message),
        },
    )

    await increment_proactive_count(state.agent_id, state.user_id)
    await _persist_proactive_state(
        state,
        trigger_type=trigger_type,
        message=message,
        assistant_message_id=assistant_message_id,
        cooldown=prep.cooldown,
        new_used_ids=set(ctx.get("used_memory_ids", [])),
        now_ts=now_ts,
    )
    # A 模式是一次全新开场: 旧话题翻篇, 用户回来后开启新会话 (B 名额恢复)
    await topic_continuity.close_session(prep.conversation_id)
    schedule_proactive_ai_memory(
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=prep.conversation_id,
        message=message,
    )
    return True


# ────────────────────────────────────────────────────────────────────
# Manual / triggered entry
# ────────────────────────────────────────────────────────────────────

async def _unlock_state_for_admin_test(state: ProactiveStateRecord) -> ProactiveStateRecord:
    """Admin manual trigger: reset transient blockers to idle without decay reset."""
    from dataclasses import replace

    try:
        await db.execute_raw(
            """
            UPDATE proactive_states
            SET
                status = 'idle',
                response_deadline_at = NULL,
                stop_reason = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            state.id,
        )
    except Exception as e:
        logger.warning(f"admin proactive unlock failed state={state.id[:8]}: {e}")
        return state
    return replace(state, status="idle", response_deadline_at=None, stop_reason=None)


async def send_manual_or_triggered_proactive(
    *,
    workspace_id: str,
    trigger_type: str,
    now: datetime | None = None,
    skip_limits: bool = False,
    admin_test_options: "AdminProactiveTestOptions | None" = None,
) -> dict[str, str | bool | None]:
    from app.services.proactive.admin_test import AdminProactiveSendOutcome

    state = await ensure_proactive_state_for_workspace(
        workspace_id, now=now, reason="manual_or_triggered",
    )
    if not state:
        return {
            "ok": False,
            "reason": "workspace_or_state_missing",
            "message": None,
            "web_search_used": False,
            "link_card_used": False,
        }
    if skip_limits and state.stop_reason == "silence_exhausted":
        state = await _unlock_state_for_admin_test(state)
    if state.status not in SENDABLE_PROACTIVE_STATUSES:
        if skip_limits and state.status in _ADMIN_UNLOCKABLE_STATUSES:
            state = await _unlock_state_for_admin_test(state)
        else:
            await log_proactive_event(
                state_id=state.id,
                workspace_id=state.workspace_id,
                user_id=state.user_id,
                agent_id=state.agent_id,
                conversation_id=state.conversation_id,
                event_type="send_skipped",
                trigger_type=trigger_type,
                payload={"reason": "state_not_sendable", "status": state.status},
            )
            return {
                "ok": False,
                "reason": f"state_not_sendable:{state.status}",
                "message": None,
                "web_search_used": False,
                "link_card_used": False,
            }

    outcome = AdminProactiveSendOutcome()
    sent = await generate_and_send_proactive(
        state,
        trigger_type=trigger_type,
        now=now,
        skip_limits=skip_limits,
        admin_test_options=admin_test_options,
        send_outcome=outcome,
    )
    if not sent:
        return {
            "ok": False,
            "reason": outcome.skip_reason or "generation_or_limit_blocked",
            "message": None,
            "web_search_used": outcome.web_search_used,
            "link_card_used": outcome.link_card_used,
            "link_card_skip_reason": outcome.link_card_skip_reason,
        }

    rows = await db.query_raw(
        """
        SELECT message
        FROM proactive_chat_logs
        WHERE workspace_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        state.workspace_id,
    )
    latest_message = str(rows[0]["message"]) if rows else None
    return {
        "ok": True,
        "reason": None,
        "message": latest_message,
        "web_search_used": outcome.web_search_used,
        "link_card_used": outcome.link_card_used,
        "link_card_skip_reason": outcome.link_card_skip_reason,
    }


# ────────────────────────────────────────────────────────────────────
# spec §12 开场主动第一句话
# ────────────────────────────────────────────────────────────────────

async def send_first_greeting(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    workspace_id: str | None = None,
    voice_eligible: bool = True,
) -> bool:
    """spec §12: 用户首次进入聊天 (对话消息数=0) 时 AI 主动发送第一句.

    不走时间窗概率/不计入每日 3 次上限; 但需计入衰减 n=1 —
    走与其他主动消息相同的 mark_proactive_sent 路径, 用户不回复时才
    能进入 spec §8 的三级衰减等待 (`status=waiting_user`,
    `response_deadline_at` 写入等).
    """
    # 绑 ContextVar 让 LLM 工厂应用该 agent 的模型 override.
    from app.services.runtime_config import bind_agent_context
    await bind_agent_context(agent_id)

    count = await db.message.count(where={"conversationId": conversation_id})
    if count > 0:
        return False

    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    # provisioning 期间不发: 此时 character profile / life_events / MBTI 衍生
    # 偏好都还没入库, build_personality_brief 只能拿到 7 维基础值, LLM 写出来
    # 的开场白不能反映完整人设. agents.py 在 activate_agent 完成后会显式
    # dispatch_first_greeting_for_agent 兜底触发, 不依赖前端 WS 重连
    # (chatSocket 是 module-level singleton, App remount 不会重连 WS).
    if getattr(agent, "status", "active") != "active":
        logger.info(
            f"first_greeting deferred: agent {agent_id[:8]} status="
            f"{getattr(agent, 'status', '?')}, will fire after activate_agent"
        )
        return False

    # Redis SETNX 锁防止并发触发 (e.g. WS 重连 + post-active dispatch 同时进入).
    # TTL 1 天足够覆盖 agent 的整个 onboarding, 不会因临时网络问题永久阻塞.
    redis = await get_redis()
    lock_key = f"first_greeting:fired:{conversation_id}"
    if not await redis.set(lock_key, "1", nx=True, ex=86400):
        logger.info(f"first_greeting skipped: lock held for conv={conversation_id[:8]}")
        return False

    from app.services.llm.usage_tracker import traced_usage_session
    try:
        async with traced_usage_session(
            name="[proactive:first_greeting]",
            scope="proactive", conversation_id=conversation_id,
            agent_id=agent_id, user_id=user_id,
        ) as tracer:
            try:
                tpl = await get_prompt_text("proactive.first_greeting")
            except PromptDisabledError:
                # 停用开场白模板 → 释放 NX 锁再跳过, 否则重新启用后 24h 内
                # 该会话的开场白被烧掉的锁永久吞掉.
                logger.info("first_greeting prompt disabled, skipping")
                await redis.delete(lock_key)
                return False
            prompt = tpl.format(
                ai_name=agent.name,
                personality_brief=build_personality_brief(agent),
                occupation=getattr(agent, "occupation", None) or "普通人",
            )
            message = (await invoke_text(get_chat_model(), prompt)).strip()
            if not message or len(message) < 4:
                return False

            now_ts = datetime.now(UTC)
            assistant_message_id = await emit_proactive_message(
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                message=message,
                trigger_type="first_greeting",
                skip_post_process=True,
                trace_id=tracer.safe_trace_id,
                voice_eligible=voice_eligible,
                # 开场白 = 会话里还一条用户消息都没有; 用户抢先开口了就不插到后面
                abort_if_user_replied_since=_EPOCH,
            )
            if not assistant_message_id:
                logger.info(f"first_greeting aborted: user spoke first conv={conversation_id[:8]}")
                return False

            # 接入 spec §8 衰减链路：首句仍需计入 n=1，用户不回复才会
            # 推进到第二/三阶段。
            ws_id = workspace_id or await resolve_workspace_id(
                user_id=user_id, agent_id=agent_id,
            )
            if ws_id:
                state = await ensure_proactive_state_for_workspace(
                    ws_id, reason="first_greeting",
                )
                if state is not None:
                    # spec §12.3: 首句计入 n=1, 用户未回复 24h 后 escalate 升到 2.
                    await mark_proactive_sent(
                        state,
                        trigger_type="first_greeting",
                        message=message,
                        assistant_message_id=assistant_message_id,
                        now=now_ts,
                        initial_silence_level_n=1,
                    )
                    await save_last_reply_timestamp(agent_id, user_id, when=now_ts)
            return True
    except Exception as e:
        logger.warning(f"send_first_greeting failed: {e}")
        # 锁清掉, 让用户下次 WS 重连或 admin 手动 retry 还能再试.
        try:
            await redis.delete(lock_key)
        except Exception:
            pass
        return False


async def dispatch_first_greeting_for_agent(*, agent_id: str, user_id: str) -> None:
    """activate_agent 后兜底触发: 找/建该 agent 会话, 对消息数=0 的发开场白.

    解决前端 WS singleton 不会随 App remount 重连导致 send_first_greeting 永远
    不被再次调用的问题. send_first_greeting 内部用 Redis SETNX 保证幂等.
    """
    convs = await _ensure_first_greeting_conversations(
        agent_id=agent_id,
        user_id=user_id,
    )
    if not convs:
        return
    # 一次查询 agent_name + username, 整个 dispatch 复用 — 避免每 conv 各查一次
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    user = await db.user.find_unique(where={"id": user_id})
    for conv in convs:
        # send_first_greeting 内部检查 message count > 0 → 跳过 (覆盖用户已开始
        # 聊天的边界情况) + Redis SETNX 防并发. 这里只 fire-and-forget.
        try:
            with bind_context(
                conversation_id=conv.id,
                workspace_id=getattr(conv, "workspaceId", None),
                agent_id=agent_id,
                agent_name=agent.name if agent else None,
                user_id=user_id,
                username=user.username if user else None,
            ):
                await send_first_greeting(
                    conversation_id=conv.id,
                    user_id=user_id,
                    agent_id=agent_id,
                    workspace_id=getattr(conv, "workspaceId", None),
                )
        except Exception as e:
            logger.warning(
                f"dispatch_first_greeting_for_agent: send failed for "
                f"conv={conv.id[:8]} agent={agent_id[:8]}: {e}"
            )


async def _ensure_first_greeting_conversations(
    *, agent_id: str, user_id: str
) -> list[Any]:
    """Return existing conversations, or create the default one before greeting.

    Flutter may still be on the creation progress screen when provisioning
    finishes. If no conversation exists yet, create the same default active
    workspace conversation that /conversations would create later so the first
    greeting can be generated before the user lands in chat.
    """
    try:
        convs = await db.conversation.find_many(
            where={"agentId": agent_id, "isDeleted": False},
        )
    except Exception as e:
        logger.warning(
            f"dispatch_first_greeting_for_agent: list convs failed for {agent_id[:8]}: {e}"
        )
        return []
    if convs:
        return convs

    # 查空到创建之间要串行化。这个函数由 WS 连接触发, 用户双设备登录或重连风暴会让
    # 两次调用并行 (多 worker 下是真并行), 双双查到"没有会话"再各建一个 —— 同一个
    # agent 下出现两个默认会话, 消息还会被分到两边。
    #
    # 拿不到锁就当"另一边正在建", 重查一次即可: 这里不需要抢, 只需要不重复建。
    async with distributed_lock(
        f"first_greeting_conv:{agent_id}",
        ttl_s=30,
        wait_timeout_s=5.0,
        fail_open=True,
    ) as locked:
        if locked:
            convs = await db.conversation.find_many(
                where={"agentId": agent_id, "isDeleted": False},
            )
            if convs:
                return convs
        return await _create_default_conversation(
            agent_id=agent_id, user_id=user_id,
        )


async def _create_default_conversation(
    *, agent_id: str, user_id: str
) -> list[Any]:
    """建默认会话。调用方负责先确认确实没有 (并持有锁)."""
    workspace = await get_active_workspace(user_id=user_id, agent_id=agent_id)
    if not workspace:
        logger.info(
            f"dispatch_first_greeting_for_agent: no active workspace for agent={agent_id[:8]}"
        )
        return []

    try:
        conv = await db.conversation.create(
            data={
                "user": {"connect": {"id": user_id}},
                "agent": {"connect": {"id": agent_id}},
                "workspace": {"connect": {"id": workspace.id}},
                "title": None,
            }
        )
        return [conv]
    except Exception as e:
        logger.warning(
            f"dispatch_first_greeting_for_agent: create default conv failed "
            f"agent={agent_id[:8]} workspace={workspace.id[:8]}: {e}"
        )
        try:
            existing = await db.conversation.find_first(
                where={
                    "workspaceId": workspace.id,
                    "agentId": agent_id,
                    "userId": user_id,
                    "isDeleted": False,
                },
                order={"updatedAt": "desc"},
            )
        except Exception:
            existing = None
        return [existing] if existing else []


# ────────────────────────────────────────────────────────────────────
# 后台任务
# ────────────────────────────────────────────────────────────────────

async def _bg_proactive_ai_memory(
    user_id: str, message: str,
    *, conversation_id: str, agent_id: str,
) -> None:
    """Spec §2.2：把刚发出的主动消息送进 per-message AI 自我记忆 pipeline。

    起独立 usage session, 让记忆抽取的 LLM token 也落到 llm_usage 表.
    """
    from app.services.llm.usage_tracker import usage_session
    async with usage_session(
        scope="post_process", conversation_id=conversation_id,
        agent_id=agent_id, user_id=user_id,
    ):
        try:
            from app.services.workspace.workspaces import resolve_workspace_id
            workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
            await process_memory_pipeline(
                user_id=user_id,
                new_conversation=f"assistant: {message}",
                side="ai",
                workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning(f"Proactive AI memory pipeline failed: {e}")
