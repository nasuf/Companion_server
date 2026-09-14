"""主动消息生成与发送入口.

按职责拆分:
  _check_send_eligibility   日限/二日限/workspace/conversation 检查
  _resolve_conversation_id  从 state 推出最终 conversation_id
  _apply_memory_cooldown    spec §9 -1/+50 冷却语义
  _generate_message         按 (trigger_type, source, decay_final) 分发 7 个 prompt
  _persist_proactive_state  调 mark_proactive_sent + save_last_reply_timestamp
  generate_and_send_proactive  主流程编排 (上述 5 段 + emit + bg AI 自我记忆 pipeline)

公共持久化与 WS 广播在 emit.py.
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
from app.services.schedule_domain.time_service import _now_corrected
from app.services.proactive.policy import select_topic_source, select_topic_theme
from app.services.relationship.emotion import emotion_to_tone
from app.services.workspace.workspaces import (
    get_active_workspace,
    get_workspace_by_id,
    resolve_workspace_id,
)
from app.services.proactive.state import (
    ProactiveStateRecord,
    determine_proactive_stage,
    ensure_proactive_state_for_workspace,
    get_active_workspace_context,
    log_proactive_event,
    mark_proactive_sent,
)
from app.services.prompting.store import PromptDisabledError, get_prompt_text
from app.services.prompting.utils import render_template
from app.services.interaction.reply_context import save_last_reply_timestamp

logger = logging.getLogger(__name__)

UTC = timezone.utc
SENDABLE_PROACTIVE_STATUSES = {"idle"}
# Admin QA (skip_limits): force-send by resetting transient blockers to idle
# without touching decay counters (silence_level_n / followup_plan_type).
_ADMIN_UNLOCKABLE_STATUSES = frozenset({
    "waiting_user",
    "running",
    "processing",
    "processing_timeout",
})

_MEMORY_SOURCES = frozenset({"ai_l1", "ai_l2", "user_l1", "user_l2", "relationship"})


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

def _build_personality_brief(agent) -> str:
    """从 agent 7 维性格导出简短描述, 给 prompt 用."""
    try:
        p = getattr(agent, "personality", None) or {}
        if not isinstance(p, dict) or not p:
            return "温和友善"
        parts: list[str] = []
        if p.get("liveliness", 50) >= 70:
            parts.append("活泼")
        elif p.get("liveliness", 50) <= 30:
            parts.append("安静")
        if p.get("humor", 50) >= 70:
            parts.append("幽默")
        if p.get("rationality", 50) >= 70:
            parts.append("理性")
        if p.get("sensitivity", 50) >= 70:
            parts.append("感性")
        if p.get("planning", 50) >= 70:
            parts.append("计划性强")
        if p.get("spontaneity", 50) >= 70:
            parts.append("随性")
        if p.get("imagination", 50) >= 70:
            parts.append("脑洞大")
        return "、".join(parts) if parts else "温和友善"
    except Exception:
        return "温和友善"


# (trigger_type, source) → prompt key
_PROMPT_KEY_BY_SOURCE: dict[tuple[str, str], str] = {
    ("silence_wakeup", "ai_l1"): "proactive.silence_ai_memory",
    ("silence_wakeup", "ai_l2"): "proactive.silence_ai_memory",
    ("silence_wakeup", "user_l1"): "proactive.silence_user_memory",
    ("silence_wakeup", "user_l2"): "proactive.silence_user_memory",
    ("silence_wakeup", "ai_schedule"): "proactive.silence_schedule",
    ("silence_wakeup", "greeting"): "proactive.silence_plain",
    ("silence_wakeup", "music"): "music.proactive_recommend",
    ("memory_proactive", "ai_l1"): "proactive.memory_ai",
    ("memory_proactive", "ai_l2"): "proactive.memory_ai",
    ("memory_proactive", "user_l1"): "proactive.memory_user",
    ("memory_proactive", "user_l2"): "proactive.memory_user",
    # Phase 2 关系记忆: 共同经历 (memories_ai 生活/交互) 走 AI 记忆模板 —
    # 素材本来就是 AI 第一人称叙述的"我和用户…", memory_ai 模板语气吻合.
    ("memory_proactive", "relationship"): "proactive.memory_ai",
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


def _format_v3_trending_item(item: dict) -> str:
    """V3 三档 prompt 里 {trending_item} 占位的渲染 —— 单条内容, 而非 V0 的多条汇总.
    保留 title/snippet/platform 供 LLM 理解, 但压紧一行避免 prompt 过长.
    """
    if not isinstance(item, dict):
        return str(item or "")
    title = str(item.get("title") or "").strip()
    snippet = str(item.get("snippet") or "").strip()[:160]
    plat = str(item.get("platform") or "").strip()
    plat_prefix = f"[{plat}] " if plat else ""
    if title and snippet:
        return f"{plat_prefix}{title} —— {snippet}"
    return f"{plat_prefix}{title or snippet}"


async def _generate_message(ctx: dict) -> str | None:
    """spec §4 按 (trigger_type, source) 分发到 7 个专属 prompt;
    spec §8.5 衰减最后一次优先 decay_final.
    """
    agent = ctx["agent"]
    trigger_type = ctx["trigger_type"]
    source = ctx.get("source") or "greeting"
    personality_brief = _build_personality_brief(agent)

    # V3 dispatch (2026-09-14): 若上游分类器 (topic_source.classify_topic_source)
    # 已经把话题源判出来并塞进 ctx["topic_source_kind"] + 选好条 ctx["topic_source_item"],
    # 走对应的三档独立 prompt (proactive.trending_user_interest / _ai_persona /
    # _socially_hot) 而非 V0 的 append_trending_section 尾追. 详见
    # app/services/proactive/topic_source.py 与 evals/proactive_naturalness.
    #
    # topic_source_kind='none' 或未设 → 落回 V0 逻辑, 保持向后兼容.
    v3_source_kind = ctx.get("topic_source_kind")
    v3_source_item = ctx.get("topic_source_item")
    v3_prompt_keys = {
        "user_interest_match": "proactive.trending_user_interest",
        "ai_persona_match":    "proactive.trending_ai_persona",
        "socially_hot":        "proactive.trending_socially_hot",
    }

    try:
        if ctx.get("is_decay_final"):
            tpl = await get_prompt_text("proactive.decay_final")
            prompt = tpl.format(personality_brief=personality_brief)
        elif v3_source_kind in v3_prompt_keys and v3_source_item:
            # V3 路径
            key = v3_prompt_keys[v3_source_kind]
            tpl = await get_prompt_text(key)
            # 三档 prompt 都有的公共占位符
            fields = {
                "personality_brief": personality_brief,
                "current_mood": emotion_to_tone(ctx.get("emotion")),
                "trending_item": _format_v3_trending_item(v3_source_item),
            }
            if v3_source_kind == "user_interest_match":
                fields["user_portrait"] = str(ctx.get("user_portrait") or "(未知)")
            prompt = tpl.format(**fields)
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
        # admin 停用该主动消息模板 → 按"本次未生成"处理 (return None),
        # 让上游正常推进窗口, 不能让异常卡死 proactive 状态机.
        logger.info(f"Proactive prompt disabled, skipping: {e}")
        return None

    # V0 尾追: 只在 V3 未走时 append (v3 已经把 item 塞进 prompt 里, 别重复注入)
    if not (v3_source_kind in v3_prompt_keys and v3_source_item):
        trending_context = (ctx.get("trending_context") or "").strip()
        if trending_context:
            from app.services.proactive.trending_context import append_trending_section

            prompt = await append_trending_section(prompt, trending_context)

    response = (await invoke_text(get_chat_model(), prompt)).strip()
    if response == "SKIP" or len(response) < 4:
        return None
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

async def generate_and_send_proactive(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    now: datetime | None = None,
    skip_limits: bool = False,
    admin_test_options: "AdminProactiveTestOptions | None" = None,
    send_outcome: "AdminProactiveSendOutcome | None" = None,
) -> bool:
    from app.services.proactive.trending_context import resolve_trending_context
    from app.services.proactive.trending_gate import should_attach_trending_link_card
    # 绑 ContextVar 让本调用栈的 LLM 工厂应用该 agent 的模型 override.
    # 不绑的话主动消息生成 / AI 自我记忆抽取都会用 system 全局, 跟 chat 路径
    # 的 per-agent 行为不一致, 同时 token stats 会把这些 LLM 调用归到全局模型名.
    from app.services.runtime_config import bind_agent_context
    await bind_agent_context(state.agent_id)

    now_ts = now or datetime.now(UTC)

    prep = await _check_send_eligibility(
        state,
        trigger_type,
        skip_limits=skip_limits,
        send_outcome=send_outcome,
    )
    if prep is None:
        return False

    # spec §2.2: 话题亲密度可变, 触发前实时算; state.stage 仅在 session
    # start/restart 时持久化, 不追踪中途 intimacy 升级.
    stage = await determine_proactive_stage(state.agent_id, state.user_id)

    # spec §3.2 话题方向 + §4.1/§4.2 来源概率表
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
    if source == "music":
        source = await _prepare_music_recommendation_source(
            ctx,
            conversation_id=prep.conversation_id,
        )
        if source == "music_skip_not_idle":
            await _log_skip(
                state,
                trigger_type,
                "music_source_not_idle",
                conversation_id=prep.conversation_id,
                send_outcome=send_outcome,
            )
            return False
        ctx["source"] = source

    # spec §4.1 沉默唤醒兜底; §4.2 记忆主动失败时取消
    if source in _MEMORY_SOURCES and not ctx.get("proactive_memories"):
        if trigger_type == "silence_wakeup":
            source = "greeting"
            ctx["source"] = "greeting"
            ctx["scene_hint"] = "优先用轻量、低打扰的方式重新建立联系。"
        else:
            await _log_skip(
                state,
                trigger_type,
                "memory_source_empty",
                conversation_id=prep.conversation_id,
                extra={"source": source},
                send_outcome=send_outcome,
            )
            return False

    # spec §8.5 衰减最后一次
    ctx["is_decay_final"] = state.followup_plan_type == "thirty_day_final"

    trending_text, trending_attached, _trending_meta = await resolve_trending_context(
        trigger_type,
        topic=topic_theme,
        admin_test_options=admin_test_options,
    )
    ctx["trending_context"] = trending_text

    # V3 三档分发 (2026-09-14): trending 命中 → 分类器决定档位 + 挑候选内容.
    # 已从 env flag 灰度转为**默认路径** —— 卡片同源硬耦合 (下面 preselected_item)
    # 与消息 prompt 分档一起构成完整闭环, 不再需要 flag.
    # 决策证据: evals/proactive_naturalness (V0 source_fit 17%/mentions_card 67%
    # → V3 50%/100%). 关闭整个 trending 用 admin UI SystemConfig.proactive_trending_enabled=False.
    # 分类器返 "none" (所有候选都被黑名单/无匹配拒了) 时 ctx 不 set,
    # _generate_message 落回 V0 append_trending_section 兜底路径.
    if (
        trending_attached
        and _trending_meta is not None
        and _trending_meta.candidates
    ):
        from app.services.proactive.topic_source import classify_topic_source

        cls = classify_topic_source(
            trending_candidates=list(_trending_meta.candidates),
            user_portrait=str(ctx.get("user_portrait") or ""),
            agent=ctx.get("agent"),
        )
        if cls.kind != "none":
            ctx["topic_source_kind"] = cls.kind
            ctx["topic_source_item"] = cls.selected_candidate
            logger.info(
                f"[V3-DISPATCH] source={cls.kind} reason={cls.reason}"
            )

    # 主动消息也开 LangSmith trace + usage_session, 名字 [proactive:trigger_type]
    # 方便 LangSmith 看板与统计 dashboard 区分被动回复.
    from app.services.llm.usage_tracker import traced_usage_session
    async with traced_usage_session(
        name=f"[proactive:{trigger_type}]",
        scope="proactive", conversation_id=prep.conversation_id,
        agent_id=state.agent_id, user_id=state.user_id,
    ) as tracer:
        message = await _generate_message(ctx)
        if not message:
            await _log_skip(
                state,
                trigger_type,
                "empty_or_skip",
                conversation_id=prep.conversation_id,
                send_outcome=send_outcome,
            )
            return False

        extra_metadata: dict[str, Any] = {"stage": stage}
        ws_payload_extra: dict[str, Any] | None = None
        if source == "music" and ctx.get("music_track") is not None:
            from app.services.music_chat import card_from_track

            card = card_from_track(
                ctx["music_track"],
                intent="recommend",
                source="proactive",
            )
            extra_metadata.update({
                "component_card": card,
                "music_proactive": True,
                "topic_source": "music",
            })
            ws_payload_extra = {"component_card": card}
        proactive_link = None
        if ws_payload_extra is None:
            from app.services.chat_links import maybe_prepare_proactive_link_recommendation

            force_link = False
            skip_link = False
            if admin_test_options is not None:
                force_link = admin_test_options.use_link_card
                skip_link = not admin_test_options.use_link_card
            elif trending_attached:
                force_link = should_attach_trending_link_card(
                    trending_attached=True,
                )

            # V3 消息-卡片硬耦合 (2026-09-14): 若 V3 分类器已挑好那条内容
            # (ctx["topic_source_item"]), 卡片必须挂那一条, 不许再独立搜 tavily
            # 拿随机 URL. 修根本问题: 截图里"文本说银锁骨链 + 卡说机场"这种同一
            # 主动消息里语义脱钩的现象.
            preselected = ctx.get("topic_source_item") if ctx.get("topic_source_kind") else None
            proactive_link = await maybe_prepare_proactive_link_recommendation(
                user_id=state.user_id,
                conversation_id=prep.conversation_id,
                trigger_type=trigger_type,
                source=source,
                topic=ctx.get("topic_theme"),
                stage=stage,
                message=message,
                force=force_link,
                skip=skip_link,
                preselected_item=preselected,
            )
            if proactive_link is not None:
                extra_metadata.update({
                    "component_card": proactive_link.component_card,
                    "link_card": proactive_link.link_card_metadata,
                    "link_proactive": True,
                    "topic_source": "link",
                })
                ws_payload_extra = {"component_card": proactive_link.component_card}

        if send_outcome is not None:
            send_outcome.web_search_used = trending_attached
            send_outcome.link_card_used = proactive_link is not None
            send_outcome.extra.update({
                "trigger_type": trigger_type,
                "source": source,
            })

        assistant_message_id = await emit_proactive_message(
            conversation_id=prep.conversation_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            workspace_id=state.workspace_id,
            message=message,
            trigger_type=trigger_type,
            extra_metadata=extra_metadata,
            ws_payload_extra=ws_payload_extra,
            trace_id=tracer.safe_trace_id,
        )
        if proactive_link is not None:
            from app.services.chat_links import bind_link_card_to_message

            await bind_link_card_to_message(
                link_id=proactive_link.link.id,
                message_id=assistant_message_id,
                user_id=state.user_id,
                conversation_id=prep.conversation_id,
            )
        if source == "music" and ctx.get("music_track") is not None:
            from app.services import music
            from app.models.music import MusicTrackPayload
            from app.services.music_status import persist_and_emit_music_status

            proactive_track = ctx["music_track"]

            await music.start_co_listening(
                user_id=state.user_id,
                agent_id=state.agent_id,
                conversation_id=prep.conversation_id,
                workspace_id=state.workspace_id,
                payload=MusicTrackPayload(
                    id=proactive_track.id,
                    title=proactive_track.title,
                    artist=proactive_track.artist,
                    album=proactive_track.album,
                    library=proactive_track.library,
                    url=proactive_track.url,
                    duration_sec=proactive_track.duration_sec,
                    cover_key=proactive_track.cover_key,
                    accent_a=proactive_track.accent_a,
                    accent_b=proactive_track.accent_b,
                    source=proactive_track.source,
                    metadata=proactive_track.metadata,
                ),
                initiated_by="agent",
                status="active",
                position_seconds=0,
                is_playing=False,
            )
            await persist_and_emit_music_status(
                conversation_id=prep.conversation_id,
                status="started",
                track=proactive_track,
                actor="agent",
                actor_name=getattr(ctx.get("agent"), "name", None) or "我",
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

    asyncio.create_task(_bg_proactive_ai_memory(
        state.user_id, message,
        conversation_id=prep.conversation_id,
        agent_id=state.agent_id,
    ))
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
    # 偏好都还没入库, _build_personality_brief 只能拿到 7 维基础值, LLM 写出来
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
                personality_brief=_build_personality_brief(agent),
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
            )

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
