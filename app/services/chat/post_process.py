"""聊天回复推送后的善后工作：持久化 + 并行后台任务。

`save_replies` 是热路径同步调用（持久化分段回复）；
`run_post_process` 是 fire-and-forget，把以下任务并行执行：

- _bg_user_emotion: 写用户情绪标签到消息 metadata（值由热路径已算好）
- _bg_memory_pipeline: spec §2.1/§2.2 记忆抽取（user + AI）
- _bg_trait_adjustment: 检测用户反馈调整 agent 性格
- _bg_positive_recovery: 正向互动 +20 耐心（spec §2.5）

`_bg_memory_pipeline` 还被 boundary_phase 通过 bg_memory_pipeline_fn 注入使用。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from prisma import Json

from app.config import settings
from app.db import db
from app.observability.events import EVT_BG_DONE
from app.services.chat.intent_replies import positive_interaction_check
from app.services.chat.reply_formatting import strip_system_markers
from app.services.interaction.boundary import (
    PATIENCE_MAX,
    check_positive_recovery,
    get_patience,
)
from app.services.memory.recording.pipeline import (
    MemoryExtractionError,
    process_memory_pipeline,
)
from app.services.memory.recording.watermark import get_watermark, set_watermark
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    DistributedLockUnavailable,
    distributed_lock,
)
from app.services.runtime.tasks import fire_background
from app.services.runtime.ws_manager import manager
from app.services.trait_adjustment import (
    apply_trait_adjustment,
    detect_direct_feedback,
    infer_feedback,
)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# Per-conversation memory pipeline lock
# ───────────────────────────────────────────────────────────────────
# 同一 conversation 的连续 batch 必须 serialize, 否则触发双层 race:
#   1. 水位线 race: 两个 batch 都读到旧 wm → 都把同一批消息当"新"抽取一遍
#   2. SINGLETON storage TOCTOU (persistence.py:168-189):
#      Task A 查 (身份/年龄) empty → Task B 也查 empty → 两边各 insert →
#      L1 重复 (生产 case 2026-05-07 19:59-20:03: 用户 30s 内连发 2 条画像
#      dump, 两个 _bg_memory_pipeline 任务 24s 重叠, 28+27 条入库且 L1
#      生日/年龄各重复 1 次).
#
# 用 dict[conv_id, asyncio.Lock] 串行同 conv 的 batch; 不同 conv 完全并行.
# 无 conv_id 入口 (proactive sender 等) 不上锁 — 那些路径不存在并发同源.
#
# 内存模型: grow-only (~100B/Lock). 1000 活跃 conv ≈ 100KB 可接受. Lock 不
# weakref 化 (WeakValueDictionary 在无人持锁时会丢对象, 下次新建的 lock 跟
# 在飞的 lock 不是同一个 → race 复现). 长期运行需 cleanup 时再加 LRU.
_pipeline_locks: dict[str, asyncio.Lock] = {}
_PIPELINE_DISTRIBUTED_LOCK_TTL = 600


def _get_pipeline_lock(conversation_id: str) -> asyncio.Lock:
    """Get or lazily create the pipeline lock for a given conversation.

    Lazy init 而非启动时全建是因为大多数 conv 短暂活跃, 大多数 lock 永远
    不会被 contend (单 batch 自然 serialize). Lazy 减少不必要的对象.
    """
    lock = _pipeline_locks.get(conversation_id)
    if lock is None:
        lock = asyncio.Lock()
        _pipeline_locks[conversation_id] = lock
    return lock


async def _resolve_workspace_id_for_conversation(conversation_id: str | None) -> str | None:
    """Best-effort conversation -> workspace lookup for memory writes.

    The chat hot path already knows the conversation scope, so background memory
    extraction must not fall back to "latest active workspace" for users who
    have multiple companions. If lookup fails, callers still degrade to the
    legacy user-level resolver inside process_memory_pipeline.
    """
    if not conversation_id:
        return None
    try:
        conv = await db.conversation.find_unique(where={"id": conversation_id})
        return getattr(conv, "workspaceId", None) if conv else None
    except Exception as e:
        logger.debug(f"conversation workspace lookup failed for {conversation_id}: {e}")
        return None


async def save_replies(
    conversation_id: str,
    replies: list[str | dict],
    trace_id: str | None = None,
    turn_user_message_ids: list[str] | None = None,
    achievement_turn_id: str | None = None,
    achievement_turn_final: bool = True,
) -> str | None:
    """spec 持久化：把分段回复写入消息表，返回首条 message_id。

    replies 元素若为 dict，除 text/index 外的字段会合并入消息 metadata
    （如 boundary/zone/attack_level/sticker_url）。
    """
    try:
        first_message_id: str | None = None
        first_created_at: datetime | None = None
        first_metadata: dict | None = None
        assistant_texts: list[str] = []
        for i, reply in enumerate(replies):
            if isinstance(reply, dict):
                text = str(reply.get("text", ""))
                metadata: dict = {"reply_index": i}
                for k, v in reply.items():
                    if k not in ("text", "index") and v is not None:
                        metadata[k] = v
            else:
                text = reply
                metadata = {"reply_index": i}
            # 防泄漏最后一道闸: 5 个直发出口已各自剥系统标记, 这里在**落库**处再兜
            # 一次. 关键不在气泡显示, 而在防污染——带 [EMO]/[X] 的 content 落库后会
            # 被下一轮历史前缀注入, 被 LLM 二次模仿, 自我强化. 剥空 (整条即标记) 则
            # 保留原文避免空 content 破坏历史 (此情形上游已换 "..."，几乎不达此处).
            _cleaned = strip_system_markers(text)
            if _cleaned:
                text = _cleaned

            # 懒触发 trace: 首条只挂 trace_id, 用户点 Trace 按钮时通过
            # /traces/resolve endpoint 触发 share + mirror 写入.
            if i == 0 and trace_id:
                metadata["trace_id"] = trace_id
            if achievement_turn_id:
                metadata["achievement_turn_id"] = achievement_turn_id

            created = await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": text,
                    "metadata": Json(metadata),
                }
            )
            if i == 0:
                try:
                    from app.services.operations.metrics import record_reply_operational_metrics

                    fire_background(record_reply_operational_metrics(
                        message_id=created.id,
                        conversation_id=conversation_id,
                        metadata=metadata,
                    ))
                except Exception as metric_err:
                    logger.debug(f"Reply operational metrics skipped: {metric_err}")
                try:
                    from app.services.notifications.service import notify_agent_message_created

                    fire_background(notify_agent_message_created(
                        conversation_id=conversation_id,
                        message_id=created.id,
                        text=text,
                        metadata=metadata,
                    ))
                except Exception as push_err:
                    logger.debug(f"[PUSH] assistant reply notification skipped: {push_err}")
            if i == 0:
                first_message_id = created.id
                first_created_at = getattr(created, "createdAt", None)
                first_metadata = metadata
            assistant_texts.append(text)
            try:
                from app.services.achievements.service import handle_assistant_message_event

                fire_background(handle_assistant_message_event(
                    conversation_id=conversation_id,
                    message_id=created.id,
                    text=text,
                    metadata=metadata,
                    occurred_at=getattr(created, "createdAt", None),
                ))
            except Exception as achievement_err:
                logger.debug(f"[ACH] assistant message hook skipped: {achievement_err}")
        if achievement_turn_final and first_message_id and assistant_texts:
            try:
                from app.services.achievements.service import handle_assistant_turn_event

                fire_background(handle_assistant_turn_event(
                    conversation_id=conversation_id,
                    message_id=first_message_id,
                    assistant_texts=assistant_texts,
                    user_message_ids=turn_user_message_ids or [],
                    turn_id=achievement_turn_id,
                    metadata=first_metadata,
                    occurred_at=first_created_at,
                ))
            except Exception as achievement_err:
                logger.debug(f"[ACH] assistant turn hook skipped: {achievement_err}")
        return first_message_id
    except Exception as e:
        logger.error(f"Failed to save replies: {e}")
        return None


async def _bg_user_emotion(
    user_message_id: str | None,
    user_emotion: dict | None,
) -> None:
    """写 LLM 算好的用户情绪标签到消息 metadata。

    用 DB 侧 jsonb merge 保留已有 client_id/component_card 等渲染字段；
    不能整列覆盖 metadata。
    """
    if not user_message_id or not user_emotion:
        return
    try:
        await db.execute_raw(
            """
            UPDATE messages
            SET metadata = COALESCE(metadata, '{}'::jsonb) || $1::jsonb
            WHERE id = $2
            """,
            json.dumps({"emotion": user_emotion}, ensure_ascii=False),
            user_message_id,
        )
        logger.info(
            f"[BG] user_emotion written to msg {user_message_id[:8]}",
            extra={
                "event": EVT_BG_DONE, "kind": "user_emotion",
                "user_message_id": user_message_id,
                "emotion": user_emotion.get("emotion"),
                "intensity": user_emotion.get("intensity"),
            },
        )
    except Exception as e:
        logger.warning(
            f"Background user emotion metadata write failed: {e}",
            extra={"event": EVT_BG_DONE, "kind": "user_emotion", "outcome": "failed",
                   "error_type": type(e).__name__},
        )


async def _bg_memory_pipeline(
    user_id: str,
    messages: list[dict],
    conversation_id: str | None = None,
    workspace_id: str | None = None,
    skip_ai_side: bool = False,
) -> None:
    """spec §2.1 / §2.2：用户侧与 AI 侧走两条独立管线，owner 由路径决定。

    取最近 6 条（3 轮 user+assistant）作为 LLM 输入窗口（解指代/情境需要多轮）。
    按 (conversation_id, side) 水位线把窗口切成【历史上下文】+【待抽取消息】:
    仅从后者抽取记忆, 前者仅供 LLM 理解; 抽完推进水位线. 两条管线水位线独立,
    同一条消息跨轮不再被重复抽取 ~3 次.

    conversation_id 为 None 时退化回老行为 (无水位线, 全部当新消息抽), 兼容
    proactive sender 等无会话上下文的入口. 每条 msg 必须含 createdAt (ISO) 才能
    参与水位线切分, 没有则归为新消息.

    抽取完成后, 若有 conversation_id 且确有新记忆入库, 通过 WS 推
    `memory_extracted` 让 admin inspector 实时刷新 (前端按当前 filter 重拉).

    并发控制: 同一 conversation 的连续 batch 串行执行 (per-conv asyncio.Lock).
    避免 batch A 在跑 ~2min extraction 时 batch B 读到旧水位线 → 重复抽取
    msg1 + 双层 race 导致 L1 SINGLETON 重复入库. 不同 conversation 完全并行.
    """
    if conversation_id is not None:
        async with _get_pipeline_lock(conversation_id):
            if not settings.is_production():
                await _do_memory_pipeline(
                    user_id, messages, conversation_id, workspace_id,
                    skip_ai_side=skip_ai_side,
                )
                return
            try:
                # 生产 fail-closed: Redis 锁不可用时**不**退回无锁执行 — 否则两
                # 实例并发跑同一 conv 管线会重现 L1 SINGLETON 重复 (2026-05-07).
                # 宁可跳过本批 (水位线未推进, 后续 batch 重新覆盖). 与 scheduler
                # 的分布式锁策略一致。
                async with distributed_lock(
                    f"memory_pipeline:{conversation_id}",
                    ttl_s=_PIPELINE_DISTRIBUTED_LOCK_TTL,
                    wait_timeout_s=_PIPELINE_DISTRIBUTED_LOCK_TTL,
                    retry_interval_s=0.5,
                    fail_open=False,
                ):
                    await _do_memory_pipeline(
                        user_id, messages, conversation_id, workspace_id,
                        skip_ai_side=skip_ai_side,
                    )
            except (DistributedLockNotAcquired, DistributedLockUnavailable) as e:
                # NotAcquired: 等满 wait_timeout 仍拿不到锁 = 另一实例的管线卡死或
                # 超长运行. Unavailable: Redis 锁后端不可用. 两者都跳过本批: 水位线
                # 未推进, 只要用户还在聊, 后续 batch 的窗口会重新覆盖这些消息;
                # 宁可少抽一批, 不可重复入库 / 跨实例并发。
                logger.error(
                    f"Memory pipeline distributed lock unavailable ({type(e).__name__}); "
                    "skipping batch (watermark not advanced, later batches re-cover)",
                    extra={
                        "event": EVT_BG_DONE,
                        "kind": "memory_pipeline",
                        "outcome": "distributed_lock_skip",
                        "error_type": type(e).__name__,
                        "conversation_id": conversation_id,
                    },
                )
    else:
        # proactive 等无 conv 入口: 调用方天然不并发同源, 无需上锁.
        await _do_memory_pipeline(
            user_id, messages, None, workspace_id,
            skip_ai_side=skip_ai_side,
        )


async def _do_memory_pipeline(
    user_id: str,
    messages: list[dict],
    conversation_id: str | None,
    workspace_id: str | None,
    *,
    skip_ai_side: bool = False,
) -> None:
    """_bg_memory_pipeline 的实现体. 由外层确保同 conv 不并发后被调用."""
    try:
        recent = messages[-6:]
        if not recent:
            return
        workspace_id = workspace_id or await _resolve_workspace_id_for_conversation(conversation_id)
        roles = {m.get("role") for m in recent}
        sides_to_run: list[tuple[Literal["user", "ai"], str]] = [
            ("user", "user"),
        ]
        if not skip_ai_side:
            sides_to_run.append(("ai", "assistant"))
        elif conversation_id:
            await _advance_side_watermark_without_extraction(
                recent, conversation_id, side="ai",
            )
        tasks = [
            _pipeline_with_watermark(
                user_id, recent, conversation_id,
                side=side, workspace_id=workspace_id,
            )
            for side, role in sides_to_run
            if role in roles
        ]
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total = sum(results)
        logger.info(
            f"[BG] memory_pipeline stored {total} memories across {len(tasks)} side(s)",
            extra={
                "event": EVT_BG_DONE, "kind": "memory_pipeline",
                "n_stored_total": total,
                "n_sides_run": len(tasks),
            },
        )
        if total > 0 and conversation_id:
            await manager.send_event(
                conversation_id,
                "memory_extracted",
                {"count": total},
            )
    except Exception as e:
        logger.error(
            f"Background memory pipeline failed: {e}",
            extra={"event": EVT_BG_DONE, "kind": "memory_pipeline", "outcome": "failed",
                   "error_type": type(e).__name__},
        )


_WATERMARK_BACKLOG_LIMIT = 30


async def _fetch_new_side_messages(
    conversation_id: str,
    wm: datetime,
    target_role: str,
) -> list[dict] | None:
    """Fetch the side's un-extracted backlog straight from the DB.

    The caller-provided window is only the last 6 messages. When a batch is
    skipped (distributed lock contention / Redis outage) while the user keeps
    chatting, more than 6 new messages can accumulate — the older ones slide
    out of the window and, once a later batch advances the watermark past
    them, are silently never extracted. Fetching `createdAt > watermark` from
    the DB makes the pipeline self-healing regardless of window size.

    Returns None on DB failure so the caller falls back to the window split
    (legacy behavior, still correct for the common no-skip case).
    """
    try:
        rows = await db.message.find_many(
            where={
                "conversationId": conversation_id,
                "role": target_role,
                "createdAt": {"gt": wm},
            },
            order={"createdAt": "asc"},
            take=_WATERMARK_BACKLOG_LIMIT,
        )
    except Exception as e:
        logger.warning(f"[MEM-WM] backlog fetch failed conv={conversation_id[:8]}: {e}")
        return None
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "createdAt": getattr(r, "createdAt", None),
        }
        for r in rows
        if (r.content or "").strip()
    ]


async def _pipeline_with_watermark(
    user_id: str,
    recent: list[dict],
    conversation_id: str | None,
    *,
    side: Literal["user", "ai"],
    workspace_id: str | None = None,
) -> int:
    """按 (conversation_id, side) 水位线切分, 调用 extraction pipeline.
    返回该侧实际入库的记忆条数 (供 _bg_memory_pipeline 汇总后推 WS 事件).

    新消息来源优先 DB 增量 (createdAt > watermark, 上限 30 条), 防止跳批期间
    积压 >6 条时旧消息滑出调用方窗口被静默丢弃; DB 失败 / 首次无水位线时退回
    旧的窗口切分行为. 调用方窗口里无时间戳的合成消息 (刚生成还未持久化的回复)
    仅在 DB 增量为空时作为抽取目标, 避免与已持久化的拆分回复重复抽取.
    """
    wm = _ensure_aware(
        await get_watermark(conversation_id, side) if conversation_id else None
    )

    # Cross-role NEW msgs go to context_msgs, not new_target_msgs — prevents
    # AI's just-generated reply from being extracted as a user fact (and vice versa).
    # target_role NEW msgs without ts (boundary short-circuit, fresh reply) use now()
    # so the watermark still advances; otherwise next round would re-extract them.
    target_role = "user" if side == "user" else "assistant"
    fallback_now = datetime.now(UTC)

    db_backlog: list[dict] | None = None
    if conversation_id and wm is not None:
        # wm None (first run / expired) keeps the window-only path: fetching the
        # whole history would re-extract an entire old conversation.
        db_backlog = await _fetch_new_side_messages(conversation_id, wm, target_role)

    context_msgs: list[dict] = []
    new_target_msgs: list[dict] = []
    max_side_ts: datetime | None = None

    def _add_target(m: dict) -> None:
        nonlocal max_side_ts
        new_target_msgs.append(m)
        ts = _parse_ts(m)
        effective = ts if ts is not None else fallback_now
        if max_side_ts is None or effective > max_side_ts:
            max_side_ts = effective

    if db_backlog is not None:
        backlog_contents = {(m.get("content") or "").strip() for m in db_backlog}
        for m in db_backlog:
            _add_target(m)
        # The window contributes context plus synthetic (un-persisted) targets.
        for m in recent:
            ts = _parse_ts(m)
            is_new_target = m.get("role") == target_role and (ts is None or ts > wm)
            if not is_new_target:
                context_msgs.append(m)
            elif ts is None and (m.get("content") or "").strip() not in backlog_contents:
                # Synthetic fresh reply (just-generated, no persisted ts) that the
                # DB backlog snapshot does NOT already contain — extract it now
                # instead of deferring a whole turn. Content-dedupe against the
                # backlog prevents double-extracting a reply that WAS persisted
                # before this snapshot (then it's authoritative via db_backlog).
                _add_target(m)
    else:
        for m in recent:
            ts = _parse_ts(m)
            is_new = wm is None or ts is None or ts > wm
            if is_new and m.get("role") == target_role:
                _add_target(m)
            else:
                context_msgs.append(m)

    if max_side_ts is None:
        return 0  # 该 side 无新消息, 跳过 LLM
    evidence_message_ids = [
        str(m.get("id"))
        for m in new_target_msgs
        if m.get("id")
    ]

    try:
        stored_ids = await process_memory_pipeline(
            user_id=user_id,
            new_conversation=_fmt_conversation(new_target_msgs),
            context_conversation=_fmt_conversation(context_msgs),
            statement_time=max_side_ts,
            side=side,
            workspace_id=workspace_id,
            evidence_message_ids=evidence_message_ids,
        )
    except MemoryExtractionError as e:
        # 抽取没有真正跑完 (LLM 失败, 或 admin 停用了抽取模板): 水位线原地不动,
        # 让后续批次重新覆盖这些消息, 而不是静默丢掉.
        #
        # 两种成因的运维含义完全不同 —— LLM 失败会自愈, 模板停用要人去开回来,
        # 所以异常里带 kind 并原样打进日志, 不要笼统说"失败".
        logger.warning(
            f"[BG] {e}; watermark held for retry",
            extra={"event": EVT_BG_DONE, "kind": "memory_pipeline",
                   "outcome": "extraction_error_watermark_held", "side": side},
        )
        return 0

    # 防时钟回退: 仅当新候选 > wm 才推进 (且抽取成功, 见上方 except)
    if conversation_id and (wm is None or max_side_ts > wm):
        await set_watermark(conversation_id, side, max_side_ts)

    return len(stored_ids)


async def _advance_side_watermark_without_extraction(
    recent: list[dict],
    conversation_id: str,
    *,
    side: Literal["user", "ai"],
) -> None:
    """Advance watermark for a side intentionally skipped from extraction.

    Schedule/current-state replies are ephemeral answers. If we skip AI memory
    extraction without moving the AI watermark, a later non-skipped turn would
    see this assistant reply as "new" and extract it retroactively.
    """
    target_role = "user" if side == "user" else "assistant"
    fallback_now = datetime.now(UTC)
    max_side_ts: datetime | None = None
    for msg in recent:
        if msg.get("role") != target_role:
            continue
        effective = _parse_ts(msg) or fallback_now
        if max_side_ts is None or effective > max_side_ts:
            max_side_ts = effective
    if max_side_ts is None:
        return

    wm = _ensure_aware(await get_watermark(conversation_id, side))
    if wm is None or max_side_ts > wm:
        await set_watermark(conversation_id, side, max_side_ts)


def _parse_ts(m: dict) -> datetime | None:
    """规范化 message.createdAt → tz-aware. naive 假定 UTC, 防 'can't compare
    offset-naive and offset-aware datetimes' (Prisma client 某些版本返 naive)."""
    from app.services.schedule_domain.time_service import ensure_aware
    ts = m.get("createdAt")
    if isinstance(ts, datetime):
        return ensure_aware(ts)
    if not isinstance(ts, str):
        return None
    try:
        return ensure_aware(datetime.fromisoformat(ts))
    except ValueError:
        return None


# 测试期望的兼容 alias (test_post_process_datetime_aware_normalize 用)
def _ensure_aware(dt: datetime | None) -> datetime | None:
    from app.services.schedule_domain.time_service import ensure_aware
    return ensure_aware(dt)


def _fmt_conversation(msgs: list[dict]) -> str:
    """Format messages as `role: content` lines for the extraction prompt.

    Only `user`/`assistant` roles are emitted. A message missing a valid role
    is skipped rather than defaulted to `user` — mislabeling an AI/system line
    as the user would let it be extracted as a user fact (owner corruption).
    """
    lines: list[str] = []
    for m in msgs:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _bg_trait_adjustment(agent_id: str, user_message: str) -> None:
    """检测用户反馈信号 → 调整 agent 性格特征。"""
    try:
        adjustments = detect_direct_feedback(user_message) or infer_feedback(user_message)
        if adjustments:
            await apply_trait_adjustment(agent_id, adjustments)
            logger.info(
                f"[BG] trait_adjustment applied n={len(adjustments)}",
                extra={"event": EVT_BG_DONE, "kind": "trait_adjustment",
                       "n_adjustments": len(adjustments)},
            )
        else:
            logger.debug(
                "[BG] trait_adjustment: no feedback signal detected",
                extra={"event": EVT_BG_DONE, "kind": "trait_adjustment", "outcome": "no_signal"},
            )
    except Exception as e:
        logger.warning(
            f"Background trait adjustment failed: {e}",
            extra={"event": EVT_BG_DONE, "kind": "trait_adjustment", "outcome": "failed",
                   "error_type": type(e).__name__},
        )


async def _bg_positive_recovery(
    agent_id: str, user_id: str, user_message: str,
) -> None:
    """spec §2.5：正向互动 +20 耐心 (仅对感谢/善意/积极反馈/正向情绪类消息生效).

    LLM 语义判定门: 防中性应答 (嗯/哦/好) + 普通问询滥发 +20, 后者会等价为
    "3 倍速自然恢复". LLM 判定失败 → 保守不发放, 走自然 +10/h 路径.

    优化: 患者 patience 已满或拉黑时, +20 必然 no-op (check_positive_recovery
    内部会 early-return), 跳过 LLM 调用省 ~200ms qwen-flash 成本.
    """
    try:
        patience = await get_patience(agent_id, user_id)
        if patience >= PATIENCE_MAX or patience <= 0:
            logger.debug(
                f"[BG] positive_recovery skipped: patience={patience}",
                extra={"event": EVT_BG_DONE, "kind": "positive_recovery",
                       "outcome": "skipped_patience_extreme", "patience": patience},
            )
            return
        if not await positive_interaction_check(user_message):
            logger.debug(
                "[BG] positive_recovery skipped: not positive signal",
                extra={"event": EVT_BG_DONE, "kind": "positive_recovery",
                       "outcome": "skipped_not_positive"},
            )
            return
        new_patience = await check_positive_recovery(agent_id, user_id)
        logger.info(
            f"[BG] positive_recovery applied: {patience} → {new_patience}",
            extra={"event": EVT_BG_DONE, "kind": "positive_recovery",
                   "patience_before": patience, "patience_after": new_patience},
        )
    except Exception as e:
        logger.warning(
            f"Background positive recovery failed: {e}",
            extra={"event": EVT_BG_DONE, "kind": "positive_recovery", "outcome": "failed",
                   "error_type": type(e).__name__},
        )


async def run_post_process(
    *,
    user_id: str,
    agent_id: str | None,
    conversation_id: str,
    user_message: str,
    user_message_id: str | None,
    full_response: str,
    messages_dicts: list[dict],
    user_emotion: dict | None = None,
    skip_ai_memory: bool = False,
    workspace_id: str | None = None,
) -> None:
    """后台任务并发：写用户情绪 / 记忆抽取 / 性格反馈 / 耐心恢复。

    起独立 usage session: fire_background 已把 ContextVar 隔离,
    这里重新开让记忆/trait 的 token 落到 llm_usage 自己一行 (scope=post_process).

    workspace_id: 热路径已知的会话 workspace, 直接透传给记忆管线, 避免
    _do_memory_pipeline 回退到 "latest active workspace" (多伴侣用户会错库)。
    """
    from app.services.llm.usage_tracker import usage_session
    async with usage_session(
        scope="post_process", conversation_id=conversation_id,
        agent_id=agent_id, user_id=user_id,
    ):
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]
        tasks: list[Any] = [
            _bg_user_emotion(user_message_id, user_emotion),
            _bg_memory_pipeline(
                user_id,
                full_messages,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                skip_ai_side=skip_ai_memory,
            ),
        ]
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id, user_message))
            # E3 表达学习: 计数节流, 每 LEARN_EVERY_N 条用户消息批量学一次
            tasks.append(_bg_expression_learning(agent_id, user_id, full_messages))
        await asyncio.gather(*tasks, return_exceptions=True)


async def _bg_expression_learning(
    agent_id: str, user_id: str, messages: list[dict],
) -> None:
    """E3 表达学习后台任务: 节流计数, 到批次阈值才调 LLM 提取.

    单条消息成本为 1 次 Redis INCR; 每 LEARN_EVERY_N 条才有 1 次小模型调用.
    失败静默 — 表达学习是增强项, 不该影响其他后台任务.
    """
    try:
        from app.services.chat.expression_learner import (
            bump_message_counter,
            learn_expressions,
        )
        if await bump_message_counter(agent_id, user_id):
            await learn_expressions(agent_id, user_id, messages)
    except Exception as e:
        logger.debug(f"expression learning skipped: {e}")
