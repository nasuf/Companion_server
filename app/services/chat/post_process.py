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
import re
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
from app.services.memory.recording.outcomes import MemoryProgressError, MemorySideOutcome
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
# 多 worker 下这是**每进程一份**: 它只负责本进程内同一 conversation 的串行化。
# 跨进程的互斥由下面的 memory_pipeline:{conv} 分布式锁负责, 两层缺一不可 ——
# 只有本地锁会在多实例下重现 L1 SINGLETON 重复, 只有分布式锁则每个 batch 都要多
# 一次 Redis 往返。
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


async def _resolve_workspace_id_for_conversation(
    conversation_id: str, user_id: str, workspace_id: str | None,
) -> str:
    """Bind extraction to its conversation; lookup failure must not select another Agent."""
    try:
        conv = await db.conversation.find_unique(where={"id": conversation_id})
    except Exception as exc:
        raise MemoryProgressError("scope_unavailable") from exc
    actual = getattr(conv, "workspaceId", None)
    if (conv is None or getattr(conv, "isDeleted", False)
            or getattr(conv, "userId", None) != user_id
            or not isinstance(actual, str) or not actual
            or (workspace_id is not None and actual != workspace_id)):
        raise MemoryProgressError("scope_invalid")
    return actual


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
        from app.services.chat.legacy_completion import current_completion
        completion = current_completion.get()
        if completion is not None:
            completion.record(conversation_id, first_message_id, achievement_turn_id, turn_user_message_ids)
        if first_message_id:
            try:
                from app.services.offline.activity_companion import note_passive_reply

                fire_background(note_passive_reply(conversation_id))
            except Exception as companion_err:
                logger.debug(
                    "[offline-companion] passive reply hook skipped: %s",
                    companion_err,
                )
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
) -> list[MemorySideOutcome]:
    """Serialize the dual-side extraction and retain bounded, content-free diagnostics.

    These observations are not durable work receipts. In particular a failed
    write may have committed some memories, so this function enables no replay.
    """
    from app.services.memory.recording.diagnostics import observe_memory_run

    async with observe_memory_run(conversation_id) as observation:
        try:
            if conversation_id is None:
                results = await _do_memory_pipeline(user_id, messages, None, workspace_id,
                    skip_ai_side=skip_ai_side)
            else:
                async with _get_pipeline_lock(conversation_id):
                    if settings.is_production():
                        # Production fail-closed: a lock outage must never fall back to
                        # unlocked extraction across API workers.
                        async with distributed_lock(
                            f"memory_pipeline:{conversation_id}",
                            ttl_s=_PIPELINE_DISTRIBUTED_LOCK_TTL,
                            wait_timeout_s=_PIPELINE_DISTRIBUTED_LOCK_TTL,
                            retry_interval_s=0.5, fail_open=False,
                        ):
                            results = await _do_memory_pipeline(user_id, messages,
                                conversation_id, workspace_id, skip_ai_side=skip_ai_side)
                    else:
                        results = await _do_memory_pipeline(user_id, messages,
                            conversation_id, workspace_id, skip_ai_side=skip_ai_side)
        except (DistributedLockNotAcquired, DistributedLockUnavailable):
            results = [MemorySideOutcome(side, "failed", "lock_unavailable") for side in ("user", "ai")]
        except Exception as exc:
            reason = exc.reason if isinstance(exc, MemoryProgressError) else "processing_failed"
            results = [MemorySideOutcome(side, "failed", reason) for side in ("user", "ai")]
        observation.extend(results or [])
        incomplete = len(results) != 2 or any(r.status in {"failed", "disabled"} for r in results)
        total = sum(r.memory_count or 0 for r in results)
        # Redis is also the telemetry store. Keep a separate safe log so a
        # lock/progress outage cannot erase its own failure evidence.
        logger.log(logging.WARNING if incomplete else logging.INFO,
            "Memory pipeline finished", extra={"event": EVT_BG_DONE,
                "kind": "memory_pipeline", "outcome": "incomplete" if incomplete else "completed",
                "n_stored_total": total if all(r.memory_count is not None for r in results) else None,
                "n_confirmed_returned": total, "n_sides_run": len(results),
                "side_outcomes": [{"side": r.side, "status": r.status, "reason": r.reason,
                                   "progress": r.progress} for r in results]})
        return results


async def _run_memory_side(
    user_id: str, recent: list[dict], conversation_id: str | None, *,
    side: Literal["user", "ai"], workspace_id: str | None, excluded: bool = False,
) -> MemorySideOutcome:
    try:
        if excluded:
            progress = await _advance_side_watermark_without_extraction(recent, conversation_id, side=side)
            return MemorySideOutcome(side, "excluded", "intentional_skip", 0, progress)
        return await _pipeline_with_watermark(user_id, recent, conversation_id,
            side=side, workspace_id=workspace_id)
    except Exception as exc:
        reason = exc.reason if isinstance(exc, MemoryProgressError) else "processing_failed"
        if reason == "watermark_write_unavailable":
            # Normal extraction catches this with its known count. Here it is
            # an excluded/empty batch, but the Redis write may have committed.
            return MemorySideOutcome(side, "failed", reason, 0, "unknown")
        return MemorySideOutcome(side, "failed", reason)


async def _do_memory_pipeline(
    user_id: str,
    messages: list[dict],
    conversation_id: str | None,
    workspace_id: str | None,
    *,
    skip_ai_side: bool = False,
) -> list[MemorySideOutcome]:
    """Keep the conversation lock until both sides have stopped, including cancellation."""
    recent = messages[-6:]
    if conversation_id:
        workspace_id = await _resolve_workspace_id_for_conversation(conversation_id, user_id, workspace_id)
    tasks = [asyncio.create_task(_run_memory_side(user_id, recent, conversation_id,
        side=side, workspace_id=workspace_id, excluded=side == "ai" and skip_ai_side))
        for side in ("user", "ai")]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    total = sum(r.memory_count or 0 for r in results)
    if total > 0 and conversation_id:
        try:
            await manager.send_event(conversation_id, "memory_extracted", {"count": total})
        except Exception:
            logger.warning("Memory refresh notification failed after processing")
    return results


_WATERMARK_BACKLOG_LIMIT = 30
# Real-world gift messages are history context, never extraction targets: the
# gift lifecycle writes 生活/馈赠 memories itself (offline/gift_memory.py), and the
# system-written receipt/thanks cards are not the user's own words.
_LIFECYCLE_MEMORY_REAL_WORLD_TYPES = frozenset({"gift"})


def _memory_written_by_lifecycle(m: dict) -> bool:
    return (m.get("metadata") or {}).get("real_world_type") in _LIFECYCLE_MEMORY_REAL_WORLD_TYPES


async def _fetch_new_side_messages(
    conversation_id: str,
    wm: datetime,
    target_role: str,
) -> list[dict]:
    """Fetch the side's un-extracted backlog straight from the DB.

    The caller-provided window is only the last 6 messages. When a batch is
    skipped (distributed lock contention / Redis outage) while the user keeps
    chatting, more than 6 new messages can accumulate — the older ones slide
    out of the window and, once a later batch advances the watermark past
    them, are silently never extracted. Fetching `createdAt > watermark` from
    the DB makes the pipeline self-healing regardless of window size.

    A failed read must hold progress: a shorter fallback window could skip
    older unprocessed rows forever. The lookahead prevents splitting a group
    with an identical timestamp, since this legacy cursor has no ID tiebreaker.
    """
    try:
        rows = await db.message.find_many(
            where={
                "conversationId": conversation_id,
                "role": target_role,
                "createdAt": {"gt": wm},
            },
            order=[{"createdAt": "asc"}, {"id": "asc"}],
            take=_WATERMARK_BACKLOG_LIMIT + 1,
        )
    except Exception as e:
        raise MemoryProgressError("backlog_unavailable") from e
    if len(rows) > _WATERMARK_BACKLOG_LIMIT:
        boundary = rows[_WATERMARK_BACKLOG_LIMIT].createdAt
        rows = [row for row in rows[:_WATERMARK_BACKLOG_LIMIT] if row.createdAt < boundary]
        if not rows:
            raise MemoryProgressError("timestamp_page_boundary")
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "createdAt": getattr(r, "createdAt", None),
            "metadata": getattr(r, "metadata", None),
        }
        for r in rows
    ]


async def _pipeline_with_watermark(
    user_id: str,
    recent: list[dict],
    conversation_id: str | None,
    *,
    side: Literal["user", "ai"],
    workspace_id: str | None = None,
) -> MemorySideOutcome:
    """按 (conversation_id, side) 水位线切分, 调用 extraction pipeline.
    返回明确结果；返回的记忆ID数量包含更新，并非新建行数。

    新消息来源优先 DB 增量 (createdAt > watermark, 上限 30 条)。
    DB 失败保持进度；首次无水位线只处理窗口，不回填历史。 调用方窗口里无时间戳的合成消息 (刚生成还未持久化的回复)
    仅在 DB 增量为空时作为抽取目标, 避免与已持久化的拆分回复重复抽取.
    """
    # An older AI job can finish after an administrator claims the chat. Its
    # backlog/context must never turn later human messages into AI memories.
    recent = [m for m in recent if not (m.get("metadata") or {}).get("human_takeover")
              and not (m.get("metadata") or {}).get("memory_confirmation")]
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
        metadata = m.get("metadata") or {}
        excluded = (
            metadata.get("human_takeover")
            or metadata.get("memory_confirmation")
            or _memory_written_by_lifecycle(m)
        )
        if not excluded and (m.get("content") or "").strip():
            new_target_msgs.append(m)
        ts = _parse_ts(m)
        effective = ts if ts is not None else fallback_now
        if max_side_ts is None or effective > max_side_ts:
            max_side_ts = effective

    if db_backlog is not None:
        backlog_text = _comparable_text("".join(m.get("content") or "" for m in db_backlog))
        for m in db_backlog:
            _add_target(m)
        # The window contributes context plus synthetic (un-persisted) targets.
        for m in recent:
            ts = _parse_ts(m)
            is_new_target = m.get("role") == target_role and (ts is None or ts > wm)
            if not is_new_target or _memory_written_by_lifecycle(m):
                context_msgs.append(m)
            elif not db_backlog and ts is None and _comparable_text(m.get("content")) not in backlog_text:
                # A synthetic latest reply must not advance the cursor beyond
                # a bounded page of older persisted messages. Preserve the
                # legacy window fallback only when the DB has no backlog.
                _add_target(m)
    else:
        for m in recent:
            ts = _parse_ts(m)
            is_new = wm is None or ts is None or ts > wm
            if is_new and m.get("role") == target_role:
                _add_target(m)
            if not (is_new and m.get("role") == target_role) or _memory_written_by_lifecycle(m):
                context_msgs.append(m)

    if max_side_ts is None:
        return MemorySideOutcome(side, "no_new_messages", "no_new_messages", 0)
    if not new_target_msgs:
        # A bounded page of reserved messages must not starve later messages.
        # These rows were intentionally handled outside generic extraction.
        if conversation_id and (wm is None or max_side_ts > wm):
            await set_watermark(conversation_id, side, max_side_ts)
            return MemorySideOutcome(side, "excluded", "reserved_messages", 0, "advanced")
        return MemorySideOutcome(side, "excluded", "reserved_messages", 0)
    context_msgs = _context_before_targets(context_msgs, new_target_msgs, recent, target_role)
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
            prev_ai_line=_ai_line_before(
                recent, new_target_msgs[0] if new_target_msgs else None
            ),
        )
    except MemoryExtractionError as exc:
        return MemorySideOutcome(side, "disabled" if exc.kind == "prompt_disabled" else "failed", exc.kind)

    progress = "not_applicable" if not conversation_id else "unchanged"
    if conversation_id and (wm is None or max_side_ts > wm):
        try:
            await set_watermark(conversation_id, side, max_side_ts)
            progress = "advanced"
        except MemoryProgressError as exc:
            return MemorySideOutcome(side, "failed", exc.reason, len(stored_ids), "unknown")
    return MemorySideOutcome(side, "completed", "processed", len(stored_ids), progress)


async def _advance_side_watermark_without_extraction(
    recent: list[dict],
    conversation_id: str | None,
    *,
    side: Literal["user", "ai"],
) -> Literal["advanced", "unchanged", "not_applicable"]:
    """Advance watermark for a side intentionally skipped from extraction.

    Schedule/current-state replies are ephemeral answers. If we skip AI memory
    extraction without moving the AI watermark, a later non-skipped turn would
    see this assistant reply as "new" and extract it retroactively.
    """
    if conversation_id is None:
        return "not_applicable"
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
        return "unchanged"

    wm = _ensure_aware(await get_watermark(conversation_id, side))
    if wm is None or max_side_ts > wm:
        await set_watermark(conversation_id, side, max_side_ts)
        return "advanced"
    return "unchanged"


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


def _ai_line_before(messages: list[dict], first_new_target: dict | None) -> str:
    """取引出目标消息的那一次 AI 发言 (该次回复的全部气泡).

    不能直接拿 context 的最后一句: 记忆管线跑在 AI 回复之后, 那个位置上的是 AI
    对本轮的**回复**, 不是引出本轮的**提问**. 判断 "不好" 该不该记, 需要的是它
    前面的 "你今天还好吗"。一次回复拆成多个气泡各占一行, 只取最后一个气泡会
    丢掉问题本身 (只剩 "你更喜欢那种呀？", 不知道在问软甜还是酷甜).
    """
    if not first_new_target:
        return ""
    target_id = first_new_target.get("id")
    reply: list[str] = []
    in_reply = False
    for m in messages:
        if target_id is not None and m.get("id") == target_id:
            break
        if m is first_new_target:
            break
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if m.get("role") == "assistant":
            if not in_reply:
                reply = []
            reply.append(content)
            in_reply = True
        else:
            in_reply = False
    return " ".join(reply)


def _comparable_text(text: Any) -> str:
    """Text with whitespace, punctuation, emoji and the || splitter removed."""
    return re.sub(r"[\W_]+", "", str(text or ""))


def _context_before_targets(
    context_msgs: list[dict],
    new_target_msgs: list[dict],
    recent: list[dict],
    target_role: str,
) -> list[dict]:
    """Keep only context rows at or before the newest extraction target.

    The extraction prompt renders 【历史上下文】 above 【待抽取消息】, so a later
    row there reads as having happened first: the user side showed the AI's
    reply to "我喜欢你" above "我喜欢你" itself (2026-10-15 trace). Rows without
    a timestamp are the just-generated synthetic ones: they are placed by their
    position in the window, and an un-timestamped target is the newest row, so
    it lifts the time bound entirely.
    """
    target_ts = [_parse_ts(m) for m in new_target_msgs]
    timed = [ts for ts in target_ts if ts is not None]
    latest_target_ts = max(timed) if timed and len(timed) == len(target_ts) else None
    last_target_pos = max(
        (i for i, m in enumerate(recent) if m.get("role") == target_role),
        default=-1,
    )
    earlier = recent[: last_target_pos + 1]
    kept: list[dict] = []
    for m in context_msgs:
        ts = _parse_ts(m)
        if ts is not None:
            if latest_target_ts is None or ts <= latest_target_ts:
                kept.append(m)
        elif any(m is row for row in earlier):
            kept.append(m)
    return kept


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
    skip_memory: bool = False,
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
        ]
        # Red packets write 生活/馈赠 via offerings; running extraction on the
        # rewritten "user just sent a red packet" line would duplicate rows
        # and leak RMB framing into memories.
        if not skip_memory:
            tasks.append(
                _bg_memory_pipeline(
                    user_id,
                    full_messages,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    skip_ai_side=skip_ai_memory,
                )
            )
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id, user_message))
            if not skip_memory:
                tasks.append(
                    _bg_offering_followup_memory(
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conversation_id,
                        workspace_id=workspace_id,
                        user_message=user_message,
                        ai_response=full_response,
                    )
                )
                # E3 表达学习: 计数节流, 每 LEARN_EVERY_N 条用户消息批量学一次
                tasks.append(_bg_expression_learning(agent_id, user_id, full_messages))
        await asyncio.gather(*tasks, return_exceptions=True)


async def _bg_offering_followup_memory(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    workspace_id: str | None,
    user_message: str,
    ai_response: str,
) -> None:
    from app.services.offerings_followup import record_offering_followup_from_chat

    await record_offering_followup_from_chat(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        user_message=user_message,
        ai_response=ai_response,
    )


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
