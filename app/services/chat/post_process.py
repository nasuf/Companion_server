"""聊天回复推送后的善后工作：持久化 + 并行后台任务。

`save_replies` 是热路径同步调用（持久化分段回复）；
`run_post_process` 是 fire-and-forget，把以下任务并行执行：

- _bg_user_emotion: 写用户 PAD 到消息 metadata（值由热路径已算好）
- _bg_memory_pipeline: spec §2.1/§2.2 记忆抽取（user + AI）
- _bg_trait_adjustment: 检测用户反馈调整 agent 性格
- _bg_positive_recovery: 正向互动 +20 耐心（spec §2.5）
- save_ai_emotion: 回写 AI PAD 缓存

`_bg_memory_pipeline` 还被 boundary_phase 通过 bg_memory_pipeline_fn 注入使用。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from prisma import Json

from app.db import db
from app.services.chat.intent_replies import positive_interaction_check
from app.services.interaction.boundary import (
    PATIENCE_MAX,
    check_positive_recovery,
    get_patience,
)
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.recording.watermark import get_watermark, set_watermark
from app.services.relationship.emotion import save_ai_emotion
from app.services.runtime.ws_manager import manager
from app.services.trait_adjustment import (
    apply_trait_adjustment,
    detect_direct_feedback,
    infer_feedback,
)

logger = logging.getLogger(__name__)


async def save_replies(
    conversation_id: str,
    replies: list[str | dict],
    trace_id: str | None = None,
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

            # 懒触发 trace: 首条只挂 trace_id, 用户点 Trace 按钮时通过
            # /traces/resolve endpoint 触发 share + mirror 写入.
            if i == 0 and trace_id:
                metadata["trace_id"] = trace_id

            created = await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": text,
                    "metadata": Json(metadata),
                }
            )
            if i == 0:
                first_message_id = created.id
        return first_message_id
    except Exception as e:
        logger.error(f"Failed to save replies: {e}")
        return None


async def _bg_user_emotion(
    user_message_id: str | None,
    user_emotion: dict | None,
) -> None:
    """Spec §3.2 用户侧：写 LLM 算好的用户 PAD 到消息 metadata。"""
    if not user_message_id or not user_emotion:
        return
    try:
        await db.message.update(
            where={"id": user_message_id},
            data={"metadata": Json({"emotion": user_emotion})},
        )
    except Exception as e:
        logger.warning(f"Background user emotion metadata write failed: {e}")


async def _bg_memory_pipeline(
    user_id: str,
    messages: list[dict],
    conversation_id: str | None = None,
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
    """
    try:
        recent = messages[-6:]
        if not recent:
            return
        roles = {m.get("role") for m in recent}
        sides_to_run: list[tuple[Literal["user", "ai"], str]] = [
            ("user", "user"),
            ("ai", "assistant"),
        ]
        tasks = [
            _pipeline_with_watermark(user_id, recent, conversation_id, side=side)
            for side, role in sides_to_run
            if role in roles
        ]
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total = sum(results)
        if total > 0 and conversation_id:
            await manager.send_event(
                conversation_id,
                "memory_extracted",
                {"count": total},
            )
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")


async def _pipeline_with_watermark(
    user_id: str,
    recent: list[dict],
    conversation_id: str | None,
    *,
    side: Literal["user", "ai"],
) -> int:
    """按 (conversation_id, side) 水位线切分 recent, 调用 extraction pipeline.
    返回该侧实际入库的记忆条数 (供 _bg_memory_pipeline 汇总后推 WS 事件)."""
    wm = await get_watermark(conversation_id, side) if conversation_id else None

    # Cross-role NEW msgs go to context_msgs, not new_target_msgs — prevents
    # AI's just-generated reply from being extracted as a user fact (and vice versa).
    # target_role NEW msgs without ts (boundary short-circuit, fresh reply) use now()
    # so the watermark still advances; otherwise next round would re-extract them.
    target_role = "user" if side == "user" else "assistant"
    fallback_now = datetime.now(UTC)
    context_msgs: list[dict] = []
    new_target_msgs: list[dict] = []
    max_side_ts: datetime | None = None
    for m in recent:
        ts = _parse_ts(m)
        is_new = wm is None or ts is None or ts > wm
        if is_new and m.get("role") == target_role:
            new_target_msgs.append(m)
            effective = ts if ts is not None else fallback_now
            if max_side_ts is None or effective > max_side_ts:
                max_side_ts = effective
        else:
            context_msgs.append(m)

    if max_side_ts is None:
        return 0  # 该 side 无新消息, 跳过 LLM

    stored_ids = await process_memory_pipeline(
        user_id=user_id,
        new_conversation=_fmt_conversation(new_target_msgs),
        context_conversation=_fmt_conversation(context_msgs),
        side=side,
    )

    # 防时钟回退: 仅当新候选 > wm 才推进
    if conversation_id and (wm is None or max_side_ts > wm):
        await set_watermark(conversation_id, side, max_side_ts)

    return len(stored_ids)


def _parse_ts(m: dict) -> datetime | None:
    ts = m.get("createdAt")
    if isinstance(ts, datetime):
        return ts
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _fmt_conversation(msgs: list[dict]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)


async def _bg_trait_adjustment(agent_id: str, user_message: str) -> None:
    """检测用户反馈信号 → 调整 agent 性格特征。"""
    try:
        adjustments = detect_direct_feedback(user_message) or infer_feedback(user_message)
        if adjustments:
            await apply_trait_adjustment(agent_id, adjustments)
    except Exception as e:
        logger.warning(f"Background trait adjustment failed: {e}")


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
            return
        if not await positive_interaction_check(user_message):
            return
        await check_positive_recovery(agent_id, user_id)
    except Exception as e:
        logger.warning(f"Background positive recovery failed: {e}")


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
    ai_emotion: dict | None = None,
) -> None:
    """5 个后台任务并发：写用户 PAD / 写 AI PAD 缓存 / 记忆抽取 / 性格反馈 / 耐心恢复。

    起独立 usage session: fire_background 已把 ContextVar 隔离,
    这里重新开让记忆/trait 的 token 落到 llm_usage 自己一行 (scope=post_process).
    """
    from app.services.llm.usage_tracker import usage_session
    async with usage_session(
        scope="post_process", conversation_id=conversation_id,
        agent_id=agent_id, user_id=user_id,
    ):
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]
        tasks: list[Any] = [
            _bg_user_emotion(user_message_id, user_emotion),
            _bg_memory_pipeline(user_id, full_messages, conversation_id=conversation_id),
        ]
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id, user_message))
            if ai_emotion:
                tasks.append(save_ai_emotion(agent_id, ai_emotion))
        await asyncio.gather(*tasks, return_exceptions=True)
