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
from app.services.interaction.boundary import check_positive_recovery
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.recording.watermark import get_watermark, set_watermark
from app.services.relationship.emotion import save_ai_emotion
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
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")


async def _pipeline_with_watermark(
    user_id: str,
    recent: list[dict],
    conversation_id: str | None,
    *,
    side: Literal["user", "ai"],
) -> None:
    """按 (conversation_id, side) 水位线切分 recent, 调用 extraction pipeline."""
    wm = await get_watermark(conversation_id, side) if conversation_id else None

    # 单次扫描: 解析 ts, 判定 new/context, 同步收集 side 最大 ts.
    # 无 ts 的消息 (刚生成未持久化的 AI reply / boundary 短路手工构造) 用 now()
    # 占位参与水位线推进, 否则混合 ts 场景下会漏推进导致下轮重抽.
    target_role = "user" if side == "user" else "assistant"
    fallback_now = datetime.now(UTC)
    context_msgs: list[dict] = []
    new_msgs: list[dict] = []
    max_side_ts: datetime | None = None
    for m in recent:
        ts = _parse_ts(m)
        is_new = wm is None or ts is None or ts > wm
        (new_msgs if is_new else context_msgs).append(m)
        if is_new and m.get("role") == target_role:
            effective = ts if ts is not None else fallback_now
            if max_side_ts is None or effective > max_side_ts:
                max_side_ts = effective

    if max_side_ts is None:
        return  # 该 side 无新消息, 跳过 LLM

    await process_memory_pipeline(
        user_id=user_id,
        new_conversation=_fmt_conversation(new_msgs),
        context_conversation=_fmt_conversation(context_msgs),
        side=side,
    )

    # 防时钟回退: 仅当新候选 > wm 才推进
    if conversation_id and (wm is None or max_side_ts > wm):
        await set_watermark(conversation_id, side, max_side_ts)


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


async def _bg_positive_recovery(agent_id: str, user_id: str) -> None:
    """spec §2.5：正向互动 +20 耐心（仅对通过边界检查的消息）。"""
    try:
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
    """5 个后台任务并发：写用户 PAD / 写 AI PAD 缓存 / 记忆抽取 / 性格反馈 / 耐心恢复。"""
    try:
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]
        tasks: list[Any] = [
            _bg_user_emotion(user_message_id, user_emotion),
            _bg_memory_pipeline(user_id, full_messages, conversation_id=conversation_id),
        ]
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id))
            if ai_emotion:
                tasks.append(save_ai_emotion(agent_id, ai_emotion))
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")
