"""聊天回复推送后的善后工作：持久化 + 5 个并行后台任务。

`save_replies` 是热路径同步调用（持久化分段回复）；
`run_post_process` 是 fire-and-forget，把 5 个 _bg_* 任务并行执行：

- _bg_user_emotion: 提取用户 PAD → 写入用户消息 metadata (spec §3.2 用户侧)
- _bg_summarizer: 跑 3 层摘要并缓存到 Redis（spec §2 摘要复用）
- _bg_memory_pipeline: spec §2.1/§2.2 记忆抽取（user + AI）
- _bg_trait_adjustment: 检测用户反馈调整 agent 性格
- _bg_positive_recovery: 正向互动 +20 耐心（spec §2.5）

AI PAD 不在此处处理 — spec §3.2 由 orchestrator 在热路径内通过 compute_ai_pad 计算。
`_bg_memory_pipeline` 还被 boundary_phase 通过 bg_memory_pipeline_fn 注入使用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prisma import Json

from app.db import db
from app.services.interaction.boundary import check_positive_recovery
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.summarizer import summarize
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

            # spec: 首条带 trace_pending，后台 share 完成后再回填 trace_url
            if i == 0 and trace_id:
                metadata["trace_id"] = trace_id
                metadata["trace_pending"] = True

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
    """Spec §3.2 用户侧：把 data_fetch_phase 已经算好的 LLM 用户 PAD 写进消息 metadata。

    用户 PAD 的 LLM 调用 (extract_emotion) 已在热路径 data_fetch_phase 并行跑完，
    这里只负责持久化到消息 metadata 供将来审计 / emotion timeline 使用。
    AI 侧 PAD 由 orchestrator 热路径内通过 compute_ai_pad 计算，不在这里处理。
    """
    if not user_message_id or not user_emotion:
        return
    try:
        await db.message.update(
            where={"id": user_message_id},
            data={"metadata": Json({"emotion": user_emotion})},
        )
    except Exception as e:
        logger.warning(f"Background user emotion metadata write failed: {e}")


async def _bg_summarizer(
    messages: list[dict],
    current_message: str,
    memories: list[str] | None,
) -> None:
    """跑 3 层摘要并缓存到 Redis 供下次请求复用。"""
    try:
        result = await summarize(messages, current_message, memories)
        if result:
            logger.debug("Background summarizer completed and cached")
    except Exception as e:
        logger.warning(f"Background summarizer failed: {e}")


async def _bg_memory_pipeline(user_id: str, messages: list[dict]) -> None:
    """spec §2.1 / §2.2：用户侧与 AI 侧走两条独立管线，owner 由路径决定。

    取最近 6 条（3 轮 user+assistant）作为共享上下文。两条管线都看得到完整对话，
    但每条 prompt 只抽取自己那一侧的记忆，避免 LLM 从混合对话里错归 owner。
    """
    try:
        recent = messages[-6:]
        if not recent:
            return
        conv_text = "\n".join(
            f"{m.get('role', 'user')}: {m['content']}" for m in recent
        )
        has_user = any(m.get("role") == "user" for m in recent)
        has_ai = any(m.get("role") == "assistant" for m in recent)
        tasks = []
        if has_user:
            tasks.append(process_memory_pipeline(user_id, conv_text, side="user"))
        if has_ai:
            tasks.append(process_memory_pipeline(user_id, conv_text, side="ai"))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")


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
    memory_strings: list[str] | None,
    user_emotion: dict | None = None,
) -> None:
    """5 个后台任务并发执行。本身 fire-and-forget。

    1. 写用户 PAD 到消息 metadata (spec §3.2 用户侧，值已由热路径算出)
    2. 3 层摘要缓存
    3. 记忆抽取 pipeline（user + AI）
    4. 性格特征调整（仅 agent_id 存在）
    5. 正向恢复（仅 agent_id 存在）

    AI PAD (spec §3.2) 由 orchestrator 热路径内 compute_ai_pad 处理，不在此处。
    用户 PAD 的 LLM 调用 extract_emotion 也在 data_fetch_phase 并行跑，这里只写 metadata。
    """
    _ = conversation_id  # 保留供未来扩展（如 conversation-level metric）
    try:
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]
        tasks: list[Any] = [
            _bg_user_emotion(user_message_id, user_emotion),
            _bg_summarizer(full_messages, user_message, memory_strings),
            _bg_memory_pipeline(user_id, full_messages),
        ]
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id))
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")
