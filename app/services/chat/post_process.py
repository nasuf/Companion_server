"""聊天回复推送后的善后工作：持久化 + 5 个并行后台任务。

`save_replies` 是热路径同步调用（持久化分段回复）；
`run_post_process` 是 fire-and-forget，把 5 个 _bg_* 任务并行执行：

- _bg_emotion: 提取用户情绪 → 更新 AI 情绪状态 → 写消息 metadata
- _bg_summarizer: 跑 3 层摘要并缓存到 Redis（spec §2 摘要复用）
- _bg_memory_pipeline: spec §2.1/§2.2 记忆抽取（user + AI）
- _bg_trait_adjustment: 检测用户反馈调整 agent 性格
- _bg_positive_recovery: 正向互动 +20 耐心（spec §2.5）

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
from app.services.relationship.emotion import (
    extract_emotion,
    get_ai_emotion,
    save_ai_emotion,
    update_emotion_state,
)
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


async def _bg_emotion(
    agent_id: str | None,
    user_message_id: str | None,
    user_message: str,
    cached_emotion: dict | None = None,
    topic_intimacy: float = 50.0,
    mbti: dict | None = None,
) -> None:
    """提取用户情绪 → 更新 AI 情绪状态 → 写消息 metadata。"""
    if not agent_id:
        return
    try:
        user_emotion = await extract_emotion(user_message)
        current_emotion = cached_emotion or await get_ai_emotion(agent_id)
        new_emotion = update_emotion_state(current_emotion, user_emotion, topic_intimacy, mbti=mbti)
        await save_ai_emotion(agent_id, new_emotion)
        if user_message_id:
            await db.message.update(
                where={"id": user_message_id},
                data={"metadata": Json({"emotion": user_emotion})},
            )
    except Exception as e:
        logger.warning(f"Background emotion update failed: {e}")


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
    """spec §2.1/§2.2：用户 + AI 消息走同一 3 步 pipeline。

    取最近 6 条（3 轮 user+assistant），spec §2.1.3 要求"用户消息 + 最近3轮上下文"。
    """
    try:
        recent = messages[-6:]
        if not recent:
            return
        conv_text = "\n".join(
            f"{m.get('role', 'user')}: {m['content']}" for m in recent
        )
        await process_memory_pipeline(user_id, conv_text)
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
    cached_emotion: dict | None = None,
    mbti: dict | None = None,
    topic_intimacy: float = 50.0,
) -> None:
    """5 个后台任务并发执行。本身 fire-and-forget。

    1. 用户情绪抽取 + AI 状态更新
    2. 3 层摘要缓存
    3. 记忆抽取 pipeline（user + AI）
    4. 性格特征调整（仅 agent_id 存在）
    5. 正向恢复（仅 agent_id 存在）
    """
    _ = conversation_id  # 保留供未来扩展（如 conversation-level metric）
    try:
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]
        tasks: list[Any] = [
            _bg_emotion(agent_id, user_message_id, user_message, cached_emotion, topic_intimacy, mbti),
            _bg_summarizer(full_messages, user_message, memory_strings),
            _bg_memory_pipeline(user_id, full_messages),
        ]
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
            tasks.append(_bg_positive_recovery(agent_id, user_id))
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")
