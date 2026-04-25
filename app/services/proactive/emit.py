"""主动消息持久化与广播.

抽离自 sender.py 与 special_dates.py 的共享路径:
- 写 messages 表 (assistant role + metadata.proactive=True)
- 写 proactive_chat_logs (审计)
- 推 WebSocket "proactive" 事件

不负责: LLM 生成 / prompt 选择 / 上下文装配 / 频率限流 (留给上层).
"""

from __future__ import annotations

import logging
from typing import Any

from prisma import Json

from app.db import db
from app.services.runtime.ws_manager import manager

logger = logging.getLogger(__name__)


async def emit_proactive_message(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    message: str,
    trigger_type: str,
    extra_metadata: dict[str, Any] | None = None,
    skip_post_process: bool = False,
    ws_payload_extra: dict[str, Any] | None = None,
) -> str:
    """持久化主动消息 + 推 WS, 返回 assistant message id.

    spec §10.4: special_date 等场景需要 skip_post_process=True 标记,
    避免下游再做 emoji/拆句加工.
    """
    metadata: dict[str, Any] = {
        "proactive": True,
        "trigger_type": trigger_type,
    }
    if skip_post_process:
        metadata["skip_post_process"] = True
    if extra_metadata:
        metadata.update(extra_metadata)

    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": message,
            "metadata": Json(metadata),
        }
    )

    # 审计日志写失败不影响主流程
    try:
        await db.proactivechatlog.create(
            data={
                "agent": {"connect": {"id": agent_id}},
                "userId": user_id,
                "workspaceId": workspace_id or "",
                "conversationId": conversation_id,
                "message": message,
                "eventType": trigger_type,
            }
        )
    except Exception as e:
        logger.warning(f"proactive_chat_log write failed: {e}")

    ws_payload: dict[str, Any] = {
        "text": message,
        "agent_id": agent_id,
        "user_id": user_id,  # send_to_workspace fallback 需要 (workspace_id=None 时退回 user 维度)
        "assistant_message_id": created.id,
        "trigger_type": trigger_type,
    }
    if ws_payload_extra:
        ws_payload.update(ws_payload_extra)
    # workspace 维度路由: 同一 user 多 agent 时不会跨 agent 广播 proactive.
    # workspace_id 为 None (历史 conv) 时 send_to_workspace 内部 fallback 到 send_to_user.
    await manager.send_to_workspace(workspace_id, "proactive", ws_payload)

    return created.id
