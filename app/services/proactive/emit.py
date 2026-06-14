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
from app.services.prompting.trace_components import snapshot_prompt_render_traces

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
    trace_id: str | None = None,
) -> str:
    """持久化主动消息 + 推 WS, 返回 assistant message id.

    spec §10.4: special_date 等场景需要 skip_post_process=True 标记,
    避免下游再做 emoji/拆句加工.

    trace_id 挂到 metadata, 让前端 Trace 按钮可点 (跟主聊天回复路径对齐).
    """
    metadata: dict[str, Any] = {
        "proactive": True,
        "trigger_type": trigger_type,
    }
    if skip_post_process:
        metadata["skip_post_process"] = True
    if trace_id:
        metadata["trace_id"] = trace_id
    prompt_traces = snapshot_prompt_render_traces()
    if prompt_traces:
        metadata["prompt_render_traces"] = prompt_traces
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
    try:
        from app.services.achievements.service import handle_assistant_message_event
        from app.services.notifications.service import notify_agent_message_created
        from app.services.runtime.tasks import fire_background

        fire_background(handle_assistant_message_event(
            conversation_id=conversation_id,
            message_id=created.id,
            text=message,
            metadata=metadata,
            occurred_at=getattr(created, "createdAt", None),
        ))
        fire_background(notify_agent_message_created(
            conversation_id=conversation_id,
            message_id=created.id,
            text=message,
            metadata=metadata,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
        ))
    except Exception as achievement_err:
        logger.debug(f"[ACH/PUSH] proactive message hook skipped: {achievement_err}")

    # 审计日志写失败不影响主流程.
    # 注: 跟 timetrigger 一样, 这个 prisma client 版本对混合 scalar+relation
    # 写法很挑剔. 历史 `agent: {connect: {id}}` + `workspaceId: workspace_id or ""`
    # 实测在 reminder 触发路径下报 "workspaceId: Field does not exist" + "agentId
    # required". 改为全 scalar 写法; workspace 可空, None 时 omit (传空串会被
    # 当 FK 校验拒绝).
    try:
        log_data: dict[str, Any] = {
            "agentId": agent_id,
            "userId": user_id,
            "conversationId": conversation_id,
            "message": message,
            "eventType": trigger_type,
        }
        if workspace_id:
            log_data["workspaceId"] = workspace_id
        await db.proactivechatlog.create(data=log_data)
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
