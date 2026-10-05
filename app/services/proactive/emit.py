"""主动消息持久化与广播.

抽离自 sender.py 与 special_dates.py 的共享路径:
- 写 messages 表 (assistant role + metadata.proactive=True)
- 写 proactive_chat_logs (审计)
- 推 WebSocket "proactive" 事件

不负责: LLM 生成 / prompt 选择 / 上下文装配 / 频率限流 (留给上层).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
    voice_eligible: bool = True,
    guard_activity_id: str | None = None,
    guard_delivery_key: str | None = None,
    abort_if_user_replied_since: datetime | None = None,
) -> str:
    """持久化主动消息 + 推 WS, 返回 assistant message id.

    spec §10.4: special_date 等场景需要 skip_post_process=True 标记,
    避免下游再做 emoji/拆句加工.

    trace_id 挂到 metadata, 让前端 Trace 按钮可点 (跟主聊天回复路径对齐).

    abort_if_user_replied_since: 该时刻之后本会话出现过用户消息就不插入, 返回 "".
    生成一条主动消息要好几秒 (LLM + 可能的联网/TTS), 用户恰好在这期间回来时,
    spec 要求取消待发任务、优先响应用户。判断与插入在同一条 SQL 里完成 (READ
    COMMITTED 下仍有毫秒级窗口看不到并发中未提交的用户消息, 可接受)。
    """
    # One delivery/bubble, but preserve every clause from older split-style prompts.
    if "||" in message:
        message = " ".join(part.strip() for part in message.split("||") if part.strip())

    # 硬保证: 一条消息最多 1 个 emoji (spec §5.3 + 2026-07-08 产品要求).
    # 主动消息不走 emit_replies, 在此单独收口.
    from app.services.emoji import limit_emojis
    # 系统标记收口: 主动消息 prompt 带 reply_prefix (response_instruction),
    # LLM 可能跟着输出 [EMO:]/条数标记 — 此路径不走 chat split 管线, 单独剥.
    # 剥完为空 (整条都是标记) 用占位省略号, 绝不回退未清理原文.
    from app.services.chat.reply_formatting import strip_system_markers

    message = limit_emojis(strip_system_markers(message) or "...")

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

    prepared_voice = None
    if guard_activity_id and guard_delivery_key:
        voice_eligible = False
    if voice_eligible and not (ws_payload_extra or {}).get("component_card"):
        from app.services.speech_output.policy import (
            VoiceContext,
            should_generate_voice,
        )

        if await should_generate_voice(
            context=VoiceContext.PROACTIVE_CHAT,
            client_supports_voice=True,
        ):
            try:
                agent = await db.aiagent.find_unique(where={"id": agent_id})
                if agent is not None:
                    from app.services.speech_output.delivery import (
                        prepare_voice_output,
                    )

                    prepared_voice = await prepare_voice_output(
                        text=message,
                        user_id=user_id,
                        agent=agent,
                        conversation_id=conversation_id,
                        source="proactive",
                    )
                    metadata["display_mode"] = "voice"
                    metadata["attachments"] = [prepared_voice.metadata]
            except Exception as voice_error:
                logger.warning(
                    "[TTS] proactive synthesis failed; falling back to text: %s",
                    type(voice_error).__name__,
                )

    try:
        if guard_activity_id and guard_delivery_key:
            rows = await db.query_raw(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, metadata, created_at
                )
                SELECT $1, $2, 'assistant', $3, $4::jsonb, CURRENT_TIMESTAMP
                FROM offline_activity_recommendations activity
                WHERE activity.id = $5
                  AND activity.status = 'accepted'
                  AND activity.reached = TRUE
                  AND activity.companion_state->>'delivery_key' = $6
                  AND NOT EXISTS (
                      SELECT 1
                      FROM messages user_message
                      WHERE user_message.conversation_id = $2
                        AND user_message.role = 'user'
                        AND user_message.created_at > (
                            activity.companion_state->>'delivery_reserved_at'
                        )::timestamptz
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM messages prior
                      WHERE prior.metadata->>'offline_companion_delivery_key' = $6
                  )
                RETURNING id, created_at
                """,
                uuid4().hex,
                conversation_id,
                message,
                json.dumps(metadata, ensure_ascii=False, default=str),
                guard_activity_id,
                guard_delivery_key,
            )
            if not rows:
                return ""
            row = rows[0]
            created_id = str(
                row["id"] if isinstance(row, dict) else row.id
            )
            created_at = (
                row.get("created_at")
                if isinstance(row, dict)
                else row.created_at
            )
        elif abort_if_user_replied_since is not None:
            created_id, created_at = await _insert_unless_user_replied(
                conversation_id=conversation_id,
                message=message,
                metadata=metadata,
                since=abort_if_user_replied_since,
            )
            if not created_id:
                logger.info(
                    "[PROACTIVE-EMIT] aborted: user replied during generation "
                    f"trigger={trigger_type}"
                )
                if prepared_voice is not None:
                    from app.services.speech_output.delivery import (
                        discard_prepared_voice_output,
                    )

                    await discard_prepared_voice_output(prepared_voice)
                return ""
        else:
            created = await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": message,
                    "metadata": Json(metadata),
                }
            )
            created_id = created.id
            created_at = created.createdAt
    except BaseException:
        if prepared_voice is not None:
            from app.services.speech_output.delivery import (
                discard_prepared_voice_output,
            )

            try:
                await discard_prepared_voice_output(prepared_voice)
            except Exception:
                logger.warning("[TTS] proactive cleanup failed", exc_info=True)
        raise
    if prepared_voice is not None:
        try:
            from app.services.speech_output.delivery import (
                bind_prepared_voice_output,
            )

            await bind_prepared_voice_output(
                prepared_voice,
                message_id=created_id,
            )
        except Exception as bind_error:
            logger.warning(
                "[TTS] proactive attachment bind failed; falling back to text: %s",
                type(bind_error).__name__,
            )
            from app.services.speech_output.delivery import (
                discard_prepared_voice_output,
            )

            await discard_prepared_voice_output(prepared_voice)
            prepared_voice = None
            metadata.pop("display_mode", None)
            metadata.pop("attachments", None)
            await db.message.update(
                where={"id": created_id},
                data={"metadata": Json(metadata)},
            )
    try:
        from app.services.achievements.service import handle_assistant_message_event
        from app.services.notifications.service import notify_agent_message_created
        from app.services.runtime.tasks import fire_background

        fire_background(handle_assistant_message_event(
            conversation_id=conversation_id,
            message_id=created_id,
            text=message,
            metadata=metadata,
            occurred_at=created_at,
        ))
        fire_background(notify_agent_message_created(
            conversation_id=conversation_id,
            message_id=created_id,
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
        "assistant_message_id": created_id,
        "trigger_type": trigger_type,
    }
    if prepared_voice is not None:
        ws_payload["display_mode"] = "voice"
        ws_payload["attachments"] = [prepared_voice.metadata]
    if ws_payload_extra:
        ws_payload.update(ws_payload_extra)
    # workspace 维度路由: 同一 user 多 agent 时不会跨 agent 广播 proactive.
    # workspace_id 为 None (历史 conv) 时 send_to_workspace 内部 fallback 到 send_to_user.
    await manager.send_to_workspace(workspace_id, "proactive", ws_payload)

    return created_id


def _naive_utc(ts: datetime) -> str:
    """messages.created_at 是 UTC 的 timestamp without time zone (Prisma 默认)."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


async def _insert_unless_user_replied(
    *,
    conversation_id: str,
    message: str,
    metadata: dict[str, Any],
    since: datetime,
) -> tuple[str | None, datetime | None]:
    """「没有更新的用户消息才插入」(单条 SQL); 返回 (id, created_at) 或 (None, None)."""
    created_at = datetime.now(timezone.utc)
    rows = await db.query_raw(
        """
        INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
        SELECT $1, $2, 'assistant', $3, $4::jsonb, $5::timestamp
        WHERE NOT EXISTS (
            SELECT 1
            FROM messages user_message
            WHERE user_message.conversation_id = $2
              AND user_message.role = 'user'
              AND user_message.created_at > $6::timestamp
        )
        RETURNING id
        """,
        str(uuid4()),
        conversation_id,
        message,
        json.dumps(metadata, ensure_ascii=False, default=str),
        _naive_utc(created_at),
        _naive_utc(since),
    )
    if not rows:
        return None, None
    # 返回自己写入的时刻 (raw query 回来的是字符串, 下游成就/推送要 datetime)
    return str(rows[0]["id"]), created_at
