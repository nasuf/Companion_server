"""AI 一轮回复全部发出后的唯一收尾点 —— 主路径与所有短路路径共用。

之前主路径和短路路径各写各的: 短路 (语气词表情 / 问当前状态 / 查计划 …)
只记了回复时间戳, 没重新 arm 主动交流, 状态停在用户发消息时置的 idle,
之后再也不会主动说话。收口到这里后, 任何回复出口都会:

1. 记录最后回复时间 (异步回复「交流状态」延迟档位读它)
2. arm 主动交流: 5 分钟后进入话题完结判定窗 (proactive/followup.py)
3. 告别回合关闭话题会话: 下一条用户消息开新会话, B 追问名额恢复
"""

from __future__ import annotations

from app.services.interaction.reply_context import save_last_reply_timestamp
from app.services.interaction.topic_continuity import close_session
from app.services.proactive.state import ARM_REASON_FAREWELL, arm_after_assistant_turn
from app.services.runtime.tasks import fire_background


async def finish_assistant_turn(
    *,
    conversation_id: str,
    agent_id: str | None,
    user_id: str,
    proactive_reason: str | None,
    workspace_id: str | None = None,
    turn_message_ids: list[str] | None = None,
) -> None:
    """`proactive_reason=None` 表示本轮之后不主动 (边界系统 / 危机短路).

    turn_message_ids: 本轮回应的用户消息; 用户在生成期间又发了更新的消息时,
    这次 arm 由 SQL 守卫跳过, 交给新消息那一轮自己收尾。
    """
    if not agent_id:
        return
    await save_last_reply_timestamp(agent_id, user_id)
    if proactive_reason is None:
        return
    fire_background(arm_after_assistant_turn(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        reason=proactive_reason,
        turn_message_ids=turn_message_ids,
    ))
    if proactive_reason == ARM_REASON_FAREWELL:
        fire_background(close_session(conversation_id))
