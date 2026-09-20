"""聊天发图/语音 → 到达中活动的媒体路由（spec §3.5）。

到达后（进行中 + 已到达）：现场图片入活动素材并走识图；语音转写文本归档（不进画廊）。
未到达 / 无进行中活动：不介入，媒体只留在聊天。整条链路 fire-and-forget，异常吞掉不
影响聊天主流程。
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.chat_media.repo import ChatAttachment
from app.services.offline import recognition
from app.services.offline import repository as repo
from app.services.offline.chat_emit import offline_trace
from app.services.offline.module_settings import is_activity_enabled

logger = logging.getLogger(__name__)


async def on_user_chat_media(
    *,
    user_id: str,
    workspace_id: str | None,
    message_id: str,
    attachments: list[ChatAttachment],
) -> None:
    try:
        images = [a for a in attachments if getattr(a, "kind", "") == "image"]
        audios = [a for a in attachments if getattr(a, "kind", "") == "audio"]
        if not images and not audios:
            return
        if not await is_activity_enabled():
            return
        activity = await _active_reached_activity(user_id, workspace_id)
        if not activity:
            return  # 未到达 / 无进行中活动 → 媒体只留聊天（spec §3.5）
        ctx = await repo.resolve_user_context(user_id, workspace_id)
        for att in images:
            await _capture_image(activity, ctx, att, message_id)
        for att in audios:
            await _capture_audio(activity, att, message_id)
    except Exception as exc:  # 后台钩子：任何异常都不得冒泡到聊天主流程
        logger.warning("[offline-capture] 处理失败 user=%s err=%s", user_id, exc)


async def _active_reached_activity(
    user_id: str, workspace_id: str | None
) -> dict[str, Any] | None:
    rows = await repo.list_activities(user_id, workspace_id)
    for activity in rows:  # list_activities 按 created_at DESC，取最近的进行中已到达
        if activity.get("status") == "accepted" and activity.get("reached"):
            return activity
    return None


async def _capture_image(
    activity: dict[str, Any],
    ctx: dict[str, Any] | None,
    att: ChatAttachment,
    message_id: str,
) -> None:
    media_id = await repo.create_captured_media(
        recommendation_id=activity["id"],
        user_id=activity["user_id"],
        storage_key=att.storage_key,
        url=att.url,
        mime=att.mime,
        size=att.size,
        width=att.width,
        height=att.height,
        source_message_id=message_id,
    )
    # 复用聊天入站时已算好的 vision_summary（真实视觉模型输出）做条件匹配，避免二次识图。
    description = (getattr(att, "vision_summary", "") or "").strip()
    if not description:
        return  # 无描述无法匹配，仅作素材
    async with offline_trace(
        "photo_recognition",
        conversation_id=(ctx or {}).get("conversation_id"),
        agent_id=(ctx or {}).get("agent_id"),
        user_id=activity["user_id"],
    ) as tracer:
        await recognition.recognize_on_photo(
            activity=activity,
            ctx=ctx,
            photo_description=description,
            media_id=media_id,
            source_message_id=message_id,
            trace_id=tracer.safe_trace_id,
        )


async def _capture_audio(
    activity: dict[str, Any], att: ChatAttachment, message_id: str
) -> None:
    # 语音在上传时已转写（transcription_text），这里只把文本归档到活动，不进画廊。
    transcript = (getattr(att, "transcription_text", "") or "").strip()
    if not transcript:
        return
    await repo.create_activity_feedback(
        recommendation_id=activity["id"],
        user_id=activity["user_id"],
        kind="voice_transcript",
        text=transcript,
        metadata={"source_message_id": message_id},
    )
