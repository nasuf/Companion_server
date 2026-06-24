from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException

from app.models.offline import OfflineActivitiesResponse, OfflineActivityItem
from app.services.offline import activity_media_repo, activity_media_storage
from app.services.offline import repository as repo
from app.services.offline.activity_generation import (
    generate_activity_card,
    generate_activity_invite_message,
)
from app.services.offline.chat_emit import (
    emit_activity_card,
    emit_assistant,
    insert_user_activity_card,
    insert_user_component_message,
)
from app.services.offline.memory_hooks import remember_user_event


def _location_for_activity(ctx: dict) -> tuple[str, str | None]:
    city = (
        ctx.get("user_location_city") or ctx.get("user_location_region") or ""
    ).strip()
    if city:
        return city, None
    latitude = ctx.get("user_location_latitude")
    longitude = ctx.get("user_location_longitude")
    if latitude is None or longitude is None:
        return "", None
    try:
        return "", f"{float(latitude):.5f},{float(longitude):.5f} 附近"
    except (TypeError, ValueError):
        return "", None


async def get_home(user_id: str, workspace_id: str | None = None) -> dict:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    activities = await repo.list_activities(
        user_id,
        ctx["workspace_id"] if ctx else workspace_id,
    )
    gifts = await repo.list_gifts(
        user_id,
        ctx["workspace_id"] if ctx else workspace_id,
    )
    pending = [a for a in activities if a["status"] in {"pending", "accepted"}]
    completed = [a for a in activities if a["status"] == "completed"]
    shipping = [g for g in gifts if g["status"] in {"ordered", "shipping"}]
    tags = await repo.list_user_tags(
        user_id,
        ctx["workspace_id"] if ctx else workspace_id,
        agent_id=ctx["agent_id"] if ctx else None,
        limit=9,
    )
    return {
        "pending_activity_count": len(pending),
        "completed_activity_count": len(completed),
        "gift_count": len(gifts),
        "shipping_gift_count": len(shipping),
        "has_location": bool(ctx and ctx.get("has_location")),
        "tags": tags,
        "latest_activity": pending[0] if pending else None,
        "gift_summary": "礼物正在向你飞奔" if shipping else "你有一份惊喜在路上",
    }


async def list_activities(
    user_id: str,
    workspace_id: str | None = None,
) -> OfflineActivitiesResponse:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    resolved_workspace = ctx["workspace_id"] if ctx else workspace_id
    rows = await repo.list_activities(user_id, resolved_workspace)
    latest = next((a for a in rows if a["status"] in {"pending", "accepted"}), None)
    items = [await _with_completion_feedback(a) for a in rows]
    return OfflineActivitiesResponse(
        latest=OfflineActivityItem(**(await _with_completion_feedback(latest)))
        if latest
        else None,
        pending=[
            OfflineActivityItem(**a)
            for a in items
            if a["status"] in {"pending", "accepted"}
        ],
        ignored=[
            OfflineActivityItem(**a)
            for a in items
            if a["status"] == "ignored"
        ],
        completed=[
            OfflineActivityItem(**a)
            for a in items
            if a["status"] == "completed"
        ],
    )


async def clear_all_activities(user_id: str) -> dict[str, int]:
    media = await activity_media_repo.delete_user_activity_media(user_id)
    for item in media:
        activity_media_storage.delete_media_file(item.storage_key)
    activity_media_storage.delete_user_media_files(user_id)
    return await repo.clear_user_activities(user_id)


async def get_activity(user_id: str, activity_id: str) -> OfflineActivityItem:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    activity = await _with_completion_feedback(activity)
    return OfflineActivityItem(**activity)


async def create_recommendation_for_user(
    *,
    user_id: str,
    workspace_id: str | None = None,
    source: str = "manual",
) -> dict | None:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    if not ctx or not ctx.get("conversation_id"):
        return None
    city, search_location = _location_for_activity(ctx)
    if not city:
        return None
    card = await generate_activity_card(
        user_id=user_id,
        workspace_id=ctx["workspace_id"],
        city=city,
        source=source,
        search_location=search_location,
    )
    if not card:
        return None
    activity = await repo.create_activity(
        {
            **card,
            "user_id": user_id,
            "agent_id": ctx["agent_id"],
            "workspace_id": ctx["workspace_id"],
            "conversation_id": ctx["conversation_id"],
            "status": "pending",
        }
    )
    message = await generate_activity_invite_message(
        activity=activity,
        user_id=user_id,
        workspace_id=ctx["workspace_id"],
    )
    await emit_assistant(
        conversation_id=ctx["conversation_id"],
        user_id=user_id,
        agent_id=ctx["agent_id"],
        workspace_id=ctx["workspace_id"],
        message=message,
        real_world_type="activity",
        source_id=activity["id"],
        trigger_type="offline_activity_recommendation",
    )
    await emit_activity_card(
        conversation_id=ctx["conversation_id"],
        user_id=user_id,
        agent_id=ctx["agent_id"],
        workspace_id=ctx["workspace_id"],
        activity=activity,
        trigger_type="offline_activity_recommendation_card",
        status_label="待回复",
    )
    await repo.update_next_activity_due(
        user_id,
        ctx["agent_id"],
        ctx["workspace_id"],
        repo.next_activity_due(datetime.now(UTC)),
    )
    return activity


async def accept_activity(user_id: str, activity_id: str) -> OfflineActivityItem:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity["status"] not in {"pending", "accepted", "ignored"}:
        raise HTTPException(status_code=409, detail="Activity cannot be accepted")
    was_ignored = activity["status"] == "ignored"
    if was_ignored:
        feedback_text = f"用户重新接受了活动推荐：{activity['title']}"
        chat_message = (
            f"好呀，我把「{activity['title']}」重新放回待出行里。"
            "彩蛋任务也还在，等你想去的时候慢慢看。"
        )
        memory_text = f"用户重新接受了线下活动推荐：{activity['title']}"
    else:
        feedback_text = f"用户接受了活动推荐：{activity['title']}"
        chat_message = (
            f"好，我把「{activity['title']}」先替你放进待确定里。"
            "彩蛋任务也解锁啦，等你想去的时候再慢慢看。"
        )
        memory_text = f"用户接受了线下活动推荐：{activity['title']}"
    trigger_type = (
        "offline_activity_reaccepted" if was_ignored else "offline_activity_accepted"
    )
    updated = await repo.update_activity_status(activity_id, user_id, "accepted")
    if not updated:
        raise HTTPException(status_code=404, detail="Activity not found")
    await repo.create_activity_feedback(
        recommendation_id=activity_id,
        user_id=user_id,
        kind="accept",
        text=feedback_text,
    )
    ctx = await repo.resolve_user_context(user_id, activity.get("workspace_id"))
    if ctx:
        await insert_user_activity_card(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx["workspace_id"],
            activity=updated,
            trigger_type=f"{trigger_type}_card",
            status_label="已接受",
        )
        await emit_assistant(
            conversation_id=ctx.get("conversation_id"),
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message=chat_message,
            real_world_type="activity",
            source_id=activity_id,
            trigger_type=trigger_type,
        )
        await repo.update_next_activity_due(
            user_id,
            ctx["agent_id"],
            ctx["workspace_id"],
            repo.next_activity_due(datetime.now(UTC), accepted_delta_days=-3),
        )
    remember_user_event(
        user_id=user_id,
        workspace_id=activity.get("workspace_id"),
        text=memory_text,
    )
    return OfflineActivityItem(**updated)


async def ignore_activity(user_id: str, activity_id: str) -> OfflineActivityItem:
    activity = await repo.get_activity(activity_id, user_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity["status"] not in {"pending", "accepted"}:
        raise HTTPException(status_code=409, detail="Activity cannot be ignored")
    updated = await repo.update_activity_status(activity_id, user_id, "ignored")
    if not updated:
        raise HTTPException(status_code=404, detail="Activity not found")
    await repo.create_activity_feedback(
        recommendation_id=activity_id,
        user_id=user_id,
        kind="ignore",
        text=f"用户暂不考虑活动推荐：{activity['title']}",
    )
    ctx = await repo.resolve_user_context(user_id, activity.get("workspace_id"))
    if ctx:
        await insert_user_activity_card(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx["workspace_id"],
            activity=updated,
            trigger_type="offline_activity_ignored_card",
            status_label="暂不考虑",
        )
        await emit_assistant(
            conversation_id=ctx.get("conversation_id"),
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message="没关系，这个先不算。下次我会换一个更轻一点、更贴近你当下状态的选择。",
            real_world_type="activity",
            source_id=activity_id,
            trigger_type="offline_activity_ignored",
        )
        await repo.update_next_activity_due(
            user_id,
            ctx["agent_id"],
            ctx["workspace_id"],
            repo.next_activity_due(datetime.now(UTC), accepted_delta_days=3),
        )
    remember_user_event(
        user_id=user_id,
        workspace_id=activity.get("workspace_id"),
        text=f"用户暂时忽略了线下活动推荐：{activity['title']}",
    )
    return OfflineActivityItem(**updated)


async def complete_activity(
    user_id: str,
    activity_id: str,
    *,
    text: str,
    photo_attachment_ids: list[str],
) -> OfflineActivityItem:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity["status"] not in {"accepted", "completed"}:
        raise HTTPException(
            status_code=409,
            detail="Accept the activity before completing it",
        )
    ctx = await repo.resolve_user_context(user_id, activity.get("workspace_id"))
    conversation_id = (
        ctx.get("conversation_id") if ctx else activity.get("conversation_id")
    )
    if photo_attachment_ids:
        found = await activity_media_repo.get_activity_media(
            media_ids=photo_attachment_ids,
            user_id=user_id,
            recommendation_id=activity_id,
        )
        if len(found) != len(photo_attachment_ids):
            raise HTTPException(status_code=400, detail="Invalid activity media")
    else:
        found = []
    updated = await repo.update_activity_status(activity_id, user_id, "completed")
    if not updated:
        raise HTTPException(status_code=404, detail="Activity not found")
    await repo.create_activity_feedback(
        recommendation_id=activity_id,
        user_id=user_id,
        kind="completion",
        text=text,
        photo_attachment_ids=photo_attachment_ids,
    )
    if conversation_id and (text.strip() or found):
        await insert_user_component_message(
            conversation_id=conversation_id,
            workspace_id=activity.get("workspace_id"),
            content=text.strip() or "分享了活动完成情况",
            metadata={
                "real_world_type": "activity",
                "source_id": activity_id,
                "trigger_type": "offline_activity_completion_share",
                "attachments": [_media_to_metadata(item) for item in found],
            },
        )
    if ctx:
        await insert_user_activity_card(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx["workspace_id"],
            activity=updated,
            trigger_type="offline_activity_completed_card",
            status_label="已完成",
        )
        await emit_assistant(
            conversation_id=ctx.get("conversation_id"),
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message=f"我看到啦。「{activity['title']}」被你带回来了。谢谢你把这一小段现实也分享给我。",
            real_world_type="activity",
            source_id=activity_id,
            trigger_type="offline_activity_completed",
        )
    remember_user_event(
        user_id=user_id,
        workspace_id=activity.get("workspace_id"),
        text=f"用户完成了线下活动「{activity['title']}」。分享内容：{text}",
    )
    updated = await _with_completion_feedback(updated)
    return OfflineActivityItem(**updated)


async def _with_completion_feedback(activity: dict | None) -> dict:
    if not activity:
        return {}
    if activity.get("status") != "completed":
        return activity
    feedback = await repo.get_activity_completion_feedback(
        recommendation_id=activity["id"],
        user_id=activity["user_id"],
    )
    return {**activity, "completion_feedback": feedback}


def _media_to_metadata(media: activity_media_repo.OfflineActivityMedia) -> dict:
    return {
        "id": media.id,
        "kind": media.kind,
        "name": media.name,
        "mime": media.mime,
        "size": media.size,
        "width": media.width,
        "height": media.height,
        "url": media.url,
        "vision_status": "ready",
    }
