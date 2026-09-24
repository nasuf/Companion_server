from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from fastapi import HTTPException

from app.config import settings
from app.models.offline import (
    OfflineActivitiesResponse,
    OfflineActivityFragmentItem,
    OfflineActivityItem,
    OfflineActivityReviewResponse,
    OfflineMemoryNoteResponse,
)
from app.services.offline import activity_media_repo, activity_media_storage, gift_repository
from app.services.offline import memory_note as memory_note_gen
from app.services.offline import prophecy as prophecy_pool
from app.services.offline import repository as repo
from app.services.offline import shooting_conditions
from app.services.offline.recognition import TIER_LABELS
from app.services.offline.geocode import geocode_address, haversine_m, make_place_key
from app.services.llm.models import get_chat_model, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.runtime.tasks import fire_background
from app.services.offline.activity_generation import (
    generate_activity_card,
    generate_activity_invite_message,
    resolve_activity_search_context,
)
from app.services.offline.chat_emit import (
    offline_trace,
    emit_activity_card,
    emit_assistant,
    insert_user_activity_card,
    insert_user_component_message,
)
from app.services.offline.memory_hooks import remember_user_event

logger = logging.getLogger(__name__)


def _location_for_activity(ctx: dict) -> tuple[str, str, tuple[str, ...]]:
    city = (
        ctx.get("user_location_city") or ctx.get("user_location_region") or ""
    ).strip()
    if not city:
        return "", "", ()
    _, search_anchor, match_terms = resolve_activity_search_context(
        city=ctx.get("user_location_city"),
        region=ctx.get("user_location_region"),
    )
    return city, search_anchor, match_terms


async def get_home(user_id: str, workspace_id: str | None = None) -> dict:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    activities = await repo.list_activities(
        user_id,
        ctx["workspace_id"] if ctx else workspace_id,
    )
    gifts = await gift_repository.list_gifts(
        user_id,
        ctx["workspace_id"] if ctx else workspace_id,
    )
    pending = [a for a in activities if a["status"] == "pending"]
    accepted = [a for a in activities if a["status"] == "accepted"]
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
        "accepted_activity_count": len(accepted),
        "completed_activity_count": len(completed),
        "gift_count": len(gifts),
        "shipping_gift_count": len(shipping),
        "has_location": bool(ctx and ctx.get("has_location")),
        "tags": tags,
        "latest_activity": (pending or accepted or completed or [None])[0],
        "gift_summary": "礼物正在向你飞奔" if shipping else "你有一份惊喜在路上",
    }


async def list_activities(
    user_id: str,
    workspace_id: str | None = None,
) -> OfflineActivitiesResponse:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    resolved_workspace = ctx["workspace_id"] if ctx else workspace_id
    rows = await repo.list_activities(user_id, resolved_workspace)
    latest = next((a for a in rows if a["status"] == "pending"), None)
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


async def admin_inspect_activity(user_id: str, activity_id: str) -> dict:
    """管理员测试页检视：活动详情 + 拍摄物品(任务)全量 + 已产出碎片。仅 admin。"""
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    items = await repo.admin_list_conditions(activity_id)
    fragments = await repo.list_fragments(activity_id)
    return {
        "id": activity["id"],
        "title": activity.get("title") or "",
        "location_name": activity.get("location_name") or activity.get("address") or "",
        "category": activity.get("category") or "",
        "summary": activity.get("summary") or activity.get("description") or "",
        "status": activity.get("status") or "",
        "reached": bool(activity.get("reached")),
        "items": items,
        "fragments": [
            {"tier": f.get("tier"), "text": f.get("text")} for f in fragments
        ],
    }


async def admin_generate_items(user_id: str, activity_id: str) -> dict:
    """管理员测试专用：绕过到达校验，直接为活动生成 3–5 拍摄物品(任务) + 分档预生成。

    幂等（已有物品则跳过）。生成后返回检视结果，供测试页展示任务细节。
    """
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    await shooting_conditions.generate_items_for_activity(activity)
    return await admin_inspect_activity(user_id, activity_id)


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
    city, search_anchor, match_terms = _location_for_activity(ctx)
    if not city:
        return None
    card = await generate_activity_card(
        user_id=user_id,
        workspace_id=ctx["workspace_id"],
        city=city,
        source=source,
        search_location=search_anchor,
        location_terms=list(match_terms),
    )
    if not card:
        return None
    # 地理编码：地址 -> 经纬度（供到达 ≤200m 校验）+ 同地点去重键。key 未配置或失败
    # 时 coords=None，不阻断推荐（到达校验按 offline_arrival_require_geocode 处理）。
    coords = await geocode_address(card.get("address"), card.get("city") or city)
    place_lat, place_lng = coords if coords else (None, None)
    place_key = make_place_key(
        card.get("location_name"), card.get("address"), card.get("city") or city
    )
    activity = await repo.create_activity(
        {
            **card,
            "user_id": user_id,
            "agent_id": ctx["agent_id"],
            "workspace_id": ctx["workspace_id"],
            "conversation_id": ctx["conversation_id"],
            "status": "pending",
            "place_lat": place_lat,
            "place_lng": place_lng,
            "place_key": place_key,
        }
    )
    async with offline_trace(
        "activity_invite",
        conversation_id=ctx["conversation_id"],
        agent_id=ctx["agent_id"], user_id=user_id,
    ) as tracer:
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
            trace_id=tracer.safe_trace_id,
        )
    await emit_activity_card(
        conversation_id=ctx["conversation_id"],
        user_id=user_id,
        agent_id=ctx["agent_id"],
        workspace_id=ctx["workspace_id"],
        activity=activity,
        trigger_type="offline_activity_recommendation_card",
        status_label="待确定",
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
    # spec §4.4 同地点复用：接受新推荐时若同地点已有进行中活动，直接返回既有（前端
    # 打开其打卡页），并把当前这条收进「暂不考虑」，避免同地点重复占列表。
    place_key = activity.get("place_key")
    if activity["status"] == "pending" and place_key:
        existing = await repo.find_accepted_by_place_key(
            user_id, place_key, exclude_id=activity_id
        )
        if existing:
            await repo.update_activity_status(activity_id, user_id, "ignored")
            return OfflineActivityItem(**existing)
    was_ignored = activity["status"] == "ignored"
    if was_ignored:
        feedback_text = f"用户重新接受了活动推荐：{activity['title']}"
        chat_message = (
            f"好呀，我把「{activity['title']}」重新放回待出行里。"
            "等你想去的时候招呼我一声，到了现场点一下『我已经抵达这里』就行。"
        )
        memory_text = f"用户重新接受了线下活动推荐：{activity['title']}"
    else:
        feedback_text = f"用户想去看看：{activity['title']}"
        chat_message = (
            f"好，「{activity['title']}」我陪你一起去看看，先放进待出行里。"
            "到了现场记得点『我已经抵达这里』，我在这儿等你。"
        )
        memory_text = f"用户接受了线下活动推荐（想去看看）：{activity['title']}"
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
            status_label="待出行",
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
    if activity.get("reached"):
        raise HTTPException(status_code=409, detail="已经到达的旅途请先收好，不能再放回待考虑")
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
            message="好，那这个先放一放。下次我换一个更轻一点、更贴近你当下状态的选择。",
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


def _verify_arrival_distance(activity: dict, lat: float, lng: float) -> None:
    """spec §3.3/§4.5：仅点击确认时校验直线距离 ≤ 半径。无坐标时按配置拦截或放行。"""
    place_lat = activity.get("place_lat")
    place_lng = activity.get("place_lng")
    if place_lat is None or place_lng is None:
        if settings.offline_arrival_require_geocode:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "no_geocode",
                    "message": "这个地点还没定位到坐标，暂时没法确认到达",
                },
            )
        return  # 放行：无坐标跳过距离校验（体验优先，开关控制）
    distance = haversine_m(lat, lng, float(place_lat), float(place_lng))
    if distance > settings.offline_arrival_radius_m:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "too_far",
                "distance_m": round(distance),
                "message": "好像还没到附近，再走近一点再试试",
            },
        )


_ARRIVAL_GUIDE_FALLBACK = "到啦～慢慢逛。看到喜欢的瞬间，随手留一张就好。"


def _parse_text_json(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("text"):
            return str(data["text"]).strip()
    except Exception:
        match = re.search(r'"text"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1).strip()
    return ""


async def _arrival_guide_text(activity: dict, ctx: dict) -> str:
    """PM #8：到达后拍照引导（严禁泄露拍摄目标）。失败回退固定陪伴语。"""
    try:
        prompt = (await get_prompt_text("offline.arrival_guide")).format(
            location_name=activity.get("location_name") or activity.get("title") or "",
            activity_type=activity.get("category") or "线下活动",
            user_name=ctx.get("username") or ctx.get("user_name") or "你",
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return _parse_text_json(raw) or _ARRIVAL_GUIDE_FALLBACK
    except Exception as exc:
        logger.warning("[offline] 到达引导生成失败 err=%s", exc)
        return _ARRIVAL_GUIDE_FALLBACK


async def arrive_activity(
    user_id: str,
    activity_id: str,
    *,
    lat: float | None = None,
    lng: float | None = None,
) -> OfflineActivityItem:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    status = activity["status"]
    if status == "completed":
        raise HTTPException(status_code=409, detail="旅途已结束，可以去回顾看看")
    if status != "accepted":
        raise HTTPException(status_code=409, detail="先接受活动再确认到达")
    if activity.get("reached"):
        return OfflineActivityItem(**activity)  # 重复确认：幂等
    other_reached = await repo.find_other_reached_activity(
        user_id,
        activity.get("workspace_id"),
        exclude_id=activity_id,
    )
    if other_reached:
        raise HTTPException(
            status_code=409,
            detail="你还有一段正在进行的旅途，先把那一段收好再开始这里吧",
        )
    # 有客户端坐标才校验（地点已地理编码时才真正拦截 >200m）；无坐标荣誉制放行。
    if lat is not None and lng is not None:
        _verify_arrival_distance(activity, lat, lng)
    updated = await repo.mark_arrived(activity_id, user_id, lat=lat, lng=lng)
    if not updated:
        # 并发：别处已置为到达
        current = await repo.get_activity(activity_id, user_id, reveal_task=True)
        if current and current.get("reached"):
            return OfflineActivityItem(**current)
        other_reached = await repo.find_other_reached_activity(
            user_id,
            activity.get("workspace_id"),
            exclude_id=activity_id,
        )
        if other_reached:
            raise HTTPException(
                status_code=409,
                detail="你还有一段正在进行的旅途，先把那一段收好再开始这里吧",
            )
        raise HTTPException(status_code=409, detail="确认到达失败，请重试")
    # 到达后后台生成 3-5 拍摄物品 + 分档回忆预生成（PM #3/#4/#5-7，不阻塞、不披露）。
    fire_background(shooting_conditions.generate_items_for_activity(updated))
    ctx = await repo.resolve_user_context(user_id, activity.get("workspace_id"))
    if ctx:
        await insert_user_activity_card(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx["workspace_id"],
            activity=updated,
            trigger_type="offline_activity_arrived_card",
            status_label="我到了",  # spec §5.4-14 到达卡「我到了」
        )
        # 拍照引导（PM #8：走提示词，严禁泄露拍摄目标；失败回退固定陪伴语）。
        guide = await _arrival_guide_text(activity, ctx)
        guide_id = await emit_assistant(
            conversation_id=ctx.get("conversation_id"),
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message=guide,
            real_world_type="activity",
            source_id=activity_id,
            trigger_type="offline_activity_arrival_guide",
        )
        if guide_id and settings.offline_activity_companion_enabled:
            try:
                from app.services.offline.activity_companion import (
                    start_opening_segment,
                )

                await start_opening_segment(activity_id, user_id)
            except Exception as companion_err:  # noqa: BLE001 - arrival still succeeds
                logger.warning(
                    "[offline-companion] opening clock skipped activity=%s err=%s",
                    activity_id,
                    companion_err,
                )
    remember_user_event(
        user_id=user_id,
        workspace_id=activity.get("workspace_id"),
        text=f"用户确认到达线下活动地点：{activity['title']}",
    )
    return OfflineActivityItem(**updated)


async def draw_prophecy(user_id: str, activity_id: str) -> OfflineActivityItem:
    """spec §4.7：未到达前每活动最多抽一次；抽后返回带 prophecy_text 的活动。"""
    activity = await repo.get_activity(activity_id, user_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity["status"] != "accepted":
        raise HTTPException(status_code=409, detail="这个活动现在不能抽预言")
    if activity.get("reached"):
        raise HTTPException(status_code=409, detail="已经到达，预言只在出发前有效")
    if activity.get("prophecy_text"):
        return OfflineActivityItem(**activity)  # 幂等：已抽过返回原文
    updated = await repo.set_prophecy(
        activity_id, user_id, prophecy_pool.pick_prophecy()
    )
    if not updated:
        current = await repo.get_activity(activity_id, user_id)
        if current and current.get("prophecy_text"):
            return OfflineActivityItem(**current)
        raise HTTPException(status_code=409, detail="抽签失败，请重试")
    return OfflineActivityItem(**updated)


async def archive_activity(user_id: str, activity_id: str) -> OfflineActivityItem:
    """手动「收好这次旅途回忆」：accepted -> completed，推档案卡。幂等。"""
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity["status"] == "completed":
        return OfflineActivityItem(**await _with_completion_feedback(activity))
    if activity["status"] != "accepted":
        raise HTTPException(status_code=409, detail="这个活动现在不能归档")
    updated = await repo.mark_archived(activity_id, user_id, auto=False)
    if not updated:
        current = await repo.get_activity(activity_id, user_id, reveal_task=True)
        if current and current["status"] == "completed":
            return OfflineActivityItem(**await _with_completion_feedback(current))
        raise HTTPException(status_code=409, detail="归档失败，请重试")
    ctx = await repo.resolve_user_context(user_id, activity.get("workspace_id"))
    if ctx:
        await emit_activity_card(
            conversation_id=ctx.get("conversation_id"),
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            activity=updated,
            trigger_type="offline_activity_archived_card",
            status_label="已收好",
        )
    remember_user_event(
        user_id=user_id,
        workspace_id=activity.get("workspace_id"),
        text=f"用户收好了线下活动的回忆：{activity['title']}",
    )
    return OfflineActivityItem(**await _with_completion_feedback(updated))


async def auto_archive_due_activities() -> dict[str, int]:
    """spec §3.6：扫描确认到达满 24h 的进行中活动，自动归档并推档案卡 + 提示。

    单条失败记日志跳过、下周期重试（spec §6），不影响其它活动。
    """
    due = await repo.list_due_for_auto_archive()
    archived = 0
    for activity in due:
        try:
            updated = await repo.mark_archived(
                activity["id"], activity["user_id"], auto=True
            )
            if not updated:
                continue  # 并发下已被手动归档
            ctx = await repo.resolve_user_context(
                activity["user_id"], activity.get("workspace_id")
            )
            if ctx:
                await emit_activity_card(
                    conversation_id=ctx.get("conversation_id"),
                    user_id=activity["user_id"],
                    agent_id=ctx["agent_id"],
                    workspace_id=ctx["workspace_id"],
                    activity=updated,
                    trigger_type="offline_activity_auto_archived_card",
                    status_label="已收好",
                )
                await emit_assistant(
                    conversation_id=ctx.get("conversation_id"),
                    user_id=activity["user_id"],
                    agent_id=ctx["agent_id"],
                    workspace_id=ctx["workspace_id"],
                    message="已过 24 小时，这次旅途我先帮你收好啦，想回味随时来回顾看看。",
                    real_world_type="activity",
                    source_id=activity["id"],
                    trigger_type="offline_activity_auto_archived",
                )
            archived += 1
        except Exception as exc:
            logger.warning(
                "[offline-auto-archive] 单条归档失败 activity=%s err=%s",
                activity.get("id"), exc,
            )
    return {"scanned": len(due), "archived": archived}


def _review_event_tags(
    activity: dict, fragments: list[dict], gallery: list[str]
) -> list[str]:
    """spec §4.17 事件记录标签。"""
    tags: list[str] = []
    if activity.get("reached"):
        tags.append("打卡完成")
    if activity.get("status") == "completed":
        tags.append("行程已归档")
    if fragments:
        tags.append("思绪碎片已收藏")
    if gallery:
        tags.append("素材已归档")
    return tags


def _activity_cover(activity: dict, gallery: list[str]) -> str | None:
    images = activity.get("image_urls") or []
    if isinstance(images, list) and images:
        return str(images[0])
    return gallery[0] if gallery else None


async def get_review(user_id: str, activity_id: str) -> OfflineActivityReviewResponse:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    fragments = await repo.list_fragments(activity_id)
    gallery = await repo.list_gallery_media(activity_id)
    arrival_message_id = await repo.find_arrival_card_message_id(
        activity_id, activity.get("conversation_id")
    )
    return OfflineActivityReviewResponse(
        id=activity["id"],
        title=activity["title"],
        arrival_message_id=arrival_message_id,
        address=activity.get("address") or activity.get("location_name"),
        cover_url=_activity_cover(activity, gallery),
        started_at=activity.get("arrival_confirmed_at") or activity.get("created_at"),
        ended_at=activity.get("completed_at") or activity.get("archived_at"),
        story=activity.get("description") or activity.get("summary") or "你把这一天慢慢走完了。",
        gallery=gallery,
        fragments=[
            OfflineActivityFragmentItem(
                id=f["id"], tier=f["tier"], text=f["text"], lead_in=f.get("lead_in")
            )
            for f in fragments
        ],
        event_tags=_review_event_tags(activity, fragments, gallery),
        has_memory_note=bool(activity.get("travel_note")),
        travel_note=activity.get("travel_note"),
    )


_TIER_EMOJI = {"rare": "💭", "epic": "📜", "legendary": "🔮"}


async def generate_memory_note(
    user_id: str, activity_id: str
) -> OfflineMemoryNoteResponse:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    fragments = await repo.list_fragments(activity_id)
    note = activity.get("travel_note")
    mood_tags: list[str] = []
    if not note:  # 幂等：已生成直接复用，未生成才调 LLM 并缓存
        voice_transcripts = await repo.list_voice_transcripts(activity_id)
        generated = await memory_note_gen.generate_note(
            activity_info=_activity_info_text(activity),
            dialogue="",
            voice_transcripts="\n".join(voice_transcripts),
            photo_keywords="",
            fragments="\n".join(f["text"] for f in fragments if f.get("text")),
        )
        note = (generated.get("body") or "").strip() or memory_note_gen.fallback_body()
        mood_tags = generated.get("mood_tags") or []
        await repo.set_travel_note(activity_id, user_id, note)
    gallery = await repo.list_gallery_media(activity_id)
    fragment_tags = [
        f"{_TIER_EMOJI.get(f['tier'], '💭')} {TIER_LABELS.get(f['tier'], '片刻感想')}"
        for f in fragments
    ]
    return OfflineMemoryNoteResponse(
        title=activity["title"],
        date_text=activity.get("arrival_confirmed_at") or activity.get("created_at") or "",
        cover_url=_activity_cover(activity, gallery),
        travel_note=note,
        fragment_tags=fragment_tags,
        mood_tags=mood_tags,
    )


def _activity_info_text(activity: dict) -> str:
    parts = [
        f"标题：{activity.get('title') or ''}",
        f"类型：{activity.get('category') or ''}",
        f"地点：{activity.get('location_name') or activity.get('address') or ''}",
        f"简介：{activity.get('summary') or activity.get('description') or ''}",
    ]
    return "｜".join(p for p in parts if p.split("：", 1)[-1].strip())


async def complete_activity(
    user_id: str,
    activity_id: str,
    *,
    text: str,
    photo_attachment_ids: list[str],
    audio_attachment_id: str | None = None,
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
    media_ids = list(photo_attachment_ids)
    if audio_attachment_id:
        media_ids.append(audio_attachment_id)
    found: list[activity_media_repo.OfflineActivityMedia] = []
    if media_ids:
        found = await activity_media_repo.get_activity_media(
            media_ids=media_ids,
            user_id=user_id,
            recommendation_id=activity_id,
        )
        if len(found) != len(media_ids):
            raise HTTPException(status_code=400, detail="Invalid activity media")
    by_id = {item.id: item for item in found}
    photos = [by_id[item_id] for item_id in photo_attachment_ids if item_id in by_id]
    audio = by_id.get(audio_attachment_id) if audio_attachment_id else None
    if any(item.kind != "image" for item in photos):
        raise HTTPException(status_code=400, detail="Invalid activity photo")
    if audio and audio.kind != "audio":
        raise HTTPException(status_code=400, detail="Invalid activity audio")
    updated = await repo.update_activity_status(activity_id, user_id, "completed")
    if not updated:
        raise HTTPException(status_code=404, detail="Activity not found")
    await repo.create_activity_feedback(
        recommendation_id=activity_id,
        user_id=user_id,
        kind="completion",
        text=text,
        photo_attachment_ids=photo_attachment_ids,
        audio_attachment_id=audio_attachment_id if audio else None,
    )
    media_for_message = [*photos, *([audio] if audio else [])]
    if conversation_id and (text.strip() or media_for_message):
        await insert_user_component_message(
            conversation_id=conversation_id,
            workspace_id=activity.get("workspace_id"),
            content=text.strip() or "分享了活动完成情况",
            metadata={
                "real_world_type": "activity",
                "source_id": activity_id,
                "trigger_type": "offline_activity_completion_share",
                "attachments": [_media_to_metadata(item) for item in media_for_message],
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
        text=(
            f"用户完成了线下活动「{activity['title']}」。"
            f"分享内容：{text}"
            + ("（包含语音）" if audio else "")
        ),
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
        "duration_seconds": media.duration_seconds,
        "url": media.url,
        "vision_status": "ready",
    }
