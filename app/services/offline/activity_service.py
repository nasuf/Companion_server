from __future__ import annotations

import logging
import math
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
from app.services.offline.content import plain_text
from app.services.offline.geocode import geocode_address, haversine_m, make_place_key, wgs84_to_gcj02
from app.services.offline.prompt_fields import (
    filled,
    format_moment,
    location_fields,
    parse_text_field,
)
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
        if source == "manual":
            raise HTTPException(status_code=422, detail={
                "reason": "conversation_required", "message": "请先创建或打开一个聊天会话，再生成活动。",
            })
        return None
    city, search_anchor, match_terms = _location_for_activity(ctx)
    if not city:
        if source == "manual":
            raise HTTPException(status_code=422, detail={
                "reason": "location_required", "message": "还没有获取到所在城市，请更新定位后再生成活动。",
            })
        return None
    async with offline_trace(
        "activity_invite",
        conversation_id=ctx["conversation_id"],
        agent_id=ctx.get("agent_id"), user_id=user_id,
    ) as tracer:
        card = await generate_activity_card(
            user_id=user_id,
            workspace_id=ctx["workspace_id"],
            city=city,
            source=source,
            conversation_id=ctx["conversation_id"],
            search_location=search_anchor,
            location_terms=list(match_terms),
            center=(float(ctx['user_location_latitude']), float(ctx['user_location_longitude']))
            if ctx.get('user_location_latitude') is not None and ctx.get('user_location_longitude') is not None else None,
        )
        if not card:
            if source == "manual":
                raise HTTPException(status_code=503, detail={
                    "reason": "no_suitable_activity", "message": "暂时没找到合适的新去处，请稍后再试。",
                })
            return None
        # 地理编码：地址 -> 经纬度（供到达距离校验）+ 同地点去重键。key 未配置或失败
        # 时 coords=None，不阻断推荐（到达校验按 offline_arrival_require_geocode 处理）。
        metadata = card.get('discovery_metadata') or {}
        if metadata.get('coordinate_source') == 'native_poi':
            place_lat, place_lng = card['place_lat'], card['place_lng']
            place_key = metadata.get('session_key') or ('poi:' + metadata['poi_id'])
        else:
            coords = await geocode_address(card.get("address"), card.get("city") or city)
            place_lat, place_lng = coords if coords else (None, None)
            place_key = make_place_key(card.get("location_name"), card.get("address"), card.get("city") or city)
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
            extra_metadata={
                "recommendation_message_status": (activity.get("discovery_metadata") or {}).get("recommendation_message_status", "not_generated"),
                "user_relevance_count": (activity.get("discovery_metadata") or {}).get("user_relevance_count", 0),
                "recommendation_evidence_status": (activity.get("discovery_metadata") or {}).get("recommendation_evidence_status", {}),
            },
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
    await _guard_event(activity)
    if activity['status'] == 'accepted':
        return OfflineActivityItem(**activity)
    # spec §4.4 同地点复用：接受新推荐时若同地点已有进行中活动，直接返回既有（前端
    # 打开其打卡页），并把当前这条收进「暂不考虑」，避免同地点重复占列表。
    place_key = activity.get("place_key")
    if activity["status"] == "pending" and place_key:
        existing = await repo.find_accepted_by_place_key(
            user_id, place_key, exclude_id=activity_id
        )
        if existing:
            raise HTTPException(status_code=409, detail="这个地点已有待出行活动，请从待出行列表打开，当前推荐会保留")
    was_ignored = activity["status"] == "ignored"
    if was_ignored:
        feedback_text = f"用户重新接受了活动推荐：{activity['title']}"
        chat_message = (
            f"好呀，我把「{activity['title']}」重新放回待出行里。"
            "等你想去的时候招呼我一声，到了现场点一下『我已经抵达这里』就行。"
        )
    else:
        feedback_text = f"用户想去看看：{activity['title']}"
        chat_message = (
            f"好，「{activity['title']}」我陪你一起去看看，先放进待出行里。"
            "到了现场记得点『我已经抵达这里』，我在这儿等你。"
        )
    trigger_type = (
        "offline_activity_reaccepted" if was_ignored else "offline_activity_accepted"
    )
    if (activity.get('discovery_metadata') or {}).get('provider') == 'cleversee':
        updated, changed = await repo.accept_native_activity(activity_id, user_id)
        if not updated:
            raise HTTPException(status_code=409, detail='这项活动已有出行安排，请从待出行列表打开')
        if not changed:
            return OfflineActivityItem(**updated)
    else:
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
    # A cancellable intention lives in the activity state, not permanent memory.
    # Actual user dialogue and completed journey evidence retain their own pipeline.
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
                    "message": "暂时没能确认到达，请稍后再试",
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


async def _guard_event(activity: dict, *, arrival: bool = False) -> None:
    from app.services.offline.discovery_facts import event_is_available
    from app.services.offline.cleversee_discovery import refresh_event
    metadata = activity.get('discovery_metadata') or {}
    if metadata.get('kind') != 'event':
        return
    if not event_is_available(activity, arrival=arrival):
        raise HTTPException(status_code=409, detail='这场活动现在还不能前往，看看其他安排吧')
    refreshed = await refresh_event(activity)
    if refreshed:
        persisted = await repo.save_discovery_metadata(activity['id'], activity['user_id'], refreshed)
        if persisted is None:
            raise HTTPException(status_code=404, detail='Activity not found')
        activity['discovery_metadata'] = persisted
        if not event_is_available(activity, arrival=arrival):
            raise HTTPException(status_code=409, detail='这场活动的安排有变化，先看看其他活动吧')
    else:
        raise HTTPException(status_code=503, detail='这场活动的安排还需确认，稍后再试试吧')


_ARRIVAL_GUIDE_FALLBACK = "到啦，慢慢逛就好，想聊时给我发消息。"


async def _arrival_guide_text(activity: dict, ctx: dict) -> str:
    """First proactive line after arrival. Falls back to a fixed companion line."""
    try:
        from app.services.offline.activity_message_context import message_context

        prompt = (await get_prompt_text("offline.arrival_guide")).format(
            **await message_context(ctx),
            **location_fields(
                activity,
                city_fallback=str(ctx.get("user_location_city") or ""),
            )
        )
        raw = await invoke_text(get_chat_model(), prompt)
        return parse_text_field(raw) or _ARRIVAL_GUIDE_FALLBACK
    except Exception as exc:
        logger.warning("[offline] 到达引导生成失败 err=%s", exc)
        return _ARRIVAL_GUIDE_FALLBACK


async def arrive_activity(
    user_id: str,
    activity_id: str,
    *,
    lat: float | None = None,
    lng: float | None = None,
    accuracy_m: float | None = None,
    manual_confirmation: bool = False,
    observed_at: datetime | None = None,
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
    await _guard_event(activity, arrival=True)
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
    has_place = activity.get('place_lat') is not None and activity.get('place_lng') is not None
    if not manual_confirmation:
        if not has_place:
            raise HTTPException(status_code=422, detail={"reason": "no_geocode", "message": "暂时没能确认到达，请稍后再试"})
        if lat is None or lng is None or not all(math.isfinite(v) for v in (lat, lng)):
            raise HTTPException(status_code=422, detail={"reason": "location_required", "message": "需要当前位置才能确认到达，请开启定位后重试"})
        if accuracy_m is None or not math.isfinite(accuracy_m) or not 0 <= accuracy_m <= 100:
            raise HTTPException(status_code=422, detail={"reason": "low_accuracy", "message": "定位还不够准确，请开启精确定位后再试一次"})
        if observed_at is not None and (observed_at.tzinfo is None or
                not -30 <= (datetime.now(UTC) - observed_at).total_seconds() <= 120):
            raise HTTPException(status_code=422, detail={'reason': 'stale_location', 'message': '再获取一下当前位置试试'})
        crs = (activity.get('discovery_metadata') or {}).get('coordinate_system', 'gcj02')
        if crs not in {'gcj02', 'wgs84'}:
            raise HTTPException(status_code=422, detail={'reason': 'no_geocode', 'message': '暂时没能确认到达，请稍后再试'})
        check_lat, check_lng = wgs84_to_gcj02(lat, lng) if crs == 'gcj02' else (lat, lng)
        _verify_arrival_distance(activity, check_lat, check_lng)
    # Explicit user confirmation overrides only location checks. It is a user
    # statement, not a GPS-verified observation or an inferred device position.
    updated = await repo.mark_arrived(
        activity_id, user_id,
        lat=None if manual_confirmation else lat,
        lng=None if manual_confirmation else lng,
        verified=has_place and not manual_confirmation,
    )
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
        # 到达后首条主动消息。失败回退固定陪伴语。
        guide = await _arrival_guide_text(updated, ctx)
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
    await _ensure_journey_note(updated)
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
            await _ensure_journey_note(updated)
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
    if activity["status"] != "completed":
        raise HTTPException(status_code=409, detail="旅途还在进行中，收好后再来回顾")
    fragments = await repo.list_fragments(activity_id)
    gallery = await repo.list_gallery_media(activity_id)
    arrival_message_id = await repo.find_arrival_card_message_id(
        activity_id, activity.get("conversation_id")
    )
    evidence = await repo.journey_evidence(activity)
    note = await _ensure_journey_note(activity, evidence)
    return OfflineActivityReviewResponse(
        id=activity["id"],
        title=activity["title"],
        arrival_message_id=arrival_message_id,
        address=activity.get("address") or activity.get("location_name"),
        cover_url=_activity_cover(activity, gallery),
        image_urls=activity.get("image_urls") or [],
        started_at=activity.get("arrival_confirmed_at") or activity.get("created_at"),
        ended_at=activity.get("completed_at") or activity.get("archived_at"),
        story=note or _arrival_only_story(activity),
        can_generate_memory_note=activity["status"] == "completed" and _has_journey_evidence(evidence),
        gallery=gallery,
        fragments=[
            OfflineActivityFragmentItem(
                id=f["id"], tier=f["tier"], text=f["text"], lead_in=f.get("lead_in")
            )
            for f in fragments
        ],
        event_tags=_review_event_tags(activity, fragments, gallery),
        has_memory_note=bool(note),
        travel_note=note,
    )


_TIER_EMOJI = {"rare": "💭", "epic": "📜", "legendary": "🔮"}


def _has_journey_evidence(evidence: dict) -> bool:
    return bool(evidence.get('dialogue') or evidence.get('voice') or evidence.get('feedback') or evidence.get('photo_count'))


def _arrival_only_story(activity: dict) -> str:
    place = plain_text(activity.get('location_name') or activity.get('title'))
    if activity.get('reached'):
        return f"你确认到了{place}，随后收好了这次旅途。还没有留下照片或感想，就先记下这次到达。"
    return f"这次去{place}的计划已经收好，还没有确认到达或留下旅途记录。"


async def _ensure_journey_note(activity: dict, evidence: dict | None = None) -> str | None:
    if activity.get('status') != 'completed':
        return None
    evidence = evidence if evidence is not None else await repo.journey_evidence(activity)
    if not _has_journey_evidence(evidence):
        return None
    if activity.get('travel_note') and activity.get('travel_note_version') == 2:
        return activity['travel_note']
    generated = await memory_note_gen.generate_note(
        activity_name=filled(activity.get('title'), empty='这次外出'),
        location=filled(activity.get('location_name'), empty='（未提供）'),
        activity_type=filled(activity.get('category'), empty='线下活动'),
        arrival_time=format_moment(activity.get('arrival_confirmed_at')),
        weather='（未提供）', dialogue=_format_note_dialogue(evidence['dialogue']),
        voice_transcripts='\n'.join(evidence['voice'] + evidence['feedback']),
        photo_keywords='\n'.join(evidence['photos']) or (f"用户分享了{evidence['photo_count']}张照片，内容未识别。" if evidence['photo_count'] else ''),
        fragments='',  # AI-written memories cannot establish the user's experience.
    )
    note = plain_text(generated.get('body'))[:300]
    # Reject invented photo actions even if the model ignores the evidence-only prompt.
    if not evidence['photo_count'] and re.search(r'照片|拍照|拍下|拍了|分享.{0,6}图片', note):
        note = ''
    if not note:
        # Fail closed to an extractive record; never invent weather, feelings or actions.
        quotes = [str(r['content']) for r in evidence['dialogue']] + evidence['voice'] + evidence['feedback']
        note = (f"这次在{plain_text(activity.get('location_name') or activity['title'])}，你留下了这些话："
                + '；'.join('「'+plain_text(q)[:80]+'」' for q in quotes[:2])) if quotes else f"这次旅途你分享了{evidence['photo_count']}张照片，已经为你收好。"
    await repo.set_travel_note(activity['id'], activity['user_id'], note)
    saved = await repo.get_activity(activity['id'], activity['user_id'])
    return (saved or {}).get('travel_note') or note


async def generate_memory_note(user_id: str, activity_id: str) -> OfflineMemoryNoteResponse:
    activity = await repo.get_activity(activity_id, user_id, reveal_task=True)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity['status'] != 'completed':
        raise HTTPException(status_code=409, detail="先收好这次旅途，再整理手札")
    note = await _ensure_journey_note(activity)
    if not note:
        raise HTTPException(status_code=409, detail="这次只记录了到达，还没有照片或感想可以整理成手札")
    fragments = await repo.list_fragments(activity_id)
    gallery = await repo.list_gallery_media(activity_id)
    return OfflineMemoryNoteResponse(
        title=activity['title'], date_text=activity.get('arrival_confirmed_at') or '',
        cover_url=_activity_cover(activity, gallery), travel_note=note,
        fragment_tags=[f"{_TIER_EMOJI.get(f['tier'], '💭')} {TIER_LABELS.get(f['tier'], '片刻感想')}" for f in fragments],
        mood_tags=[],
    )


def _format_note_dialogue(rows: list[dict]) -> str:
    lines: list[str] = []
    for row in rows:
        text = str(row.get("content") or "").replace("\n", " ").strip()
        if not text:
            continue
        role = "用户" if row.get("role") == "user" else "好友"
        stamp = format_moment(row.get("created_at"))
        prefix = f"[{stamp}] " if stamp else ""
        lines.append(f"{prefix}{role}：{text[:180]}")
    return "\n".join(lines)[-4000:]


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


async def delete_activity(user_id: str, activity_id: str) -> dict:
    activity = await repo.get_activity(activity_id, user_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity['status'] == 'cancelled':
        return {'ok': True}
    if not await repo.cancel_unstarted_activity(activity_id, user_id):
        raise HTTPException(status_code=409, detail="已经到达的旅途请收好回忆，不能删除待出行记录")
    return {'ok': True}
