"""Web 后台「礼物管理」：管理员线下采购、手动录入礼物与物流，寄给指定用户。

和自动送礼共用同一套状态机与用户侧逻辑（gift_service）：寄出提示、派送提醒、签收揭晓
都走同一条路径，保证用户看到的体验一致。区别只在于 provider='manual'（见
gift_status.MANUAL_PROVIDER）：不会调用任何下单 / 物流接口，轨迹完全由后台录入。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException

from app.models.admin_gifts import (
    AdminGiftActionResponse,
    AdminGiftCreateRequest,
    AdminGiftDetail,
    AdminGiftListResponse,
    AdminGiftRecommendationRequest,
    AdminGiftRecommendationResponse,
    AdminGiftUpdateRequest,
    AdminGiftUserOverview,
)
from app.observability.events import (
    EVT_GIFT_ADMIN_SENT,
    EVT_GIFT_ADMIN_UPDATED,
    EVT_GIFT_CANCELLED,
    EVT_GIFT_STAGE_CHANGED,
)
from app.services.offline import activity_media_storage, gift_fulfillment, gift_recommendation, gift_service
from app.services.offline import gift_admin_repository as admin_repo
from app.services.offline import gift_repository as gift_repo
from app.services.offline import repository as repo
from app.services.offline.chat_emit import emit_gift_card, offline_trace
from app.services.offline.gift_budget import available_gift_budget_cents
from app.services.offline.gift_messages import gift_sent_message
from app.services.offline.gift_status import (
    CONFIRMABLE_STATUS,
    IN_TRANSIT_STATUSES,
    transit_rank,
    gift_status_for_event,
)
from app.services.offline.module_settings import is_gift_enabled
from app.services.offline.providers.gift_types import GiftProviderError

logger = logging.getLogger(__name__)


async def user_overview(user_id: str) -> AdminGiftUserOverview:
    user = await admin_repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    workspaces = await admin_repo.list_user_workspaces(user_id)
    primary = workspaces[0] if workspaces else None
    workspace_id = primary["workspace_id"] if primary else None
    address = await gift_repo.default_address(user_id, masked=False)
    gifts, _ = await admin_repo.list_gifts(user_id=user_id, limit=50)
    try:
        provider = gift_fulfillment.commerce_provider_name()
    except GiftProviderError:
        provider = "unconfigured"
    return AdminGiftUserOverview(
        user=user,
        workspaces=workspaces,
        address=address,
        budget={
            "recharge_total_cents": await gift_repo.recharge_total_cents(user_id),
            "historical_spend_cents": await gift_repo.historical_gift_spend_cents(user_id),
            "available_cents": await available_gift_budget_cents(user_id),
        },
        tags=await repo.list_user_tags(
            user_id, workspace_id, agent_id=primary["agent_id"] if primary else None, limit=12,
        ),
        memory_count=await admin_repo.count_user_memories(user_id, workspace_id),
        gifts=gifts,
        auto_gift_enabled=await is_gift_enabled(),
        commerce_provider=provider,
    )


async def recommend(user_id: str, data: AdminGiftRecommendationRequest) -> AdminGiftRecommendationResponse:
    if not await admin_repo.get_user(user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
    workspace = await _workspace_for(user_id, data.workspace_id, required=False)
    result = await gift_recommendation.generate_recommendations(
        user_id=user_id,
        workspace=workspace,
        budget_min_yuan=data.budget_min_yuan,
        budget_max_yuan=data.budget_max_yuan,
        exclude_names=data.exclude_names,
        count=data.count,
    )
    return AdminGiftRecommendationResponse(**result)


async def list_gifts(
    *,
    status_filter: str,
    user_id: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> AdminGiftListResponse:
    items, total = await admin_repo.list_gifts(
        status_filter=status_filter, user_id=user_id, query=query, limit=limit, offset=offset,
    )
    return AdminGiftListResponse(items=items, total=total)


async def gift_detail(gift_id: str) -> AdminGiftDetail:
    gift = await _require_gift(gift_id)
    return AdminGiftDetail(
        **gift,
        events=await admin_repo.list_tracking_events(gift_id),
        messages=await admin_repo.gift_messages(gift_id, gift["user_id"]),
    )


async def create_manual_gift(admin_id: str, data: AdminGiftCreateRequest) -> AdminGiftActionResponse:
    gift_id = _gift_id_from_request(data.request_id)
    existing = await admin_repo.get_gift(gift_id)
    if existing:
        # 同一次提交重放（网络重试 / 双击 / 上次中途失败）：不再新建，只补齐没完成的步骤。
        if existing["user_id"] != data.user_id:
            raise HTTPException(status_code=409, detail="礼物编号冲突，请刷新页面后重试")
        return await _finish_manual_send(admin_id, existing, data)

    if not await admin_repo.get_user(data.user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
    workspace = await _workspace_for(data.user_id, data.workspace_id, required=True)
    assert workspace is not None  # required=True 时没有工作区会直接 409
    ctx = await repo.resolve_user_context(data.user_id, workspace["workspace_id"])
    if not ctx or not ctx.get("conversation_id"):
        raise HTTPException(status_code=409, detail="该用户在这个 AI 伙伴下还没有聊天会话，礼物卡片无处发送")
    address = await gift_repo.default_address(data.user_id, masked=False)
    if not address:
        raise HTTPException(status_code=409, detail="该用户还没有填写收货地址，无法寄出")

    image_url, stored_key = _resolve_image(data.user_id, data, current=None)
    try:
        gift, created = await admin_repo.insert_manual_gift({
            "id": gift_id,
            "user_id": data.user_id,
            "agent_id": ctx["agent_id"],
            "workspace_id": ctx["workspace_id"],
            "conversation_id": ctx["conversation_id"],
            "status": data.initial_status,
            "gift_name": data.gift_name,
            "gift_reason": data.gift_reason or None,
            "gift_note": data.gift_note or None,
            "product_image_url": image_url,
            "product_snapshot": {
                "source": "admin_manual",
                "created_by_admin_id": admin_id,
                "recommendation": data.recommendation,
            },
            "paid_amount_cents": _cents(data.paid_amount_yuan),
            "tracking_number": data.tracking_number or None,
            "logistics_provider": data.carrier or None,
            "address_snapshot": address,
        })
    except Exception:
        activity_media_storage.delete_media_file(stored_key)
        raise
    if not created:
        # 并发的同一提交抢先插入了：本次上传的图片作废。
        activity_media_storage.delete_media_file(stored_key)
    if not gift:
        raise HTTPException(status_code=409, detail="礼物编号冲突，请刷新页面后重试")
    return await _finish_manual_send(admin_id, gift, data)


async def _finish_manual_send(
    admin_id: str,
    gift: dict[str, Any],
    data: AdminGiftCreateRequest,
) -> AdminGiftActionResponse:
    """寄出后的步骤全部幂等，重放时补齐上次没做完的部分，不会重复打扰用户。"""
    gift_id = gift["id"]
    if gift["status"] == "cancelled":
        return AdminGiftActionResponse(gift=await gift_detail(gift_id), notification="skipped")
    await gift_repo.add_tracking_events(gift_id, _initial_events(data))  # 按内容去重
    ctx = {
        "workspace_id": gift.get("workspace_id"),
        "agent_id": gift["agent_id"],
        "conversation_id": gift.get("conversation_id"),
    }
    await gift_repo.update_last_gift_paid(data.user_id, ctx["agent_id"], ctx["workspace_id"])
    notification, message = "skipped", None
    if not await gift_repo.gift_notice_sent(gift_id, data.user_id, "gift_sent"):
        notification, message = await _announce_sent(data.user_id, gift, ctx)
        if notification == "sent":
            gift_service.remember_mystery_gift_sent(data.user_id, ctx["workspace_id"])
    logger.info(
        "[gift-admin] manual gift sent gift_id=%s user_id=%s notification=%s",
        gift_id,
        data.user_id,
        notification,
        extra={
            "event": EVT_GIFT_ADMIN_SENT,
            "gift_id": gift_id,
            "user_id": data.user_id,
            "admin_id": admin_id,
            "initial_status": data.initial_status,
            "notification": notification,
        },
    )
    return AdminGiftActionResponse(
        gift=await gift_detail(gift_id),
        notification=notification,
        assistant_message=message,
    )


async def update_gift(admin_id: str, gift_id: str, data: AdminGiftUpdateRequest) -> AdminGiftDetail:
    gift = await _require_gift(gift_id)
    provided = data.model_fields_set
    fields: dict[str, Any] = {}
    for name in ("gift_name", "gift_note", "gift_reason", "tracking_number"):
        if name in provided:
            fields[name] = getattr(data, name) or None
    if "gift_name" in fields and not fields["gift_name"]:
        raise HTTPException(status_code=422, detail="礼物名称不能为空")
    if "carrier" in provided:
        fields["logistics_provider"] = data.carrier or None
    if "paid_amount_yuan" in provided and data.paid_amount_yuan is not None:
        fields["paid_amount_cents"] = _cents(data.paid_amount_yuan)
    stored_key = None
    if provided & {"image_url", "image_base64"}:
        fields["product_image_url"], stored_key = _resolve_image(
            gift["user_id"], data, current=gift.get("product_image_url"),
        )
    updated = await admin_repo.update_gift_details(gift_id, fields)
    if not updated:
        activity_media_storage.delete_media_file(stored_key)
        raise HTTPException(status_code=409, detail="已撤销的礼物不能再修改")
    logger.info(
        "[gift-admin] gift updated gift_id=%s fields=%s",
        gift_id,
        sorted(fields),
        extra={"event": EVT_GIFT_ADMIN_UPDATED, "gift_id": gift_id, "admin_id": admin_id, "fields": sorted(fields)},
    )
    return await gift_detail(gift_id)


async def set_stage(admin_id: str, gift_id: str, stage: str) -> AdminGiftActionResponse:
    """把在途礼物推进（或纠错回退）到 下单 / 运输 / 派送 中的某个节点。

    签收是第四个节点，只能由用户在 App 里确认，后台不能代办。回退时删掉更靠后的节点，
    这样误点「派送」后退回运输，用户就不能再对没到的包裹签收。
    """
    gift = await _require_gift(gift_id)
    current = gift["status"]
    if current not in IN_TRANSIT_STATUSES:
        raise HTTPException(status_code=409, detail="只有在途的礼物可以调整物流进度（签收只能由用户在 App 里完成）")
    if stage == current:
        raise HTTPException(status_code=409, detail="礼物已经在这个节点了")
    if transit_rank(stage) < transit_rank(current):
        later = [s for s in IN_TRANSIT_STATUSES if transit_rank(s) > transit_rank(stage)]
        await admin_repo.delete_events_after_stage(gift_id, later + (["packed"] if stage == "ordered" else []))
    events = await admin_repo.list_tracking_events(gift_id)
    if not any(gift_status_for_event(e["status"]) == stage for e in events):
        await admin_repo.insert_tracking_event(gift_id, _stage_event(stage, gift.get("carrier")))
    applied, notification = await _sync_status_with_latest_event(gift)
    logger.info(
        "[gift-admin] stage changed gift_id=%s %s -> %s notification=%s",
        gift_id,
        current,
        stage,
        notification,
        extra={
            "event": EVT_GIFT_STAGE_CHANGED,
            "gift_id": gift_id,
            "admin_id": admin_id,
            "from_status": current,
            "to_status": applied,
            "notification": notification,
        },
    )
    return AdminGiftActionResponse(gift=await gift_detail(gift_id), notification=notification)


async def _sync_status_with_latest_event(gift: dict[str, Any]) -> tuple[str | None, str]:
    """在途礼物的状态跟随时间最新的物流节点；返回 (新状态或 None, 通知结果)。

    同一分钟内以后录入的为准。进入派送中时，同一份礼物只提醒一次签收。
    """
    events = [e for e in await admin_repo.list_tracking_events(gift["id"]) if gift_status_for_event(e["status"])]
    latest = max(
        events,
        key=lambda e: (_minute(_parse_iso(e["occurred_at"])), _parse_iso(e.get("created_at")) or _EPOCH),
        default=None,
    )
    target = gift_status_for_event(latest["status"]) if latest else "ordered"
    if target is None or target == gift["status"]:
        return None, "skipped"
    if target != CONFIRMABLE_STATUS:
        await admin_repo.set_transit_status(gift["id"], target)
        return target, "skipped"
    moved = await gift_service.advance_to_out_for_delivery(gift["user_id"], gift, notify=False)
    if not moved or await gift_repo.gift_notice_sent(gift["id"], gift["user_id"], "gift_out_for_delivery"):
        return target, "skipped"
    message_id = await gift_service.notify_out_for_delivery(gift["user_id"], moved)
    return target, "sent" if message_id else "failed"


async def cancel_gift(admin_id: str, gift_id: str, reason: str) -> AdminGiftDetail:
    await _require_gift(gift_id)
    if not await admin_repo.cancel_gift(gift_id, reason.strip()):
        raise HTTPException(status_code=409, detail="已签收或已关闭的礼物不能撤销")
    logger.info(
        "[gift-admin] gift cancelled gift_id=%s",
        gift_id,
        extra={"event": EVT_GIFT_CANCELLED, "gift_id": gift_id, "admin_id": admin_id},
    )
    return await gift_detail(gift_id)


async def _announce_sent(user_id: str, gift: dict[str, Any], ctx: dict[str, Any]) -> tuple[str, str | None]:
    """神秘礼物卡片 + AI「有惊喜在路上」。失败不回滚礼物：运营可在详情里看到通知失败。"""
    try:
        async with offline_trace(
            "gift_sent",
            conversation_id=ctx["conversation_id"],
            agent_id=ctx["agent_id"], user_id=user_id,
        ) as tracer:
            message = await gift_sent_message(user_id, ctx["workspace_id"])
            message_id = await emit_gift_card(
                conversation_id=ctx["conversation_id"],
                user_id=user_id,
                agent_id=ctx["agent_id"],
                workspace_id=ctx["workspace_id"],
                gift=gift,
                trigger_type="gift_sent",
                status_label="在路上",
                message=message,
                trace_id=tracer.safe_trace_id,
            )
        return ("sent", message) if message_id else ("failed", None)
    except Exception as exc:
        logger.warning("[gift-admin] sent notification failed gift_id=%s: %s", gift.get("id"), exc)
        return "failed", None


_STAGE_EVENTS = {
    "ordered": ("礼物已下单", "TA 为你挑的礼物已经下单，正在准备发出"),
    "shipping": ("包裹运输中", "包裹正在运输途中"),
    "out_for_delivery": ("快递员正在派送", "收到后记得在礼物页确认签收，签收后就能揭晓礼物啦。"),
}


def _stage_event(stage: str, carrier: str | None = None) -> dict[str, Any]:
    title, description = _STAGE_EVENTS[stage]
    if stage == "shipping" and carrier:
        description = f"由{carrier}承运，正在运输途中"
    return {
        "status": stage,
        "title": title,
        "description": description,
        "location": None,
        "occurred_at": datetime.now(UTC),
    }


def _initial_events(data: AdminGiftCreateRequest) -> list[dict[str, Any]]:
    events = [_stage_event("ordered")]
    if data.initial_status == "shipping":
        events.append(_stage_event("shipping", data.carrier))
    return events


def _resolve_image(user_id: str, data: Any, *, current: str | None) -> tuple[str | None, str | None]:
    """返回 (图片 URL, 本次新存的文件 key)。上传的图片按收件用户存放，只有该用户和管理员能读。"""
    if data.image_base64:
        mime = activity_media_storage.normalize_image_mime(data.image_mime)
        blob = activity_media_storage.decode_image_base64(data.image_base64)
        key = activity_media_storage.save_image_blob(user_id=user_id, blob=blob, mime=mime, kind="gift")
        return activity_media_storage.media_url(key), key
    if data.image_url is None:
        return (current if "image_url" not in data.model_fields_set else None), None
    return _validate_image_url(user_id, data.image_url), None


def _validate_image_url(user_id: str, value: str) -> str | None:
    url = value.strip()
    if not url:
        return None
    prefix = activity_media_storage.media_url("")
    if url.startswith(prefix):
        key = url[len(prefix):]
        activity_media_storage.storage_path(key)  # 拒绝路径穿越
        if not key.startswith(f"{user_id}_"):
            raise HTTPException(status_code=422, detail="只能引用该用户自己的图片")
        return url
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="图片地址需要是 http(s) 链接")
    return url


async def _workspace_for(user_id: str, workspace_id: str | None, *, required: bool) -> dict[str, Any] | None:
    workspaces = await admin_repo.list_user_workspaces(user_id)
    if workspace_id:
        match = next((w for w in workspaces if w["workspace_id"] == workspace_id), None)
        if not match:
            raise HTTPException(status_code=404, detail="该用户没有这个 AI 伙伴工作区")
        return match
    if required and not workspaces:
        raise HTTPException(status_code=409, detail="该用户还没有 AI 伙伴，礼物无法寄出")
    return workspaces[0] if workspaces else None


async def _require_gift(gift_id: str) -> dict[str, Any]:
    gift = await admin_repo.get_gift(gift_id)
    if not gift:
        raise HTTPException(status_code=404, detail="礼物不存在")
    return gift


def _gift_id_from_request(request_id: str) -> str:
    try:
        return uuid.UUID(request_id.strip()).hex
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="request_id 需要是 UUID") from exc


def _cents(yuan: float) -> int:
    return int(round(yuan * 100))


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _minute(value: datetime | None) -> datetime:
    return (value or _EPOCH).replace(second=0, microsecond=0)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None
