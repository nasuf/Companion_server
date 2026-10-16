from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import HTTPException

from app.models.offline import (
    GiftAddressRequest,
    GiftAddressResponse,
    GiftReceiptResponse,
    GiftThanksResponse,
    GiftTrackingEvent,
    GiftTrackingResponse,
    GiftYearGroup,
    GiftsHomeResponse,
    RealWorldGiftItem,
)
from app.observability.events import EVT_GIFT_OUT_FOR_DELIVERY, EVT_GIFT_RECEIVED
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import gift_fulfillment
from app.services.offline import gift_repository as gift_repo
from app.services.offline import repository as repo
from app.services.offline.chat_emit import (
    emit_assistant,
    emit_gift_card,
    insert_user_component_message,
    offline_trace,
)
from app.services.offline.gift_amount import sample_gift_amount_cents
from app.services.offline.gift_budget import available_gift_budget_cents
from app.services.offline.gift_messages import (
    first_address_request_message,
    gift_out_for_delivery_message,
    gift_received_message,
    gift_sent_message,
    gift_thanks_reply,
)
from app.services.offline.gift_status import (
    ACTIVE_STATUSES,
    IN_TRANSIT_STATUSES,
    REVEALED_STATUS,
    SPENT_STATUSES,
    user_view,
)
from app.services.offline.gift_selection import parse_json_object
from app.services.offline.memory_hooks import remember_user_event
from app.services.offline.providers.gift_types import GiftProviderError
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


async def get_gifts(user_id: str, workspace_id: str | None = None) -> GiftsHomeResponse:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    resolved_workspace = ctx["workspace_id"] if ctx else workspace_id
    if ctx:
        await _refresh_deliveries(user_id, resolved_workspace, ctx)
    address = await gift_repo.default_address(user_id, masked=True)
    gifts = await gift_repo.list_gifts(user_id, resolved_workspace)
    # list_gifts 按创建时间倒序，第一份就是最新寄出的。
    in_transit = [RealWorldGiftItem(**user_view(g)) for g in gifts if g["status"] in IN_TRANSIT_STATUSES]
    grouped: dict[int, list[RealWorldGiftItem]] = {}
    for gift in gifts:
        if gift["status"] != REVEALED_STATUS:
            continue
        year = _year_for_gift(gift)
        grouped.setdefault(year, []).append(RealWorldGiftItem(**user_view(gift)))
    return GiftsHomeResponse(
        address=GiftAddressResponse(**address) if address else None,
        shipping_gift=in_transit[0] if in_transit else None,
        shipping_gifts=in_transit,
        groups=[
            GiftYearGroup(year=year, gifts=items)
            for year, items in sorted(grouped.items(), reverse=True)
        ],
    )


async def get_address(user_id: str) -> GiftAddressResponse:
    # Authenticated owner-only editing endpoint; home listings stay masked.
    address = await gift_repo.default_address(user_id, masked=False)
    return GiftAddressResponse(**address) if address else GiftAddressResponse()


async def delete_address(user_id: str) -> None:
    await gift_repo.delete_default_address(user_id)


async def clear_all_gifts(user_id: str) -> dict[str, int]:
    result = await gift_repo.clear_user_gifts(user_id)
    logger.info("[offline] admin gift clear user_id=%s counts=%s", user_id, result)
    return result


async def save_address(user_id: str, data: GiftAddressRequest) -> GiftAddressResponse:
    address = await gift_repo.upsert_address(user_id, data.model_dump())
    remember_user_event(
        user_id=user_id,
        workspace_id=None,
        text=f"用户更新了礼物收货城市：{data.city}{data.district}",
    )
    try:
        await _resume_pending_address_gift(user_id)
    except Exception as exc:
        logger.warning("[offline] resume pending gift after address failed: %s", exc)
    return GiftAddressResponse(**address)


async def get_gift(user_id: str, gift_id: str) -> RealWorldGiftItem:
    gift = await gift_repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    ctx = await repo.resolve_user_context(user_id, gift.get("workspace_id"))
    if ctx:
        gift = await _refresh_one_delivery(user_id, gift, ctx) or gift
    return RealWorldGiftItem(**user_view(gift))


async def get_tracking(user_id: str, gift_id: str) -> GiftTrackingResponse:
    gift = await gift_repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    ctx = await repo.resolve_user_context(user_id, gift.get("workspace_id"))
    if ctx:
        await _refresh_one_delivery(user_id, gift, ctx)
    events = await gift_repo.gift_tracking(gift_id, user_id)
    return GiftTrackingResponse(
        gift_id=gift_id,
        events=[GiftTrackingEvent(**event) for event in events],
    )


async def refresh_due_gift_deliveries(
    user_id: str,
    workspace_id: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> int:
    resolved = ctx or await repo.resolve_user_context(user_id, workspace_id)
    if not resolved:
        return 0
    refreshed = 0
    for gift in await gift_repo.list_gifts(user_id, resolved.get("workspace_id")):
        before = gift.get("status")
        updated = await _refresh_one_delivery(user_id, gift, resolved)
        if before in {"ordered", "shipping"} and updated and updated.get("status") == "out_for_delivery":
            refreshed += 1
    return refreshed


async def send_thanks(
    user_id: str,
    gift_id: str,
    message: str,
    *,
    client_id: str | None = None,
) -> GiftThanksResponse:
    gift = await gift_repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    if gift.get("status") != REVEALED_STATUS:
        # 感谢消息会带上礼物名，签收前发出去等于提前揭晓。
        raise HTTPException(status_code=409, detail="签收礼物后才能道谢哦")
    if gift.get("thanks_sent_at"):
        return GiftThanksResponse(ok=True, gift=RealWorldGiftItem(**user_view(gift)), assistant_message=None)
    ctx = await repo.resolve_user_context(user_id, gift.get("workspace_id"))
    if ctx:
        await insert_user_component_message(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx.get("workspace_id"),
            content=f"我收到了「{gift['gift_name']}」，想说：{message}",
            client_id=client_id,
            metadata={
                "user_id": user_id,
                "real_world_type": "gift",
                "source_id": gift_id,
                "trigger_type": "gift_thanks",
            },
        )
    updated = await gift_repo.mark_gift_thanked(gift_id, user_id, message)
    if not updated:
        raise HTTPException(status_code=404, detail="Gift not found")
    async with offline_trace(
        "gift_thanks",
        conversation_id=(ctx or {}).get("conversation_id"),
        agent_id=(ctx or {}).get("agent_id"), user_id=user_id,
    ) as tracer:
        assistant = await gift_thanks_reply(gift, message)
        if ctx and assistant:
            await emit_assistant(
                conversation_id=ctx.get("conversation_id"),
                user_id=user_id,
                agent_id=ctx["agent_id"],
                workspace_id=ctx["workspace_id"],
                message=assistant,
                real_world_type="gift",
                source_id=gift_id,
                trigger_type="gift_thanks_reply",
                trace_id=tracer.safe_trace_id,
            )
    remember_user_event(
        user_id=user_id,
        workspace_id=gift.get("workspace_id"),
        text=f"用户收到礼物「{gift['gift_name']}」后表达感谢：{message}",
    )
    return GiftThanksResponse(
        ok=True,
        gift=RealWorldGiftItem(**user_view(updated)),
        assistant_message=assistant,
    )


async def create_gift_for_user(
    *,
    user_id: str,
    workspace_id: str | None = None,
    trigger_type: str = "daily_probability",
) -> dict[str, Any] | None:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    if not ctx or not ctx.get("conversation_id"):
        return None
    existing = await gift_repo.list_gifts(user_id, ctx["workspace_id"])
    active = next((g for g in existing if g["status"] in ACTIVE_STATUSES), None)
    if active:
        if active["status"] == "pending_address":
            address = await gift_repo.default_address(user_id, masked=False)
            if address:
                return await _fulfill_gift(
                    user_id=user_id,
                    ctx=ctx,
                    address=address,
                    trigger_type=active["trigger_type"] or trigger_type,
                    amount_cents=active.get("target_amount_cents") or None,
                    pending_gift=active,
                )
        return active
    budget = await available_gift_budget_cents(user_id)
    amount_cents = sample_gift_amount_cents(budget)
    if amount_cents is None:
        return None

    address = await gift_repo.default_address(user_id, masked=False)
    if not address:
        gift = await gift_repo.create_gift(
            {
                "user_id": user_id,
                "agent_id": ctx["agent_id"],
                "workspace_id": ctx["workspace_id"],
                "conversation_id": ctx["conversation_id"],
                "status": "pending_address",
                "trigger_type": trigger_type,
                "gift_name": "还没写上名字的小惊喜",
                "gift_reason": "想给你准备一点现实里的小心意，但还缺收货地址。",
                "target_amount_cents": amount_cents,
            }
        )
        async with offline_trace(
            "gift_address_request",
            conversation_id=ctx["conversation_id"],
            agent_id=ctx["agent_id"], user_id=user_id,
        ) as tracer:
            message = await first_address_request_message(user_id, ctx["workspace_id"])
            await emit_assistant(
                conversation_id=ctx["conversation_id"],
                user_id=user_id,
                agent_id=ctx["agent_id"],
                workspace_id=ctx["workspace_id"],
                message=message,
                real_world_type="gift",
                source_id=gift["id"],
                trigger_type="gift_address_needed",
                trace_id=tracer.safe_trace_id,
            )
        return gift

    return await _fulfill_gift(
        user_id=user_id,
        ctx=ctx,
        address=address,
        trigger_type=trigger_type,
        amount_cents=amount_cents,
    )


_MOCK_GIFT_SPECS: list[dict[str, Any]] = [
    {
        "gift_name": "暖手宝",
        "gift_reason": "天冷了，想让你随手揣个暖手的小物件。",
        "gift_note": "焐着手再敲键盘，别老把指尖冻得冰凉。",
        "amount_cents": 3900,
    },
    {
        "gift_name": "桂花乌龙茶",
        "gift_reason": "你提过喜欢带花香的茶。",
        "gift_note": "忙里偷闲泡一杯，给自己几分钟发个呆。",
        "amount_cents": 5800,
    },
    {
        "gift_name": "帆布手账本",
        "gift_reason": "你常说想把零碎灵感记下来。",
        "gift_note": "随手写写画画，日子也会更有痕迹。",
        "amount_cents": 4600,
    },
]

_MOCK_ADDRESS: dict[str, Any] = {
    "recipient_name": "测试收件人",
    "phone": "13800000000",
    "province": "上海市",
    "city": "上海市",
    "district": "黄浦区",
    "detail": "南京东路 100 号测试大厦 1001 室",
    "is_default": True,
}


MockGiftStage = Literal["shipping", "out_for_delivery", "delivered"]


async def create_mock_gift_for_user(
    *,
    user_id: str,
    workspace_id: str | None = None,
    stage: MockGiftStage = "shipping",
) -> dict[str, Any] | None:
    """管理员测试：为当前用户注入一份礼物，用于验证前端展示。

    仅在 mock provider 下可用：跳过触发判定、预算门控与 LLM 选礼，直接造一份 spec 走
    purchase_gift（mock 下单）+ sync_tracking_events（mock 物流轨迹）。
    stage 决定停在哪一步：shipping 运输中；out_for_delivery 派送中（可测签收）；
    delivered 再走一遍真实的签收流程（揭晓卡片 + AI 消息），用于测感谢交互。

    ⚠️ 若当前 GIFT_COMMERCE_PROVIDER 是真实 provider（如 ali1688），此接口会触发
    真实下单+真实扣款，故显式拒绝——测试注入只允许在 mock 下进行。
    """
    if gift_fulfillment.commerce_provider_name() != "mock":
        raise HTTPException(
            status_code=400,
            detail="测试注入仅在 mock provider 下可用；当前 provider 会产生真实下单/扣款。",
        )

    ctx = await repo.resolve_user_context(user_id, workspace_id)
    if not ctx or not ctx.get("conversation_id"):
        raise HTTPException(
            status_code=400,
            detail="需要先创建一个 AI 伙伴并有聊天会话，才能注入测试礼物。",
        )

    address = await gift_repo.default_address(user_id, masked=False)
    if not address:
        await gift_repo.upsert_address(user_id, dict(_MOCK_ADDRESS))
        address = await gift_repo.default_address(user_id, masked=False)
    if not address:
        raise HTTPException(status_code=500, detail="测试收货地址创建失败。")

    existing_count = len(await gift_repo.list_gifts(user_id, ctx["workspace_id"]))
    spec = dict(_MOCK_GIFT_SPECS[existing_count % len(_MOCK_GIFT_SPECS)])

    gift = await _fulfill_gift(
        user_id=user_id,
        ctx=ctx,
        address=address,
        trigger_type="admin_mock",
        spec_override=spec,
    )
    if not gift:
        raise HTTPException(status_code=500, detail="测试礼物下单失败，请检查 gift provider 配置。")

    if stage in {"out_for_delivery", "delivered"}:
        gift = await advance_to_out_for_delivery(
            user_id, gift, ctx=ctx, event=_courier_at_door_event(datetime.now(UTC)),
        ) or gift
    if stage == "delivered":
        gift = (await _confirm_receipt(user_id, gift["id"], actor="admin"))[0]
    return gift


async def _resume_pending_address_gift(user_id: str) -> dict[str, Any] | None:
    ctx = await repo.resolve_user_context(user_id)
    if not ctx:
        return None
    pending = next(
        (
            gift for gift in await gift_repo.list_gifts(user_id, ctx["workspace_id"])
            if gift["status"] == "pending_address"
        ),
        None,
    )
    if not pending:
        return None
    address = await gift_repo.default_address(user_id, masked=False)
    if not address:
        return None
    return await _fulfill_gift(
        user_id=user_id,
        ctx=ctx,
        address=address,
        trigger_type=pending["trigger_type"],
        amount_cents=pending.get("target_amount_cents") or None,
        pending_gift=pending,
    )


async def _fulfill_gift(
    *,
    user_id: str,
    ctx: dict[str, Any],
    address: dict[str, Any],
    trigger_type: str,
    amount_cents: int | None = None,
    pending_gift: dict[str, Any] | None = None,
    spec_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    # spec_override 仅用于管理员测试注入：跳过预算门控与 LLM 选礼，直接用给定 spec
    # 走真实的 provider 下单 + 物流同步链路（默认 mock provider）。
    if spec_override is not None:
        spec = spec_override
    else:
        budget = await available_gift_budget_cents(user_id)
        amount = (
            amount_cents
            if amount_cents and amount_cents <= budget
            else sample_gift_amount_cents(budget)
        )
        if amount is None:
            if pending_gift:
                await gift_repo.update_gift_status(
                    pending_gift["id"],
                    user_id,
                    "skipped",
                    failure_reason="gift budget below minimum",
                )
            return None

        spec = await _select_gift(user_id, ctx["workspace_id"], amount)
    try:
        commerce_provider_name = gift_fulfillment.commerce_provider_name()
        logistics_provider_name = gift_fulfillment.logistics_provider_name()
    except GiftProviderError as exc:
        logger.warning("[offline] gift provider is not configured: %s", exc)
        if pending_gift:
            await gift_repo.update_gift_status(
                pending_gift["id"],
                user_id,
                "failed",
                failure_reason=str(exc)[:300],
            )
        return None
    now = datetime.now(UTC)
    staging_payload = _gift_staging_payload(
        user_id=user_id,
        ctx=ctx,
        address=address,
        trigger_type=trigger_type,
        spec=spec,
        provider_name=commerce_provider_name,
        status="selecting",
        ordered_at=now,
    )
    gift = (
        await gift_repo.update_gift_order_details(pending_gift["id"], user_id, staging_payload)
        if pending_gift
        else await gift_repo.create_gift(staging_payload)
    )
    if not gift:
        return None

    try:
        candidate, order = await gift_fulfillment.purchase_gift(
            gift_id=gift["id"],
            spec=spec,
            address=address,
        )
    except GiftProviderError as exc:
        logger.warning("[offline] gift purchase failed gift_id=%s: %s", gift["id"], exc)
        await gift_repo.update_gift_status(
            gift["id"],
            user_id,
            "failed",
            failure_reason=str(exc)[:300],
        )
        return None

    now = datetime.now(UTC)
    payload = {
        "user_id": user_id,
        "agent_id": ctx["agent_id"],
        "workspace_id": ctx["workspace_id"],
        "conversation_id": ctx["conversation_id"],
        "status": _initial_order_status(order.status),
        "trigger_type": trigger_type,
        "gift_name": spec["gift_name"],
        "gift_reason": spec["gift_reason"],
        "gift_note": spec["gift_note"],
        "product_image_url": order.product_image_url or candidate.image_url,
        "provider_product_id": candidate.external_product_id,
        "product_url": candidate.product_url,
        "product_snapshot": gift_fulfillment.candidate_snapshot(candidate),
        "target_amount_cents": spec["amount_cents"],
        "paid_amount_cents": order.paid_amount_cents,
        "provider": order.provider,
        "provider_order_id": order.provider_order_id,
        "tracking_number": order.tracking_number,
        "logistics_provider": logistics_provider_name,
        "provider_payload": order.raw,
        "logistics_payload": {},
        "address_snapshot": address,
        "ordered_at": now,
        "shipped_at": order.shipped_at,
        "delivered_at": order.delivered_at,
        "last_tracking_synced_at": None,
    }
    gift = await gift_repo.update_gift_order_details(gift["id"], user_id, payload)
    if not gift:
        logger.warning("[offline] gift cancelled while ordering user_id=%s", user_id)
        return None
    await gift_fulfillment.sync_tracking_events(user_id, gift)
    await gift_repo.update_last_gift_paid(user_id, ctx["agent_id"], ctx["workspace_id"])
    async with offline_trace(
        "gift_sent",
        conversation_id=ctx["conversation_id"],
        agent_id=ctx["agent_id"], user_id=user_id,
    ) as tracer:
        message = await gift_sent_message(user_id, ctx["workspace_id"])
        await emit_gift_card(
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
    remember_mystery_gift_sent(user_id, ctx["workspace_id"])
    return gift


def _initial_order_status(provider_status: str) -> str:
    # provider 下单即报「已送达」也只算派送中之前的在途：签收必须由用户确认。
    if provider_status in {"ordered", "shipping"}:
        return provider_status
    return "shipping" if provider_status in {"out_for_delivery", "delivered"} else "ordered"


def _gift_staging_payload(
    *,
    user_id: str,
    ctx: dict[str, Any],
    address: dict[str, Any],
    trigger_type: str,
    spec: dict[str, Any],
    provider_name: str,
    status: str,
    ordered_at: datetime,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "agent_id": ctx["agent_id"],
        "workspace_id": ctx["workspace_id"],
        "conversation_id": ctx["conversation_id"],
        "status": status,
        "trigger_type": trigger_type,
        "gift_name": spec["gift_name"],
        "gift_reason": spec["gift_reason"],
        "gift_note": spec["gift_note"],
        "target_amount_cents": spec["amount_cents"],
        "paid_amount_cents": 0,
        "provider": provider_name,
        "address_snapshot": address,
        "ordered_at": ordered_at,
    }


async def _select_gift(user_id: str, workspace_id: str | None, amount_cents: int) -> dict[str, Any]:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=8)
    memory = await repo.memory_brief(user_id, workspace_id, limit=80)
    sent_gifts = [
        gift["gift_name"] for gift in await gift_repo.list_gifts(user_id, workspace_id)
        if gift.get("gift_name") and gift["status"] in SPENT_STATUSES
    ][:20]
    try:
        prompt_text = (await get_prompt_text("offline.gift_selection")).format(
            amount_yuan=f"{amount_cents / 100:.2f}",
            min_yuan=f"{amount_cents * 0.8 / 100:.2f}",
            max_yuan=f"{amount_cents * 1.2 / 100:.2f}",
            tags=", ".join(tags) if tags else "暂无",
            memory=memory or "暂无",
            sent_gifts=", ".join(sent_gifts) if sent_gifts else "暂无",
        )
        raw = await invoke_text(get_chat_model(), prompt_text)
        parsed = parse_json_object(raw) or {}
    except Exception as exc:
        logger.warning("[offline] gift selection failed: %s", exc)
        parsed = {}
    amount = int(parsed.get("amount_cents") or amount_cents)
    amount = max(500, min(amount, round(amount_cents * 1.2)))
    return {
        "gift_name": str(parsed.get("gift_name") or "手冲咖啡壶套装")[:40],
        "gift_reason": str(parsed.get("gift_reason") or "记得你喜欢把生活过得慢一点。")[:120],
        "gift_note": str(parsed.get("gift_note") or "希望这一点小心意，能替我把今天照亮一点。")[:180],
        "amount_cents": amount,
    }


async def _refresh_deliveries(
    user_id: str,
    workspace_id: str | None,
    ctx: dict[str, Any],
) -> None:
    for gift in await gift_repo.list_gifts(user_id, workspace_id):
        await _refresh_one_delivery(user_id, gift, ctx)


async def _refresh_one_delivery(
    user_id: str,
    gift: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any] | None:
    # 派送中也继续同步：快递员 / 驿站的后续轨迹要能看到；状态迁移由 CAS 保证不重复。
    if gift.get("status") not in IN_TRANSIT_STATUSES:
        return gift
    try:
        snapshot = await gift_fulfillment.sync_tracking_events(user_id, gift)
    except GiftProviderError as exc:
        logger.warning("[offline] gift tracking refresh failed gift_id=%s: %s", gift["id"], exc)
        return gift
    if snapshot is None:
        # 手动礼物：轨迹由后台录入，派送中由管理员推进。
        return gift
    if gift_fulfillment.provider_gift_status(snapshot.current_status) == "out_for_delivery":
        return await advance_to_out_for_delivery(user_id, gift, ctx=ctx) or gift
    # provider 只给了预计送达时间（mock 即如此）：到点视为快递已在派送。
    eta = _parse_dt(gift.get("delivered_at"))
    if eta and eta <= datetime.now(UTC):
        return await advance_to_out_for_delivery(
            user_id, gift, ctx=ctx, event=_courier_at_door_event(eta),
        ) or gift
    return gift


def _courier_at_door_event(occurred_at: datetime) -> dict[str, Any]:
    return {
        "status": "out_for_delivery",
        "title": "快递员正在派送",
        "description": "收到后记得在礼物页确认签收，签收后就能揭晓礼物啦。",
        "location": "收货地址附近",
        "occurred_at": occurred_at,
    }


async def advance_to_out_for_delivery(
    user_id: str,
    gift: dict[str, Any],
    *,
    ctx: dict[str, Any] | None = None,
    event: dict[str, Any] | None = None,
    notify: bool = True,
) -> dict[str, Any] | None:
    """在途 → 派送中（待签收），并提醒用户收到后签收。

    状态迁移是 CAS：并发刷新 / 重复请求只有一次真正迁移，提醒消息也只发一次。
    没有发生迁移时返回 None。同一份礼物只提醒一次（以聊天里是否已有提醒消息为准），
    notify=False 时由调用方自行决定是否提醒。
    """
    updated = await gift_repo.mark_gift_out_for_delivery(gift["id"], user_id)
    if not updated:
        return None
    if event:
        await gift_repo.add_tracking_events(gift["id"], [event])
    logger.info(
        "[offline] gift out for delivery gift_id=%s",
        gift["id"],
        extra={"event": EVT_GIFT_OUT_FOR_DELIVERY, "gift_id": gift["id"], "user_id": user_id},
    )
    if notify and not await gift_repo.gift_notice_sent(gift["id"], user_id, "gift_out_for_delivery"):
        await notify_out_for_delivery(user_id, updated, ctx=ctx)
    return updated


async def notify_out_for_delivery(
    user_id: str,
    gift: dict[str, Any],
    *,
    ctx: dict[str, Any] | None = None,
) -> str | None:
    """AI 提醒用户快递在派送、收到后去签收。失败只记日志：状态已经迁移成功。"""
    try:
        ctx = ctx or await repo.resolve_user_context(user_id, gift.get("workspace_id"))
        if not ctx:
            return None
        async with offline_trace(
            "gift_out_for_delivery",
            conversation_id=ctx.get("conversation_id"),
            agent_id=ctx["agent_id"], user_id=user_id,
        ) as tracer:
            message = await gift_out_for_delivery_message(user_id, ctx.get("workspace_id"))
            return await emit_gift_card(
                conversation_id=ctx.get("conversation_id"),
                user_id=user_id,
                agent_id=ctx["agent_id"],
                workspace_id=ctx.get("workspace_id"),
                gift=gift,
                trigger_type="gift_out_for_delivery",
                status_label="派送中",
                message=message,
                trace_id=tracer.safe_trace_id,
            )
    except Exception as exc:
        logger.warning("[offline] gift out-for-delivery notify failed gift_id=%s: %s", gift.get("id"), exc)
        return None


async def confirm_receipt(user_id: str, gift_id: str) -> GiftReceiptResponse:
    """用户确认签收：派送中 → 已收到，并揭晓礼物。"""
    gift, assistant, already = await _confirm_receipt(user_id, gift_id, actor="user")
    return GiftReceiptResponse(
        ok=True,
        gift=RealWorldGiftItem(**user_view(gift)),
        assistant_message=assistant,
        already_confirmed=already,
    )


async def _confirm_receipt(
    user_id: str,
    gift_id: str,
    *,
    actor: str,
) -> tuple[dict[str, Any], str | None, bool]:
    gift = await gift_repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    if gift["status"] == REVEALED_STATUS:
        return gift, None, True
    updated = await gift_repo.mark_gift_received(gift_id, user_id)
    if not updated:
        latest = await gift_repo.get_gift(gift_id, user_id)
        if latest and latest["status"] == REVEALED_STATUS:
            return latest, None, True
        raise HTTPException(status_code=409, detail="礼物还没到派送环节，收到快递后再来签收吧")
    await gift_repo.add_tracking_events(
        gift_id,
        [{
            "status": REVEALED_STATUS,
            "title": "已签收",
            "description": "你已确认收到这份礼物" if actor == "user" else "已由客服代你确认签收",
            "location": None,
            "occurred_at": datetime.now(UTC),
        }],
    )
    logger.info(
        "[offline] gift received gift_id=%s actor=%s",
        gift_id,
        actor,
        extra={"event": EVT_GIFT_RECEIVED, "gift_id": gift_id, "user_id": user_id, "actor": actor},
    )
    assistant = await _announce_received(user_id, updated)
    remember_user_event(
        user_id=user_id,
        workspace_id=updated.get("workspace_id"),
        text=f"用户签收了 AI 寄出的礼物「{updated['gift_name']}」，礼物已经揭晓，可以自然地聊它",
    )
    return updated, assistant, False


async def _announce_received(user_id: str, gift: dict[str, Any]) -> str | None:
    """签收后的揭晓卡片 + AI 消息。失败只记日志：签收本身已经成功。"""
    try:
        ctx = await repo.resolve_user_context(user_id, gift.get("workspace_id"))
        if not ctx:
            return None
        async with offline_trace(
            "gift_received",
            conversation_id=ctx.get("conversation_id"),
            agent_id=ctx["agent_id"], user_id=user_id,
        ) as tracer:
            message = await gift_received_message(user_id, ctx.get("workspace_id"), gift)
            await emit_gift_card(
                conversation_id=ctx.get("conversation_id"),
                user_id=user_id,
                agent_id=ctx["agent_id"],
                workspace_id=ctx.get("workspace_id"),
                gift=gift,
                trigger_type="gift_received",
                status_label="已签收",
                message=message,
                trace_id=tracer.safe_trace_id,
            )
            return message
    except Exception as exc:
        logger.warning("[offline] gift received announce failed gift_id=%s: %s", gift.get("id"), exc)
        return None


def remember_mystery_gift_sent(user_id: str, workspace_id: str | None) -> None:
    # 礼物名不进记忆：记忆会注入聊天，AI 可能在用户签收前说漏嘴。
    remember_user_event(
        user_id=user_id,
        workspace_id=workspace_id,
        text="AI 给用户寄出了一份神秘礼物，用户签收前要保密，不要提前透露礼物是什么",
    )


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            return None
    return None


def _year_for_gift(gift: dict[str, Any]) -> int:
    raw = gift.get("delivered_at") or gift.get("ordered_at") or gift.get("created_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).year
        except Exception:
            return datetime.now(UTC).year
    return datetime.now(UTC).year
