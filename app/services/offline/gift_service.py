from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from app.models.offline import (
    GiftAddressRequest,
    GiftAddressResponse,
    GiftThanksResponse,
    GiftTrackingEvent,
    GiftTrackingResponse,
    GiftYearGroup,
    GiftsHomeResponse,
    RealWorldGiftItem,
)
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import repository as repo
from app.services.offline.chat_emit import emit_assistant, insert_user_component_message
from app.services.offline.gift_budget import available_gift_budget_cents
from app.services.offline.memory_hooks import remember_user_event
from app.services.offline.providers.gift import mock_order_gift, mock_tracking_events
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


async def get_gifts(user_id: str, workspace_id: str | None = None) -> GiftsHomeResponse:
    ctx = await repo.resolve_user_context(user_id, workspace_id)
    resolved_workspace = ctx["workspace_id"] if ctx else workspace_id
    address = await repo.default_address(user_id, masked=True)
    gifts = await repo.list_gifts(user_id, resolved_workspace)
    shipping = next((g for g in gifts if g["status"] in {"ordered", "shipping"}), None)
    grouped: dict[int, list[RealWorldGiftItem]] = {}
    for gift in gifts:
        if shipping and gift["id"] == shipping["id"]:
            continue
        year = _year_for_gift(gift)
        grouped.setdefault(year, []).append(RealWorldGiftItem(**gift))
    return GiftsHomeResponse(
        address=GiftAddressResponse(**address) if address else None,
        shipping_gift=RealWorldGiftItem(**shipping) if shipping else None,
        groups=[
            GiftYearGroup(year=year, gifts=items)
            for year, items in sorted(grouped.items(), reverse=True)
        ],
    )


async def get_address(user_id: str) -> GiftAddressResponse:
    address = await repo.default_address(user_id, masked=True)
    return GiftAddressResponse(**address) if address else GiftAddressResponse()


async def save_address(user_id: str, data: GiftAddressRequest) -> GiftAddressResponse:
    address = await repo.upsert_address(user_id, data.model_dump())
    remember_user_event(
        user_id=user_id,
        workspace_id=None,
        text=f"用户更新了礼物收货城市：{data.city}{data.district}",
    )
    return GiftAddressResponse(**address)


async def get_gift(user_id: str, gift_id: str) -> RealWorldGiftItem:
    gift = await repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    return RealWorldGiftItem(**gift)


async def get_tracking(user_id: str, gift_id: str) -> GiftTrackingResponse:
    gift = await repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    events = await repo.gift_tracking(gift_id, user_id)
    return GiftTrackingResponse(
        gift_id=gift_id,
        events=[GiftTrackingEvent(**event) for event in events],
    )


async def send_thanks(user_id: str, gift_id: str, message: str) -> GiftThanksResponse:
    gift = await repo.get_gift(gift_id, user_id)
    if not gift:
        raise HTTPException(status_code=404, detail="Gift not found")
    if gift.get("thanks_sent_at"):
        return GiftThanksResponse(ok=True, gift=RealWorldGiftItem(**gift), assistant_message=None)
    ctx = await repo.resolve_user_context(user_id, gift.get("workspace_id"))
    if ctx:
        await insert_user_component_message(
            conversation_id=ctx.get("conversation_id"),
            workspace_id=ctx.get("workspace_id"),
            content=f"我收到了「{gift['gift_name']}」，想说：{message}",
            metadata={
                "user_id": user_id,
                "real_world_type": "gift",
                "source_id": gift_id,
                "trigger_type": "gift_thanks",
            },
        )
    updated = await repo.mark_gift_thanked(gift_id, user_id, message)
    if not updated:
        raise HTTPException(status_code=404, detail="Gift not found")
    assistant = await _thank_you_reply(gift, message)
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
        )
    remember_user_event(
        user_id=user_id,
        workspace_id=gift.get("workspace_id"),
        text=f"用户收到礼物「{gift['gift_name']}」后表达感谢：{message}",
    )
    return GiftThanksResponse(
        ok=True,
        gift=RealWorldGiftItem(**updated),
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
    existing = await repo.list_gifts(user_id, ctx["workspace_id"])
    active = next(
        (g for g in existing if g["status"] in {"pending_address", "selecting", "ordered", "shipping"}),
        None,
    )
    if active:
        return active
    budget = await available_gift_budget_cents(user_id)
    if budget < 500:
        return None

    address = await repo.default_address(user_id, masked=False)
    if not address:
        gift = await repo.create_gift(
            {
                "user_id": user_id,
                "agent_id": ctx["agent_id"],
                "workspace_id": ctx["workspace_id"],
                "conversation_id": ctx["conversation_id"],
                "status": "pending_address",
                "trigger_type": trigger_type,
                "gift_name": "还没写上名字的小惊喜",
                "gift_reason": "想给你准备一点现实里的小心意，但还缺收货地址。",
                "target_amount_cents": min(budget, 3900),
            }
        )
        await emit_assistant(
            conversation_id=ctx["conversation_id"],
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message="我有一点现实里的小心意想寄给你。先去「我的礼物」里补一下收货地址吧，我会把它稳稳放好。",
            real_world_type="gift",
            source_id=gift["id"],
            trigger_type="gift_address_needed",
        )
        return gift

    spec = await _select_gift(user_id, ctx["workspace_id"], budget)
    order = await mock_order_gift(spec["gift_name"])
    now = datetime.now(UTC)
    gift = await repo.create_gift(
        {
            "user_id": user_id,
            "agent_id": ctx["agent_id"],
            "workspace_id": ctx["workspace_id"],
            "conversation_id": ctx["conversation_id"],
            "status": "shipping",
            "trigger_type": trigger_type,
            "gift_name": spec["gift_name"],
            "gift_reason": spec["gift_reason"],
            "gift_note": spec["gift_note"],
            "product_image_url": order.product_image_url,
            "target_amount_cents": spec["amount_cents"],
            "paid_amount_cents": spec["amount_cents"],
            "provider": "mock",
            "provider_order_id": order.provider_order_id,
            "tracking_number": order.tracking_number,
            "address_snapshot": address,
            "ordered_at": now,
            "shipped_at": now + timedelta(hours=8),
        }
    )
    await repo.add_tracking_events(gift["id"], await mock_tracking_events())
    await repo.update_last_gift_paid(user_id, ctx["agent_id"], ctx["workspace_id"])
    await emit_assistant(
        conversation_id=ctx["conversation_id"],
        user_id=user_id,
        agent_id=ctx["agent_id"],
        workspace_id=ctx["workspace_id"],
        message=f"我给你寄出了一份「{gift['gift_name']}」。不用立刻做什么，它现在已经在路上了。",
        real_world_type="gift",
        source_id=gift["id"],
        trigger_type="gift_sent",
    )
    return gift


async def _select_gift(user_id: str, workspace_id: str | None, budget: int) -> dict[str, Any]:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=8)
    memory = await repo.memory_brief(user_id, workspace_id, limit=40)
    try:
        prompt_text = (await get_prompt_text("offline.gift_selection")).format(
            budget_yuan=f"{budget / 100:.0f}",
            tags=", ".join(tags) if tags else "暂无",
            memory=memory or "暂无",
        )
        raw = await invoke_text(get_chat_model(), prompt_text)
        parsed = _json_object(raw) or {}
    except Exception as exc:
        logger.warning("[offline] gift selection failed: %s", exc)
        parsed = {}
    amount = int(parsed.get("amount_cents") or min(max(1800, budget // 2), 3900))
    amount = max(500, min(amount, budget, 12800))
    return {
        "gift_name": str(parsed.get("gift_name") or "手冲咖啡壶套装")[:40],
        "gift_reason": str(parsed.get("gift_reason") or "记得你喜欢把生活过得慢一点。")[:120],
        "gift_note": str(parsed.get("gift_note") or "希望这一点小心意，能替我把今天照亮一点。")[:180],
        "amount_cents": amount,
    }


async def _thank_you_reply(gift: dict[str, Any], message: str) -> str:
    try:
        prompt_text = (await get_prompt_text("offline.gift_thanks_reply")).format(
            gift_name=gift.get("gift_name") or "礼物",
            message=message,
        )
        reply = (await invoke_text(get_chat_model(), prompt_text)).strip()
        return reply.split("\n", 1)[0][:80]
    except Exception:
        return "收到你的谢谢，我会偷偷开心很久。"


def _year_for_gift(gift: dict[str, Any]) -> int:
    raw = gift.get("delivered_at") or gift.get("ordered_at") or gift.get("created_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).year
        except Exception:
            return datetime.now(UTC).year
    return datetime.now(UTC).year
