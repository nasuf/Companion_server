"""神秘礼物 + 用户签收：在途脱敏、派送中迁移与提醒、签收揭晓的幂等与失败路径。"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.services.offline import chat_emit, gift_fulfillment, gift_service
from app.services.offline.gift_status import (
    MYSTERY_GIFT_NAME,
    advances_transit,
    gift_status_for_event,
    normalize_provider_event_status,
    user_view,
)
from app.services.offline.providers.gift_types import GiftTrackingSnapshot, GiftTrackingUpdate

NOW = "2026-10-15T08:00:00+00:00"


def _gift(status="shipping", **extra):
    gift = {
        "id": "gift-1",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "workspace_id": "ws-1",
        "conversation_id": "conv-1",
        "status": status,
        "trigger_type": "admin_manual",
        "gift_name": "桂花乌龙茶",
        "gift_reason": "用户提过喜欢花香的茶（M3）",
        "gift_note": "泡一杯，给自己几分钟发呆。",
        "product_image_url": "/offline/media/user-1_image_abc.jpg",
        "product_url": "https://example.com/item",
        "provider_product_id": "P-1",
        "provider": "manual",
        "paid_amount_cents": 5800,
        "tracking_number": "SF123",
        "logistics_provider": "顺丰",
        "delivered_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    gift.update(extra)
    return gift


CTX = {"workspace_id": "ws-1", "agent_id": "agent-1", "conversation_id": "conv-1"}


# ── 可见性规则 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["ordered", "shipping", "out_for_delivery", "cancelled", "pending_address"])
def test_user_view_hides_everything_that_identifies_the_gift_before_receipt(status):
    view = user_view(_gift(status, delivered_at="2026-10-18T00:00:00+00:00"))
    assert view["revealed"] is False
    assert view["gift_name"] == MYSTERY_GIFT_NAME
    for field in ("gift_reason", "gift_note", "product_image_url", "product_url", "provider_product_id", "delivered_at"):
        assert view[field] is None
    assert view["paid_amount_cents"] == 0
    # 物流信息照常可见：用户要能查快递。
    assert view["tracking_number"] == "SF123"
    assert view["logistics_provider"] == "顺丰"
    assert view["can_confirm_receipt"] is (status == "out_for_delivery")


def test_user_view_reveals_after_receipt_but_keeps_admin_reason_internal():
    manual = user_view(_gift("delivered"))
    assert manual["revealed"] is True
    assert manual["gift_name"] == "桂花乌龙茶"
    assert manual["product_image_url"].endswith(".jpg")
    assert manual["gift_note"] == "泡一杯，给自己几分钟发呆。"
    assert manual["gift_reason"] is None  # 运营分析不给用户看
    automatic = user_view(_gift("delivered", provider="mock", gift_reason="记得你喜欢慢生活"))
    assert automatic["gift_reason"] == "记得你喜欢慢生活"


def test_status_helpers_never_let_logistics_sign_on_behalf_of_user():
    assert normalize_provider_event_status("delivered") == "out_for_delivery"
    assert normalize_provider_event_status("packed") == "shipping"  # 用户只看到 下单/运输/派送/签收
    assert gift_status_for_event("packed") == "shipping"
    assert gift_status_for_event("delivered") is None
    assert advances_transit("ordered", "shipping") is True
    assert advances_transit("out_for_delivery", "shipping") is False
    assert advances_transit("delivered", "out_for_delivery") is False


def test_chat_card_persists_no_gift_identity_until_revealed():
    card = chat_emit.build_gift_component_card(_gift("shipping"), status_label="在路上", message="有惊喜在路上")
    assert card["title"] == MYSTERY_GIFT_NAME
    assert card["payload"]["image_url"] is None
    assert card["payload"]["gift_name"] is None
    assert card["payload"]["revealed"] is False
    assert "桂花" not in str(card)

    revealed = chat_emit.build_gift_component_card(_gift("delivered"), status_label="已签收", message="是桂花乌龙茶！")
    assert revealed["title"] == "桂花乌龙茶"
    assert revealed["payload"]["revealed"] is True
    assert revealed["payload"]["image_url"].endswith(".jpg")


# ── 用户侧接口 ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gifts_home_lists_all_in_transit_mysteries_and_only_revealed_history(monkeypatch):
    gifts = [
        _gift("out_for_delivery", id="g-door"),
        _gift("shipping", id="g-ship"),
        _gift("delivered", id="g-done", delivered_at="2025-12-01T00:00:00+00:00"),
        _gift("cancelled", id="g-cancel"),
    ]
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(gift_service, "_refresh_deliveries", AsyncMock())
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value=None))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=gifts))

    home = await gift_service.get_gifts("user-1")

    assert [g.id for g in home.shipping_gifts] == ["g-door", "g-ship"]
    assert home.shipping_gift.id == "g-door"
    assert all(g.gift_name == MYSTERY_GIFT_NAME and g.product_image_url is None for g in home.shipping_gifts)
    assert home.shipping_gifts[0].can_confirm_receipt is True
    assert [g.id for group in home.groups for g in group.gifts] == ["g-done"]
    assert home.groups[0].gifts[0].gift_name == "桂花乌龙茶"


@pytest.mark.asyncio
async def test_thanks_before_receipt_is_rejected_so_the_name_cannot_leak(monkeypatch):
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=_gift("out_for_delivery")))
    insert = AsyncMock()
    monkeypatch.setattr(gift_service, "insert_user_component_message", insert)
    with pytest.raises(HTTPException) as exc:
        await gift_service.send_thanks("user-1", "gift-1", "谢谢")
    assert exc.value.status_code == 409
    insert.assert_not_awaited()


# ── 签收 ─────────────────────────────────────────────────────────────────


def _patch_receipt(monkeypatch, *, current, cas_result, reread=None):
    get_gift = AsyncMock(side_effect=[current, reread] if reread is not None else [current])
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", get_gift)
    monkeypatch.setattr(gift_service.gift_repo, "mark_gift_received", AsyncMock(return_value=cas_result))
    add_events = AsyncMock()
    monkeypatch.setattr(gift_service.gift_repo, "add_tracking_events", add_events)
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(gift_service, "gift_received_message", AsyncMock(return_value="是桂花乌龙茶，记得你爱花香"))
    emit = AsyncMock(return_value="msg-1")
    monkeypatch.setattr(gift_service, "emit_gift_card", emit)
    remember = Mock()
    monkeypatch.setattr(gift_service, "remember_user_event", remember)
    return add_events, emit, remember


@pytest.mark.asyncio
async def test_user_confirms_receipt_reveals_gift_once(monkeypatch):
    add_events, emit, remember = _patch_receipt(
        monkeypatch, current=_gift("out_for_delivery"), cas_result=_gift("delivered"),
    )

    result = await gift_service.confirm_receipt("user-1", "gift-1")

    assert result.already_confirmed is False
    assert result.gift.revealed is True
    assert result.gift.gift_name == "桂花乌龙茶"
    assert result.assistant_message == "是桂花乌龙茶，记得你爱花香"
    event = add_events.await_args.args[1][0]
    assert event["status"] == "delivered" and event["description"] == "你已确认收到这份礼物"
    card_kwargs = emit.await_args.kwargs
    assert card_kwargs["trigger_type"] == "gift_received"
    assert card_kwargs["gift"]["status"] == "delivered"
    assert "桂花乌龙茶" in remember.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_receipt_requires_out_for_delivery(monkeypatch):
    add_events, emit, _ = _patch_receipt(
        monkeypatch, current=_gift("shipping"), cas_result=None, reread=_gift("shipping"),
    )
    with pytest.raises(HTTPException) as exc:
        await gift_service.confirm_receipt("user-1", "gift-1")
    assert exc.value.status_code == 409
    add_events.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeat_or_concurrent_receipt_is_idempotent(monkeypatch):
    # 已签收：直接返回，不再发消息。
    _, emit, _ = _patch_receipt(monkeypatch, current=_gift("delivered"), cas_result=None)
    result = await gift_service.confirm_receipt("user-1", "gift-1")
    assert result.already_confirmed is True and result.gift.revealed is True
    emit.assert_not_awaited()

    # 并发：CAS 输给另一请求，重读发现已签收。
    _, emit, _ = _patch_receipt(
        monkeypatch, current=_gift("out_for_delivery"), cas_result=None, reread=_gift("delivered"),
    )
    result = await gift_service.confirm_receipt("user-1", "gift-1")
    assert result.already_confirmed is True
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_survives_announcement_failure(monkeypatch):
    _, emit, remember = _patch_receipt(
        monkeypatch, current=_gift("out_for_delivery"), cas_result=_gift("delivered"),
    )
    emit.side_effect = RuntimeError("ws down")
    result = await gift_service.confirm_receipt("user-1", "gift-1")
    assert result.gift.revealed is True
    assert result.assistant_message is None
    remember.assert_called_once()


# ── 派送中迁移 ────────────────────────────────────────────────────────────


def _patch_door(monkeypatch, *, cas_result, notice_sent=False):
    cas = AsyncMock(return_value=cas_result)
    monkeypatch.setattr(gift_service.gift_repo, "mark_gift_out_for_delivery", cas)
    monkeypatch.setattr(gift_service.gift_repo, "gift_notice_sent", AsyncMock(return_value=notice_sent))
    add_events = AsyncMock()
    monkeypatch.setattr(gift_service.gift_repo, "add_tracking_events", add_events)
    monkeypatch.setattr(gift_service, "gift_out_for_delivery_message", AsyncMock(return_value="快递在派送啦"))
    emit = AsyncMock(return_value="msg-door")
    monkeypatch.setattr(gift_service, "emit_gift_card", emit)
    return cas, add_events, emit


@pytest.mark.asyncio
async def test_manual_gift_refresh_never_touches_providers(monkeypatch):
    sync = AsyncMock(return_value=None)
    monkeypatch.setattr(gift_service.gift_fulfillment, "sync_tracking_events", sync)
    cas, _, emit = _patch_door(monkeypatch, cas_result=_gift("out_for_delivery"))
    gift = _gift("shipping", delivered_at="2020-01-01T00:00:00+00:00")
    assert await gift_service._refresh_one_delivery("user-1", gift, CTX) is gift
    cas.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_courier_delivered_moves_to_out_for_delivery_and_notifies_once(monkeypatch):
    snapshot = GiftTrackingSnapshot(current_status="delivered")
    monkeypatch.setattr(gift_service.gift_fulfillment, "sync_tracking_events", AsyncMock(return_value=snapshot))
    cas, _, emit = _patch_door(monkeypatch, cas_result=_gift("out_for_delivery", provider="custom_http"))
    result = await gift_service._refresh_one_delivery("user-1", _gift("shipping", provider="custom_http"), CTX)
    assert result["status"] == "out_for_delivery"
    assert emit.await_args.kwargs["trigger_type"] == "gift_out_for_delivery"
    assert emit.await_args.kwargs["gift"]["gift_name"] == "桂花乌龙茶"  # 卡片由 builder 脱敏

    # 第二次刷新：CAS 落空，不重复提醒。
    cas.return_value = None
    emit.reset_mock()
    await gift_service._refresh_one_delivery("user-1", _gift("shipping", provider="custom_http"), CTX)
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_out_for_delivery_gift_keeps_syncing_courier_events_without_renotifying(monkeypatch):
    sync = AsyncMock(return_value=GiftTrackingSnapshot(current_status="delivered"))
    monkeypatch.setattr(gift_service.gift_fulfillment, "sync_tracking_events", sync)
    cas, _, emit = _patch_door(monkeypatch, cas_result=None)  # 已经是派送中：CAS 落空
    gift = _gift("out_for_delivery", provider="custom_http")
    assert await gift_service._refresh_one_delivery("user-1", gift, CTX) is gift
    sync.assert_awaited_once()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_door_reminder_is_not_repeated_after_status_correction(monkeypatch):
    # 物流纠错退回运输中后再次进入派送中：聊天里已有提醒，就不再发。
    monkeypatch.setattr(
        gift_service.gift_fulfillment, "sync_tracking_events",
        AsyncMock(return_value=GiftTrackingSnapshot(current_status="delivered")),
    )
    cas, _, emit = _patch_door(monkeypatch, cas_result=_gift("out_for_delivery", provider="mock"), notice_sent=True)
    await gift_service._refresh_one_delivery("user-1", _gift("shipping", provider="mock"), CTX)
    cas.assert_awaited_once()
    emit.assert_not_awaited()


def test_provider_reporting_delivered_at_purchase_still_requires_signing():
    assert gift_service._initial_order_status("delivered") == "shipping"
    assert gift_service._initial_order_status("out_for_delivery") == "shipping"
    assert gift_service._initial_order_status("shipping") == "shipping"
    assert gift_service._initial_order_status("weird") == "ordered"


@pytest.mark.asyncio
async def test_provider_eta_passing_counts_as_courier_at_door(monkeypatch):
    monkeypatch.setattr(
        gift_service.gift_fulfillment, "sync_tracking_events",
        AsyncMock(return_value=GiftTrackingSnapshot(current_status="shipping")),
    )
    cas, add_events, emit = _patch_door(monkeypatch, cas_result=_gift("out_for_delivery", provider="mock"))
    eta = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    await gift_service._refresh_one_delivery("user-1", _gift("shipping", provider="mock", delivered_at=eta), CTX)
    cas.assert_awaited_once()
    assert add_events.await_args.args[1][0]["status"] == "out_for_delivery"
    emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_skips_manual_and_records_courier_delivery_as_out_for_delivery(monkeypatch):
    provider = Mock(name="provider")
    provider.name = "custom_http"
    provider.fetch_tracking = AsyncMock(return_value=GiftTrackingSnapshot(
        current_status="delivered",
        tracking_number="T1",
        events=[GiftTrackingUpdate(status="delivered", title="已签收，驿站代收")],
    ))
    monkeypatch.setattr(gift_fulfillment, "get_logistics_provider", lambda: provider)
    add_events = AsyncMock()
    snapshot_update = AsyncMock()
    monkeypatch.setattr(gift_fulfillment.gift_repo, "add_tracking_events", add_events)
    monkeypatch.setattr(gift_fulfillment.gift_repo, "update_tracking_snapshot", snapshot_update)

    assert await gift_fulfillment.sync_tracking_events("user-1", _gift("shipping", provider="manual")) is None
    provider.fetch_tracking.assert_not_awaited()

    await gift_fulfillment.sync_tracking_events("user-1", _gift("shipping", provider="custom_http"))
    assert add_events.await_args.args[1][0]["status"] == "out_for_delivery"
    # 派送中必须走 CAS（带幂等通知），快照更新不直接改状态。
    assert snapshot_update.await_args.kwargs["status"] is None


@pytest.mark.asyncio
async def test_sent_message_and_memory_never_see_the_gift(monkeypatch):
    rendered = {}

    class _Template:
        def format(self, **kwargs):
            rendered.update(kwargs)
            return "prompt"

    from app.services.offline import gift_messages

    monkeypatch.setattr(gift_messages.repo, "memory_brief", AsyncMock(return_value="- 喜欢喝茶"))
    monkeypatch.setattr(gift_messages, "get_prompt_text", AsyncMock(return_value=_Template()))
    monkeypatch.setattr(gift_messages, "invoke_text", AsyncMock(return_value="我给你寄了个小惊喜，先保密"))
    assert await gift_messages.gift_sent_message("user-1", "ws-1") == "我给你寄了个小惊喜，先保密"
    assert set(rendered) == {"memory"}

    remember = Mock()
    monkeypatch.setattr(gift_service, "remember_user_event", remember)
    gift_service.remember_mystery_gift_sent("user-1", "ws-1")
    assert "神秘礼物" in remember.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_mock_injection_out_for_delivery_stage(monkeypatch):
    monkeypatch.setattr(gift_service.gift_fulfillment, "commerce_provider_name", lambda: "mock")
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value={"id": "a"}))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=[]))
    monkeypatch.setattr(gift_service, "_fulfill_gift", AsyncMock(return_value=_gift("shipping", provider="mock")))
    advance = AsyncMock(return_value=_gift("out_for_delivery", provider="mock"))
    monkeypatch.setattr(gift_service, "advance_to_out_for_delivery", advance)
    confirm = AsyncMock()
    monkeypatch.setattr(gift_service, "_confirm_receipt", confirm)

    gift = await gift_service.create_mock_gift_for_user(user_id="user-1", stage="out_for_delivery")
    assert gift["status"] == "out_for_delivery"
    confirm.assert_not_awaited()


def test_legacy_delivered_flag_maps_to_delivered_stage(api_client, auth_header, monkeypatch):
    create = AsyncMock(return_value=_gift("delivered"))
    monkeypatch.setattr(gift_service, "create_mock_gift_for_user", create)
    from app.api.public import offline

    monkeypatch.setattr(offline, "is_gift_enabled", AsyncMock(return_value=True))
    response = api_client.post("/offline/admin/gifts/mock?delivered=true", headers=auth_header("admin-1", role="admin"))
    assert response.status_code == 200
    assert create.await_args.kwargs["stage"] == "delivered"
    assert response.json()["revealed"] is True


@pytest.mark.parametrize("role,status", [(None, 401), ("user", 200)])
def test_confirm_receipt_endpoint_uses_jwt_owner(api_client, auth_header, monkeypatch, role, status):
    from app.models.offline import GiftReceiptResponse, RealWorldGiftItem

    confirm = AsyncMock(return_value=GiftReceiptResponse(
        ok=True, gift=RealWorldGiftItem(**user_view(_gift("delivered"))),
    ))
    monkeypatch.setattr(gift_service, "confirm_receipt", confirm)
    headers = auth_header("owner", role=role) if role else {}
    response = api_client.post("/offline/gifts/gift-1/confirm-receipt", headers=headers)
    assert response.status_code == status
    if role:
        confirm.assert_awaited_once_with("owner", "gift-1")
