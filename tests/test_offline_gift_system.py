from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.models.offline import GiftAddressRequest
from app.services.offline import gift_service, gift_trigger_policy, scheduler
from app.services.offline.gift_amount import sample_gift_amount_cents
from app.services.offline.gift_trigger_policy import (
    decide_gift_trigger,
    gift_cooldown_days,
    gift_trigger_probability,
)
from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    RecipientAddress,
)
from app.services.offline.providers.mock_commerce import MockGiftCommerceProvider
from app.services.offline.providers.mock_logistics import MockGiftLogisticsProvider


def test_gift_amount_uses_lognormal_formula_and_minimum():
    assert sample_gift_amount_cents(400) is None
    assert sample_gift_amount_cents(10_000, normal_sample=lambda _mu, _sigma: 0) == 2_000

    high = sample_gift_amount_cents(10_000, normal_sample=lambda _mu, _sigma: 5)
    assert high == 10_000


def test_gift_address_request_validates_phone_format():
    with pytest.raises(ValidationError):
        GiftAddressRequest(
            recipient_name="小悠",
            phone="not-a-phone",
            city="镇江",
            detail="某条路 1 号",
        )

    address = GiftAddressRequest(
        recipient_name=" 小悠 ",
        phone="138 1234-5678",
        city=" 镇江 ",
        detail=" 某条路 1 号 ",
    )
    assert address.recipient_name == "小悠"
    assert address.phone == "13812345678"


def test_gift_probability_and_cooldown_match_spec_formula():
    assert gift_trigger_probability(0, 0) == 0
    assert gift_trigger_probability(1000, 0) == 0.3
    assert gift_trigger_probability(400, 200) == 0.15
    assert gift_cooldown_days(0.3) == 30
    assert gift_cooldown_days(0.0) == 183


@pytest.mark.asyncio
async def test_mock_gift_providers_follow_search_order_tracking_contract():
    commerce = MockGiftCommerceProvider()
    logistics = MockGiftLogisticsProvider()

    candidates = await commerce.search_products(
        query="手冲咖啡壶套装",
        min_amount_cents=1600,
        max_amount_cents=2400,
    )
    assert candidates
    assert candidates[0].source == "mock"

    order = await commerce.place_order(
        candidate=candidates[0],
        address=RecipientAddress(recipient_name="小悠", phone="13812345678", city="镇江", detail="路 1 号"),
        idempotency_key="gift-1",
    )
    assert order.provider_order_id.startswith("MOCK-")
    assert order.status == "shipping"

    tracking = await logistics.fetch_tracking(
        provider=order.provider,
        provider_order_id=order.provider_order_id,
        tracking_number=order.tracking_number,
    )
    assert tracking.current_status == "shipping"
    assert [event.status for event in tracking.events] == ["ordered", "packed", "shipping"]


@pytest.mark.asyncio
async def test_forced_gift_trigger_on_login_day(monkeypatch):
    monkeypatch.setattr(gift_trigger_policy.repo, "user_birthday_mmdd", AsyncMock(return_value=None))

    decision = await decide_gift_trigger(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        day=5,
        now=datetime(2026, 6, 27, 10, 0, tzinfo=UTC),
        last_gift_paid_at=None,
    )

    assert decision.should_trigger is True
    assert decision.trigger_type == "login_day_5"


@pytest.mark.asyncio
async def test_normal_gift_trigger_respects_probability_and_cooldown(monkeypatch):
    monkeypatch.setattr(gift_trigger_policy.repo, "user_birthday_mmdd", AsyncMock(return_value=None))
    monkeypatch.setattr(gift_trigger_policy.repo, "recharge_total_cents", AsyncMock(return_value=20_000))
    monkeypatch.setattr(
        gift_trigger_policy,
        "get_intimacy_data",
        AsyncMock(return_value=SimpleNamespace(growth_intimacy=400)),
    )
    now = datetime(2026, 6, 27, 10, 0, tzinfo=UTC)

    miss = await decide_gift_trigger(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        day=50,
        now=now,
        last_gift_paid_at=None,
        random_value=0.2,
    )
    assert miss.should_trigger is False
    assert miss.reason == "probability_miss"

    cooldown = await decide_gift_trigger(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        day=50,
        now=now,
        last_gift_paid_at=now - timedelta(days=10),
        random_value=0.01,
    )
    assert cooldown.should_trigger is False
    assert cooldown.reason == "cooldown"

    hit = await decide_gift_trigger(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        day=50,
        now=now,
        last_gift_paid_at=now - timedelta(days=200),
        random_value=0.01,
    )
    assert hit.should_trigger is True
    assert hit.trigger_type == "daily_probability"


@pytest.mark.asyncio
async def test_create_gift_resumes_pending_address_task(monkeypatch):
    pending = {
        "id": "gift-pending",
        "status": "pending_address",
        "trigger_type": "login_day_5",
        "target_amount_cents": 3_000,
    }
    ctx = {
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "agent_id": "agent-1",
    }
    address = {"recipient_name": "小悠", "city": "镇江", "detail": "路 1 号"}
    fulfilled = {
        **pending,
        "status": "shipping",
        "gift_name": "手冲咖啡壶套装",
        "gift_reason": "慢一点也很好。",
        "gift_note": "给你一点热气。",
        "product_image_url": "https://example.com/gift.jpg",
        "provider": "mock",
        "provider_product_id": "mock-product-1",
        "provider_order_id": "MOCK-1",
        "product_url": "https://example.com/product",
        "logistics_provider": "mock",
        "paid_amount_cents": 2000,
        "tracking_number": "RW1",
        "created_at": "2026-06-27T10:00:00Z",
        "updated_at": "2026-06-27T10:00:00Z",
    }

    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=[pending]))
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value=address))
    monkeypatch.setattr(gift_service, "available_gift_budget_cents", AsyncMock(return_value=10_000))
    monkeypatch.setattr(gift_service, "sample_gift_amount_cents", lambda _budget: 2_000)
    select_gift = AsyncMock(
        return_value={
            "gift_name": "手冲咖啡壶套装",
            "gift_reason": "慢一点也很好。",
            "gift_note": "给你一点热气。",
            "amount_cents": 2_000,
        }
    )
    monkeypatch.setattr(gift_service, "_select_gift", select_gift)
    purchase = AsyncMock(
        return_value=(
            GiftProductCandidate(
                external_product_id="mock-product-1",
                title="手冲咖啡壶套装",
                price_cents=2_000,
                image_url="https://example.com/gift.jpg",
                product_url="https://example.com/product",
                source="mock",
            ),
            GiftOrderResult(
                provider="mock",
                provider_order_id="MOCK-1",
                status="shipping",
                paid_amount_cents=2_000,
                product_image_url="https://example.com/gift.jpg",
                tracking_number="RW1",
            ),
        )
    )
    monkeypatch.setattr(gift_service.gift_fulfillment, "purchase_gift", purchase)
    update_order = AsyncMock(side_effect=[{**fulfilled, "status": "selecting"}, fulfilled])
    monkeypatch.setattr(gift_service.gift_repo, "update_gift_order_details", update_order)
    sync_tracking = AsyncMock(return_value=None)
    monkeypatch.setattr(gift_service.gift_fulfillment, "sync_tracking_events", sync_tracking)
    monkeypatch.setattr(gift_service.gift_repo, "update_last_gift_paid", AsyncMock())
    monkeypatch.setattr(gift_service, "gift_sent_message", AsyncMock(return_value="我给你寄了个小东西。"))
    monkeypatch.setattr(gift_service, "emit_assistant", AsyncMock())

    result = await gift_service.create_gift_for_user(user_id="user-1", workspace_id="workspace-1")

    assert result == fulfilled
    select_gift.assert_awaited_once_with("user-1", "workspace-1", 3_000)
    purchase.assert_awaited_once()
    sync_tracking.assert_awaited_once()
    assert update_order.await_count == 2
    assert update_order.await_args_list[-1].args[0] == "gift-pending"
    assert update_order.await_args_list[-1].args[2]["status"] == "shipping"


@pytest.mark.asyncio
async def test_create_gift_without_address_creates_pending_task_and_message(monkeypatch):
    ctx = {
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "agent_id": "agent-1",
    }
    pending = {
        "id": "gift-pending",
        "status": "pending_address",
        "trigger_type": "login_day_5",
        "gift_name": "还没写上名字的小惊喜",
        "target_amount_cents": 2_000,
        "created_at": "2026-06-27T10:00:00Z",
        "updated_at": "2026-06-27T10:00:00Z",
    }

    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=[]))
    monkeypatch.setattr(gift_service, "available_gift_budget_cents", AsyncMock(return_value=10_000))
    monkeypatch.setattr(gift_service, "sample_gift_amount_cents", lambda _budget: 2_000)
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value=None))
    create_gift = AsyncMock(return_value=pending)
    emit = AsyncMock()
    monkeypatch.setattr(gift_service.gift_repo, "create_gift", create_gift)
    monkeypatch.setattr(gift_service, "first_address_request_message", AsyncMock(return_value="我想寄你一个小惊喜。"))
    monkeypatch.setattr(gift_service, "emit_assistant", emit)

    result = await gift_service.create_gift_for_user(
        user_id="user-1",
        workspace_id="workspace-1",
        trigger_type="login_day_5",
    )

    assert result == pending
    create_gift.assert_awaited_once()
    assert create_gift.await_args.args[0]["status"] == "pending_address"
    assert create_gift.await_args.args[0]["target_amount_cents"] == 2_000
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["trigger_type"] == "gift_address_needed"
    assert emit.await_args.kwargs["source_id"] == "gift-pending"


@pytest.mark.asyncio
async def test_pending_address_resume_does_not_exceed_current_budget(monkeypatch):
    pending = {
        "id": "gift-pending",
        "status": "pending_address",
        "trigger_type": "login_day_5",
        "target_amount_cents": 3_000,
    }
    ctx = {
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "agent_id": "agent-1",
    }

    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=[pending]))
    monkeypatch.setattr(
        gift_service.gift_repo,
        "default_address",
        AsyncMock(return_value={"recipient_name": "小悠", "city": "镇江", "detail": "路 1 号"}),
    )
    monkeypatch.setattr(gift_service, "available_gift_budget_cents", AsyncMock(return_value=400))
    monkeypatch.setattr(gift_service, "sample_gift_amount_cents", lambda _budget: None)
    update_status = AsyncMock(return_value={**pending, "status": "skipped"})
    monkeypatch.setattr(gift_service.gift_repo, "update_gift_status", update_status)
    monkeypatch.setattr(gift_service, "_select_gift", AsyncMock())

    result = await gift_service.create_gift_for_user(user_id="user-1", workspace_id="workspace-1")

    assert result is None
    update_status.assert_awaited_once_with(
        "gift-pending",
        "user-1",
        "skipped",
        failure_reason="gift budget below minimum",
    )
    gift_service._select_gift.assert_not_awaited()


@pytest.mark.asyncio
async def test_gifts_home_only_groups_delivered_history(monkeypatch):
    gifts = [
        {
            "id": "gift-shipping",
            "status": "shipping",
            "trigger_type": "daily_probability",
            "gift_name": "在路上的礼物",
            "paid_amount_cents": 2000,
            "created_at": "2026-06-27T10:00:00Z",
            "updated_at": "2026-06-27T10:00:00Z",
        },
        {
            "id": "gift-pending",
            "status": "pending_address",
            "trigger_type": "daily_probability",
            "gift_name": "还没写上名字的小惊喜",
            "paid_amount_cents": 0,
            "created_at": "2026-06-27T10:00:00Z",
            "updated_at": "2026-06-27T10:00:00Z",
        },
        {
            "id": "gift-delivered",
            "status": "delivered",
            "trigger_type": "daily_probability",
            "gift_name": "已经收到的礼物",
            "paid_amount_cents": 2500,
            "delivered_at": "2026-06-25T10:00:00Z",
            "created_at": "2026-06-25T10:00:00Z",
            "updated_at": "2026-06-25T10:00:00Z",
        },
    ]
    monkeypatch.setattr(
        gift_service.repo,
        "resolve_user_context",
        AsyncMock(return_value={"workspace_id": "workspace-1", "agent_id": "agent-1"}),
    )
    monkeypatch.setattr(gift_service, "_refresh_deliveries", AsyncMock())
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value=None))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=gifts))

    response = await gift_service.get_gifts("user-1", "workspace-1")

    assert response.shipping_gift is not None
    assert response.shipping_gift.id == "gift-shipping"
    assert len(response.groups) == 1
    assert [gift.id for gift in response.groups[0].gifts] == ["gift-delivered"]


@pytest.mark.asyncio
async def test_send_thanks_writes_user_message_assistant_reply_and_memory(monkeypatch):
    gift = {
        "id": "gift-1",
        "status": "delivered",
        "trigger_type": "daily_probability",
        "workspace_id": "workspace-1",
        "gift_name": "手冲咖啡壶套装",
        "gift_reason": "慢一点也很好。",
        "paid_amount_cents": 2_000,
        "created_at": "2026-06-27T10:00:00Z",
        "updated_at": "2026-06-27T10:00:00Z",
        "thanks_sent_at": None,
    }
    updated = {
        **gift,
        "thanks_sent_at": "2026-06-27T10:10:00Z",
    }
    ctx = {
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "agent_id": "agent-1",
    }
    user_message = AsyncMock()
    assistant_message = AsyncMock()
    remember = Mock()

    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=gift))
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(gift_service, "insert_user_component_message", user_message)
    monkeypatch.setattr(gift_service.gift_repo, "mark_gift_thanked", AsyncMock(return_value=updated))
    monkeypatch.setattr(gift_service, "gift_thanks_reply", AsyncMock(return_value="收到你的谢谢，我会偷偷开心很久。"))
    monkeypatch.setattr(gift_service, "emit_assistant", assistant_message)
    monkeypatch.setattr(gift_service, "remember_user_event", remember)

    result = await gift_service.send_thanks("user-1", "gift-1", "我收到礼物啦，谢谢你")

    assert result.ok is True
    assert result.gift.thanks_sent_at == "2026-06-27T10:10:00Z"
    user_message.assert_awaited_once()
    assert user_message.await_args.kwargs["metadata"]["trigger_type"] == "gift_thanks"
    assistant_message.assert_awaited_once()
    assert assistant_message.await_args.kwargs["trigger_type"] == "gift_thanks_reply"
    remember.assert_called_once()


@pytest.mark.asyncio
async def test_scheduler_refreshes_due_gift_deliveries_before_new_trigger(monkeypatch):
    ctx = {
        "workspace_id": "workspace-1",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "conversation_id": "conversation-1",
        "user_created_at": datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        "last_gift_paid_at": datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
    }
    monkeypatch.setattr(scheduler.repo, "list_real_world_contexts", AsyncMock(return_value=[ctx]))
    monkeypatch.setattr(scheduler.repo, "ensure_trigger_state", AsyncMock(return_value={}))
    refresh = AsyncMock(return_value=1)
    monkeypatch.setattr(scheduler, "refresh_due_gift_deliveries", refresh)
    monkeypatch.setattr(scheduler, "_should_create_activity", AsyncMock(return_value=False))
    monkeypatch.setattr(
        scheduler,
        "_should_create_gift",
        AsyncMock(return_value=SimpleNamespace(should_trigger=False)),
    )

    stats = await scheduler.scan_offline_triggers()

    assert stats["gift_deliveries"] == 1
    refresh.assert_awaited_once_with("user-1", "workspace-1", ctx)
