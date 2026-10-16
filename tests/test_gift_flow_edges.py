"""礼物流程的边界与失败路径：后台路由、缺数据/冲突、通知失败、文案兜底、推荐清洗。"""

import base64
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.models.admin_gifts import (
    AdminGiftCreateRequest,
    AdminGiftRecommendationRequest,
    AdminGiftUpdateRequest,
)
from app.services.offline import activity_service, gift_messages, gift_service
from app.services.offline import gift_admin_repository as admin_repo
from app.services.offline import gift_admin_service as svc
from app.services.offline import gift_recommendation as rec
from app.services.offline.gift_selection import _parse_index
from app.services.offline.providers.gift_types import GiftProviderError

NOW = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
CTX = {"workspace_id": "ws-1", "agent_id": "agent-1", "conversation_id": "conv-1"}
WORKSPACE = {"workspace_id": "ws-1", "agent_id": "agent-1", "agent_name": "阿澈", "conversation_id": "conv-1"}


def _gift(status="shipping", **extra):
    gift = {
        "id": "g" * 32, "user_id": "user-1", "agent_id": "agent-1", "workspace_id": "ws-1",
        "conversation_id": "conv-1", "status": status, "trigger_type": "admin_manual", "provider": "manual",
        "gift_name": "暖手宝", "gift_reason": None, "gift_note": "揣着它", "product_image_url": None,
        "paid_amount_cents": 0, "carrier": "顺丰", "tracking_number": "SF1", "address_snapshot": {},
        "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
    }
    gift.update(extra)
    return gift


def _create(**extra):
    data = {"request_id": "3f1c2a7e-8c1b-4b8e-9d55-0a2f6e3b9c11", "user_id": "user-1", "gift_name": "暖手宝"}
    data.update(extra)
    return AdminGiftCreateRequest(**data)


@pytest.fixture
def plain_response(monkeypatch):
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(side_effect=lambda gift_id: {"id": gift_id}))
    monkeypatch.setattr(svc, "AdminGiftActionResponse", lambda **kw: kw)


# ── 后台路由：鉴权之后的转发 ──────────────────────────────────────────────


def test_admin_routes_forward_payloads_and_operator(api_client, auth_header, monkeypatch):
    detail = {**_gift(), "events": [], "messages": []}
    action = {"gift": detail, "notification": "skipped"}
    overview = {
        "user": {"user_id": "user-1", "username": "u"}, "workspaces": [], "budget": {
            "recharge_total_cents": 0, "historical_spend_cents": 0, "available_cents": 0},
    }
    batch = {"batch_id": "b", "generated_at": NOW.isoformat(), "budget_label": "x", "candidates": []}
    calls = {}

    def record(name, value):
        async def fake(*args, **kwargs):
            calls[name] = (args, kwargs)
            return value
        monkeypatch.setattr(svc, name, fake)

    record("user_overview", overview)
    record("recommend", batch)
    record("list_gifts", {"items": [_gift()], "total": 1})
    record("gift_detail", detail)
    record("update_gift", detail)
    record("set_stage", action)
    record("cancel_gift", detail)
    from app.services import game_points

    monkeypatch.setattr(game_points, "search_users", AsyncMock(return_value=[{"user_id": "user-1", "username": "u"}]))
    headers = auth_header("admin-7", role="admin")

    assert api_client.get("/admin-api/gifts/users?q=u", headers=headers).json()[0]["user_id"] == "user-1"
    assert api_client.get("/admin-api/gifts/users/user-1", headers=headers).status_code == 200
    assert api_client.post("/admin-api/gifts/users/user-1/recommendations", json={"count": 3},
                           headers=headers).status_code == 200
    assert calls["recommend"][0][1].count == 3
    listed = api_client.get("/admin-api/gifts?status=out_for_delivery&q=SF&limit=10&offset=20", headers=headers)
    assert listed.json()["total"] == 1
    assert calls["list_gifts"][1] == {
        "status_filter": "out_for_delivery", "user_id": None, "query": "SF", "limit": 10, "offset": 20,
    }
    assert api_client.get("/admin-api/gifts/g1", headers=headers).status_code == 200
    assert api_client.patch("/admin-api/gifts/g1", json={"gift_note": "新"}, headers=headers).status_code == 200
    assert calls["update_gift"][0][:2] == ("admin-7", "g1")
    assert api_client.post("/admin-api/gifts/g1/stage", json={"stage": "out_for_delivery"},
                           headers=headers).status_code == 200
    assert calls["set_stage"][0] == ("admin-7", "g1", "out_for_delivery")
    assert api_client.post("/admin-api/gifts/g1/cancel", json={"reason": "寄错"}, headers=headers).status_code == 200
    assert calls["cancel_gift"][0] == ("admin-7", "g1", "寄错")
    # 签收是第四个节点，只能由用户在 App 里完成：后台既不能推进到签收，也没有代签收接口。
    assert api_client.post("/admin-api/gifts/g1/stage", json={"stage": "delivered"},
                           headers=headers).status_code == 422
    assert api_client.post("/admin-api/gifts/g1/confirm-receipt", headers=headers).status_code in {404, 405}


# ── 概览 / 推荐 / 详情 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_overview_exposes_full_address_and_survives_unconfigured_provider(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value={"user_id": "user-1", "username": "u"}))
    monkeypatch.setattr(svc.admin_repo, "list_user_workspaces", AsyncMock(return_value=[WORKSPACE]))
    address = AsyncMock(return_value={"recipient_name": "小悠", "phone": "13800000000", "display": "上海 测试路 1 号"})
    monkeypatch.setattr(svc.gift_repo, "default_address", address)
    monkeypatch.setattr(svc.admin_repo, "list_gifts", AsyncMock(return_value=([_gift()], 1)))
    monkeypatch.setattr(svc.gift_repo, "recharge_total_cents", AsyncMock(return_value=10000))
    monkeypatch.setattr(svc.gift_repo, "historical_gift_spend_cents", AsyncMock(return_value=2000))
    monkeypatch.setattr(svc, "available_gift_budget_cents", AsyncMock(return_value=6000))
    monkeypatch.setattr(svc.repo, "list_user_tags", AsyncMock(return_value=["咖啡"]))
    monkeypatch.setattr(svc.admin_repo, "count_user_memories", AsyncMock(return_value=12))
    monkeypatch.setattr(svc, "is_gift_enabled", AsyncMock(return_value=True))

    def broken_provider():
        raise GiftProviderError("missing key")

    monkeypatch.setattr(svc.gift_fulfillment, "commerce_provider_name", broken_provider)
    overview = await svc.user_overview("user-1")
    assert address.await_args.kwargs == {"masked": False}
    assert overview.address.phone == "13800000000"
    assert overview.budget.available_cents == 6000
    assert overview.commerce_provider == "unconfigured"
    assert overview.auto_gift_enabled is True and overview.memory_count == 12

    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as missing:
        await svc.user_overview("ghost")
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_recommend_scopes_workspace_to_the_user(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value={"user_id": "user-1"}))
    monkeypatch.setattr(svc.admin_repo, "list_user_workspaces", AsyncMock(return_value=[WORKSPACE]))
    generate = AsyncMock(return_value={
        "batch_id": "b", "generated_at": NOW.isoformat(), "budget_label": "x", "candidates": [],
    })
    monkeypatch.setattr(svc.gift_recommendation, "generate_recommendations", generate)
    result = await svc.recommend("user-1", AdminGiftRecommendationRequest(
        workspace_id="ws-1", exclude_names=["  香薰  ", "", "  "],
    ))
    assert result.batch_id == "b"
    assert generate.await_args.kwargs["workspace"] == WORKSPACE
    assert generate.await_args.kwargs["exclude_names"] == ["香薰"]

    with pytest.raises(HTTPException) as foreign:
        await svc.recommend("user-1", AdminGiftRecommendationRequest(workspace_id="someone-elses"))
    assert foreign.value.status_code == 404
    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as missing:
        await svc.recommend("ghost", AdminGiftRecommendationRequest())
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_detail_and_list_combine_repository_views(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_gift()))
    monkeypatch.setattr(svc.admin_repo, "list_tracking_events", AsyncMock(return_value=[{
        "id": "e", "status": "shipping", "title": "已发出", "occurred_at": NOW.isoformat(),
    }]))
    messages = AsyncMock(return_value=[{
        "id": "m", "role": "assistant", "trigger_type": "gift_sent", "text_preview": "惊喜",
        "has_trace": True, "created_at": NOW.isoformat(),
    }])
    monkeypatch.setattr(svc.admin_repo, "gift_messages", messages)
    detail = await svc.gift_detail("g")
    assert [e.title for e in detail.events] == ["已发出"] and detail.messages[0].has_trace
    messages.assert_awaited_once_with("g", "user-1")

    monkeypatch.setattr(svc.admin_repo, "list_gifts", AsyncMock(return_value=([_gift()], 41)))
    listed = await svc.list_gifts(status_filter="in_transit", user_id=None, query="暖", limit=20, offset=20)
    assert listed.total == 41 and listed.items[0].gift_name == "暖手宝"

    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as missing:
        await svc.gift_detail("nope")
    assert missing.value.status_code == 404


# ── 手动寄送的失败路径 ────────────────────────────────────────────────────


def _patch_create(monkeypatch, *, user=True, workspaces=(WORKSPACE,), ctx=CTX, inserted=None):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=None))
    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value={"user_id": "user-1"} if user else None))
    monkeypatch.setattr(svc.admin_repo, "list_user_workspaces", AsyncMock(return_value=list(workspaces)))
    monkeypatch.setattr(svc.repo, "resolve_user_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(svc.gift_repo, "default_address", AsyncMock(return_value={"city": "上海"}))
    insert = AsyncMock(return_value=inserted or (_gift("ordered"), True))
    monkeypatch.setattr(svc.admin_repo, "insert_manual_gift", insert)
    add_events = AsyncMock()
    monkeypatch.setattr(svc.gift_repo, "add_tracking_events", add_events)
    monkeypatch.setattr(svc.gift_repo, "update_last_gift_paid", AsyncMock())
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", AsyncMock(return_value=False))
    monkeypatch.setattr(svc, "gift_sent_message", AsyncMock(return_value="有个惊喜"))
    emit = AsyncMock(return_value="msg")
    monkeypatch.setattr(svc, "emit_gift_card", emit)
    remember = Mock()
    monkeypatch.setattr(svc.gift_memory, "remember_gift_sent", remember)
    return insert, add_events, emit, remember


@pytest.mark.asyncio
@pytest.mark.parametrize("case,status", [
    ("no_user", 404), ("no_workspace", 409), ("no_conversation", 409), ("id_conflict", 409),
])
async def test_manual_send_refuses_without_a_destination(monkeypatch, plain_response, case, status):
    insert, *_ = _patch_create(
        monkeypatch,
        user=case != "no_user",
        workspaces=() if case == "no_workspace" else (WORKSPACE,),
        ctx={**CTX, "conversation_id": None} if case == "no_conversation" else CTX,
        inserted=(None, False) if case == "id_conflict" else None,
    )
    with pytest.raises(HTTPException) as exc:
        await svc.create_manual_gift("admin-1", _create())
    assert exc.value.status_code == status
    if case != "id_conflict":
        insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_send_cleans_up_upload_when_insert_fails(monkeypatch, plain_response):
    insert, *_ = _patch_create(monkeypatch)
    insert.side_effect = RuntimeError("db down")
    monkeypatch.setattr(svc.activity_media_storage, "save_image_blob", Mock(return_value="user-1_gift_a.png"))
    deleted = Mock()
    monkeypatch.setattr(svc.activity_media_storage, "delete_media_file", deleted)
    png = base64.b64encode(b"\x89PNG").decode()
    with pytest.raises(RuntimeError):
        await svc.create_manual_gift("admin-1", _create(image_base64=png, image_mime="image/png"))
    deleted.assert_called_once_with("user-1_gift_a.png")


@pytest.mark.asyncio
async def test_ordered_send_with_failed_notification_keeps_gift_and_skips_memory(monkeypatch, plain_response):
    insert, add_events, emit, remember = _patch_create(monkeypatch)
    fetch = AsyncMock(return_value=(b"\x89PNG", "image/png"))
    monkeypatch.setattr(svc, "fetch_gift_image", fetch)
    monkeypatch.setattr(svc.activity_media_storage, "save_image_blob", Mock(return_value="user-1_gift_c.png"))
    emit.side_effect = RuntimeError("ws down")
    result = await svc.create_manual_gift("admin-1", _create(image_url="https://img.example/a.jpg"))
    fetch.assert_awaited_once_with("https://img.example/a.jpg")
    assert insert.await_args.args[0]["product_image_url"] == "/offline/media/user-1_gift_c.png"  # 外链已转存
    assert result["notification"] == "failed"
    assert [e["status"] for e in add_events.await_args.args[1]] == ["ordered"]
    remember.assert_not_called()  # 用户没收到消息，就不写「寄出了神秘礼物」的记忆


@pytest.mark.asyncio
async def test_replaying_a_cancelled_gift_does_nothing(monkeypatch, plain_response):
    _, add_events, emit, _ = _patch_create(monkeypatch)
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_gift("cancelled")))
    result = await svc.create_manual_gift("admin-1", _create())
    assert result["notification"] == "skipped"
    add_events.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_validation_amount_and_cancelled_refusal(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_gift()))
    update = AsyncMock(return_value=_gift())
    monkeypatch.setattr(svc.admin_repo, "update_gift_details", update)
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))

    with pytest.raises(HTTPException) as empty:
        await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(gift_name=None))
    assert empty.value.status_code == 422
    await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(paid_amount_yuan=19.99))
    assert update.await_args.args[1] == {"paid_amount_cents": 1999}

    update.return_value = None
    monkeypatch.setattr(svc.activity_media_storage, "save_image_blob", Mock(return_value="user-1_gift_b.png"))
    deleted = Mock()
    monkeypatch.setattr(svc.activity_media_storage, "delete_media_file", deleted)
    png = base64.b64encode(b"\x89PNG").decode()
    with pytest.raises(HTTPException) as cancelled:
        await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(image_base64=png, image_mime="image/png"))
    assert cancelled.value.status_code == 409
    deleted.assert_called_once_with("user-1_gift_b.png")


@pytest.mark.asyncio
async def test_cancel_trims_reason(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_gift("shipping")))
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))
    cancel = AsyncMock(return_value=True)
    monkeypatch.setattr(svc.admin_repo, "cancel_gift", cancel)
    notice = AsyncMock(return_value=True)
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", notice)
    forget = Mock()
    monkeypatch.setattr(svc.gift_memory, "remember_gift_cancelled", forget)
    await svc.cancel_gift("admin-1", "g", "  寄错了  ")
    cancel.assert_awaited_once_with("g", "寄错了")
    # 用户已被告知有礼物在路上：AI 记忆里也要记下它不会到了。
    assert forget.call_args.args[0] == "user-1"

    forget.reset_mock()
    notice.return_value = False
    await svc.cancel_gift("admin-1", "g", "寄错了")
    forget.assert_not_called()  # 用户从没听说过这份礼物，就不必记


# ── 用户侧服务的剩余分支 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_gift_and_thanks_after_receipt_stay_consistent(monkeypatch):
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=_gift("out_for_delivery")))
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    item = await gift_service.get_gift("user-1", "g")
    assert item.gift_name == "神秘礼物" and item.can_confirm_receipt

    thanked = _gift("delivered", thanks_sent_at=NOW.isoformat())
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=thanked))
    insert = AsyncMock()
    monkeypatch.setattr(gift_service, "insert_user_component_message", insert)
    result = await gift_service.send_thanks("user-1", "g", "谢谢")
    assert result.gift.gift_name == "暖手宝" and result.assistant_message is None
    insert.assert_not_awaited()

    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as missing:
        await gift_service.confirm_receipt("user-1", "nope")
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_scheduler_refresh_counts_door_transitions(monkeypatch):
    gifts = [_gift("shipping", id="a"), _gift("delivered", id="b"), _gift("ordered", id="c")]
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=gifts))
    results = {"a": _gift("out_for_delivery", id="a"), "b": gifts[1], "c": gifts[2]}
    monkeypatch.setattr(gift_service, "_refresh_one_delivery", AsyncMock(side_effect=lambda u, g, c: results[g["id"]]))
    assert await gift_service.refresh_due_gift_deliveries("user-1", ctx=CTX) == 1


@pytest.mark.asyncio
async def test_refresh_leaves_closed_and_undue_gifts_alone(monkeypatch):
    sync = AsyncMock(return_value=gift_service.gift_fulfillment.GiftTrackingSnapshot(current_status="shipping"))
    monkeypatch.setattr(gift_service.gift_fulfillment, "sync_tracking_events", sync)
    advance = AsyncMock()
    monkeypatch.setattr(gift_service, "advance_to_out_for_delivery", advance)
    closed = _gift("delivered")
    assert await gift_service._refresh_one_delivery("user-1", closed, CTX) is closed
    future = _gift("shipping", provider="mock", delivered_at=(datetime.now(UTC) + timedelta(days=2)).isoformat())
    assert await gift_service._refresh_one_delivery("user-1", future, CTX) is future
    sync.assert_awaited_once()
    advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_shipped_notice_is_sent_once_and_reports_failures(monkeypatch):
    notice = AsyncMock(return_value=False)
    monkeypatch.setattr(gift_service.gift_repo, "gift_notice_sent", notice)
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(gift_service, "gift_shipped_message", AsyncMock(return_value="礼物出发啦"))
    emit = AsyncMock(return_value="msg-ship")
    monkeypatch.setattr(gift_service, "emit_gift_card", emit)
    gift = _gift("shipping")

    assert await gift_service.notify_shipped_once("user-1", gift) == "sent"
    kwargs = emit.await_args.kwargs
    assert kwargs["trigger_type"] == "gift_shipped" and kwargs["message"] == "礼物出发啦"
    assert kwargs["gift"]["status"] == "shipping" and kwargs["conversation_id"] == "conv-1"
    assert notice.await_args.args == (gift["id"], "user-1", "gift_shipped")

    emit.return_value = None
    assert await gift_service.notify_shipped_once("user-1", gift) == "failed"
    emit.side_effect = RuntimeError("ws down")
    assert await gift_service.notify_shipped_once("user-1", gift) == "failed"
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    assert await gift_service.notify_shipped_once("user-1", gift) == "failed"

    # 发过一次就不再发（回退后再推进到运输也一样）。
    emit.reset_mock()
    notice.return_value = True
    assert await gift_service.notify_shipped_once("user-1", gift) == "skipped"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_notifications_degrade_without_context_or_on_errors(monkeypatch):
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    assert await gift_service.notify_out_for_delivery("user-1", _gift()) is None
    assert await gift_service._announce_received("user-1", _gift("delivered"), None) is None

    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(side_effect=RuntimeError("db")))
    assert await gift_service.notify_out_for_delivery("user-1", _gift()) is None


@pytest.mark.asyncio
async def test_mock_received_stage_reuses_the_receipt_path(monkeypatch):
    confirm = AsyncMock(return_value=(_gift("delivered"), "揭晓", False))
    monkeypatch.setattr(gift_service, "_confirm_receipt", confirm)

    monkeypatch.setattr(gift_service.gift_fulfillment, "commerce_provider_name", lambda: "mock")
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(gift_service.gift_repo, "default_address", AsyncMock(return_value={"id": "a"}))
    monkeypatch.setattr(gift_service.gift_repo, "list_gifts", AsyncMock(return_value=[]))
    monkeypatch.setattr(gift_service, "_fulfill_gift", AsyncMock(return_value=_gift("shipping", provider="mock")))
    monkeypatch.setattr(gift_service, "advance_to_out_for_delivery",
                        AsyncMock(return_value=_gift("out_for_delivery", provider="mock")))
    gift = await gift_service.create_mock_gift_for_user(user_id="user-1", stage="delivered")
    assert gift["status"] == "delivered"
    assert confirm.await_args.kwargs == {"actor": "admin"}


@pytest.mark.asyncio
async def test_order_cancelled_mid_purchase_is_not_announced(monkeypatch):
    monkeypatch.setattr(gift_service.gift_fulfillment, "commerce_provider_name", lambda: "mock")
    monkeypatch.setattr(gift_service.gift_fulfillment, "logistics_provider_name", lambda: "mock")
    monkeypatch.setattr(gift_service.gift_repo, "create_gift", AsyncMock(return_value=_gift("selecting")))
    candidate = Mock(external_product_id="p", image_url=None, product_url=None)
    order = Mock(status="shipping", product_image_url=None, paid_amount_cents=100, provider="mock",
                 provider_order_id="o", tracking_number="t", raw={}, shipped_at=None, delivered_at=None)
    monkeypatch.setattr(gift_service.gift_fulfillment, "purchase_gift", AsyncMock(return_value=(candidate, order)))
    monkeypatch.setattr(gift_service.gift_fulfillment, "candidate_snapshot", lambda c: {})
    # 后台在下单过程中撤销了：落库的更新被状态守卫挡下。
    monkeypatch.setattr(gift_service.gift_repo, "update_gift_order_details", AsyncMock(return_value=None))
    emit = AsyncMock()
    monkeypatch.setattr(gift_service, "emit_gift_card", emit)
    spec = {"gift_name": "茶", "gift_reason": "r", "gift_note": "n", "amount_cents": 100}
    result = await gift_service._fulfill_gift(
        user_id="user-1", ctx=CTX, address={"city": "上海"}, trigger_type="admin_mock", spec_override=spec,
    )
    assert result is None
    emit.assert_not_awaited()


# ── AI 文案 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_door_and_reveal_messages_render_only_what_each_stage_may_know(monkeypatch):
    rendered = []

    class _Template:
        def format(self, **kwargs):
            rendered.append(kwargs)
            return "prompt"

    monkeypatch.setattr(gift_messages.repo, "memory_brief", AsyncMock(return_value="- 怕冷"))
    monkeypatch.setattr(gift_messages, "get_prompt_text", AsyncMock(return_value=_Template()))
    monkeypatch.setattr(gift_messages, "invoke_text", AsyncMock(return_value="快递在派送啦，记得签收"))
    monkeypatch.setattr(gift_messages, "get_chat_model", lambda: object())
    assert await gift_messages.gift_out_for_delivery_message("user-1", "ws-1") == "快递在派送啦，记得签收"
    assert await gift_messages.gift_shipped_message("user-1", "ws-1") == "快递在派送啦，记得签收"
    assert set(rendered.pop()) == {"memory"}  # 发货提醒同样拿不到任何礼物信息
    monkeypatch.setattr(gift_messages, "invoke_text", AsyncMock(return_value="是暖手宝！"))
    assert await gift_messages.gift_received_message("user-1", "ws-1", _gift("delivered")) == "是暖手宝！"
    assert set(rendered[0]) == {"memory"}
    assert rendered[1]["gift_name"] == "暖手宝" and rendered[1]["gift_note"] == "揣着它"

    monkeypatch.setattr(gift_messages, "invoke_text", AsyncMock(side_effect=RuntimeError("llm down")))
    door = await gift_messages.gift_out_for_delivery_message("user-1", "ws-1")
    shipped = await gift_messages.gift_shipped_message("user-1", "ws-1")
    assert "发货" in shipped and "暖手宝" not in shipped and "签收" not in shipped
    reveal = await gift_messages.gift_received_message("user-1", "ws-1", _gift("delivered"))
    assert "签收" in door and "暖手宝" not in door
    assert "暖手宝" in reveal


# ── 推荐清洗 ─────────────────────────────────────────────────────────────


def test_candidate_cleanup_handles_messy_model_output():
    tags = ["咖啡"]
    memories = [{"content": "怕冷", "category": "生活", "updated_at": None}]
    items = [
        "not a dict",
        {"gift_name": "暖手宝", "reason": "怕冷", "gift_note": "揣着",
         "price_min_yuan": "120", "price_max_yuan": 40,
         "evidence": ["bad", {"ref": "X1"}, {"ref": "M1"}, {"ref": "m1"}, {"ref": "T1"}]},
        {"gift_name": "滤杯", "reason": "咖啡", "gift_note": ""},
        {"gift_name": "第三个", "reason": "超出数量"},
    ]
    candidates, dropped = rec._build_candidates(
        items, tags=tags, memories=memories, exclude=[], count=2, budget=(50, 100),
    )
    assert [c["gift_name"] for c in candidates] == ["暖手宝", "滤杯"]
    assert dropped == 1  # 只有非 dict 那条算丢弃；超出数量的直接截断
    first = candidates[0]
    assert [e["ref"] for e in first["evidence"]] == ["M1", "T1"]  # 非法/重复编号被忽略
    assert (first["price_min_yuan"], first["price_max_yuan"]) == (50, 100)  # 先交换再按预算收口
    assert candidates[1]["search_keywords"] == ["滤杯"] and candidates[1]["grounded"] is False


def test_budget_labels_and_price_swaps():
    assert rec._budget_label(None, 80) == "不超过 80 元"
    assert rec._budget_label(30, None).startswith("不低于 30 元")
    assert rec._budget_label(150, 30) == "30-150 元"
    assert rec._price_range(90, 40, (None, None)) == (40, 90)
    assert rec._price_range(10, 20, (30, None)) == (20, 20)  # 抬到下限后不能高于上限


@pytest.mark.asyncio
async def test_all_duplicate_batch_reports_failure_instead_of_empty_success(monkeypatch):
    monkeypatch.setattr(rec.repo, "list_user_tags", AsyncMock(return_value=["咖啡"]))
    monkeypatch.setattr(rec.admin_repo, "user_memories_for_gift", AsyncMock(return_value=[]))
    monkeypatch.setattr(rec.admin_repo, "gift_names_for_user", AsyncMock(return_value=["滤杯"]))
    template = Mock()
    template.format = Mock(return_value="prompt")
    monkeypatch.setattr(rec, "get_prompt_text", AsyncMock(return_value=template))
    monkeypatch.setattr(rec, "get_chat_model", lambda: object())
    monkeypatch.setattr(rec, "invoke_text", AsyncMock(
        return_value='{"candidates": [{"gift_name": "滤 杯", "reason": "x", "gift_note": "y"}]}',
    ))
    with pytest.raises(HTTPException) as exc:
        await rec.generate_recommendations(
            user_id="u", workspace=None, budget_min_yuan=None, budget_max_yuan=None, exclude_names=[], count=5,
        )
    assert exc.value.status_code == 502


# ── 杂项 ────────────────────────────────────────────────────────────────


def test_json_columns_and_llm_index_parsing():
    assert admin_repo._json_dict('{"a": 1}') == {"a": 1}
    assert admin_repo._json_dict("[1]") == {}
    assert admin_repo._json_dict("{bad") == {}
    assert admin_repo._json_dict(None) == {}
    assert _parse_index('{"index": 1, "reason": "x"}', 3) == 1
    assert _parse_index("选第 2 个", 3) == 2


@pytest.mark.asyncio
async def test_home_counts_door_gifts_and_activity_reset_keeps_gift_images(monkeypatch):
    monkeypatch.setattr(activity_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    monkeypatch.setattr(activity_service.repo, "list_activities", AsyncMock(return_value=[]))
    monkeypatch.setattr(activity_service.gift_repository, "list_gifts", AsyncMock(return_value=[
        _gift("out_for_delivery"), _gift("delivered"),
    ]))
    monkeypatch.setattr(activity_service.repo, "list_user_tags", AsyncMock(return_value=[]))
    home = await activity_service.get_home("user-1")
    assert home["shipping_gift_count"] == 1

    monkeypatch.setattr(activity_service.activity_media_repo, "delete_user_activity_media", AsyncMock(return_value=[]))
    delete_files = Mock(return_value=0)
    monkeypatch.setattr(activity_service.activity_media_storage, "delete_user_media_files", delete_files)
    monkeypatch.setattr(activity_service.repo, "clear_user_activities", AsyncMock(return_value={}))
    await activity_service.clear_all_activities("user-1")
    delete_files.assert_called_once_with("user-1", keep_gift_images=True)


@pytest.mark.asyncio
async def test_thanks_is_sent_as_a_user_gift_card_with_the_users_words(monkeypatch):
    gift = _gift("delivered", product_image_url="/offline/media/user-1_gift_a.png")
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(return_value=gift))
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    insert = AsyncMock(return_value="m")
    monkeypatch.setattr(gift_service, "insert_user_component_message", insert)
    monkeypatch.setattr(gift_service.gift_repo, "mark_gift_thanked", AsyncMock(return_value={
        **gift, "thanks_sent_at": NOW.isoformat(),
    }))
    monkeypatch.setattr(gift_service, "gift_thanks_reply", AsyncMock(return_value="喜欢就好"))
    monkeypatch.setattr(gift_service, "emit_assistant", AsyncMock())
    remember = Mock()
    monkeypatch.setattr(gift_service.gift_memory, "remember_gift_thanks", remember)

    await gift_service.send_thanks("user-1", gift["id"], "好暖和||谢谢你", client_id="c-1")

    kwargs = insert.await_args.kwargs
    card = kwargs["metadata"]["component_card"]
    assert kwargs["client_id"] == "c-1"
    assert card["payload"]["kind"] == "thanks" and card["payload"]["status_label"] == "谢谢你"
    assert card["payload"]["image_url"] == "/offline/media/user-1_gift_a.png"
    assert card["title"] == "暖手宝" and card["body"] == "好暖和 谢谢你"
    # 记忆拿到的是聊天里那条道谢消息本身，作为用户侧抽取的出处。
    memory = remember.call_args.kwargs
    assert memory["message"] == "好暖和||谢谢你" and memory["message_id"] == "m"
    assert memory["message_content"] == kwargs["content"]


@pytest.mark.asyncio
async def test_thanks_double_tap_posts_one_card_one_reply_one_memory(monkeypatch):
    gift = _gift("delivered")
    thanked = {**gift, "thanks_sent_at": NOW.isoformat()}
    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(side_effect=[gift, thanked]))
    # 另一次点击已经抢先占住「已道谢」。
    monkeypatch.setattr(gift_service.gift_repo, "mark_gift_thanked", AsyncMock(return_value=None))
    insert = AsyncMock()
    monkeypatch.setattr(gift_service, "insert_user_component_message", insert)
    reply = AsyncMock()
    monkeypatch.setattr(gift_service, "gift_thanks_reply", reply)
    remember = Mock()
    monkeypatch.setattr(gift_service.gift_memory, "remember_gift_thanks", remember)

    result = await gift_service.send_thanks("user-1", gift["id"], "谢谢")

    assert result.ok and result.assistant_message is None and result.gift.thanks_sent_at
    insert.assert_not_awaited()
    reply.assert_not_awaited()
    remember.assert_not_called()

    monkeypatch.setattr(gift_service.gift_repo, "get_gift", AsyncMock(side_effect=[gift, None]))
    with pytest.raises(HTTPException) as gone:
        await gift_service.send_thanks("user-1", gift["id"], "谢谢")
    assert gone.value.status_code == 404
