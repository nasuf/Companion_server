"""后台礼物管理：推荐依据可追溯、手动寄送幂等、物流录入状态规则、代签收/撤销、鉴权。"""

import base64
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.models.admin_gifts import (
    AdminGiftCreateRequest,
    AdminGiftTrackingRequest,
    AdminGiftUpdateRequest,
)
from app.services.offline import gift_admin_service as svc
from app.services.offline import gift_recommendation as rec

NOW = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
CTX = {"workspace_id": "ws-1", "agent_id": "agent-1", "conversation_id": "conv-1"}
WORKSPACE = {"workspace_id": "ws-1", "agent_id": "agent-1", "agent_name": "小悠", "conversation_id": "conv-1"}
ADDRESS = {"id": "addr", "recipient_name": "小悠", "phone": "13800000000", "city": "上海", "detail": "某路 1 号"}


def _admin_gift(status="shipping", **extra):
    gift = {
        "id": "a" * 32,
        "user_id": "user-1",
        "agent_id": "agent-1",
        "workspace_id": "ws-1",
        "conversation_id": "conv-1",
        "status": status,
        "trigger_type": "admin_manual",
        "provider": "manual",
        "gift_name": "桂花乌龙茶",
        "gift_reason": None,
        "gift_note": None,
        "product_image_url": None,
        "paid_amount_cents": 0,
        "carrier": "顺丰",
        "tracking_number": "SF1",
        "address_snapshot": {},
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }
    gift.update(extra)
    return gift


def _event(status, occurred_at):
    return {"id": f"e-{status}-{occurred_at.isoformat()}", "status": status, "title": status,
            "occurred_at": occurred_at.isoformat()}


# ── 推荐：依据可追溯 ──────────────────────────────────────────────────────


MEMORIES = [
    {"id": "m1", "content": "最近在学做手冲咖啡", "category": "生活 / 爱好", "updated_at": "2026-10-01T00:00:00"},
    {"id": "m2", "content": "冬天手总是很冰", "category": "生活 / 健康", "updated_at": "2026-09-20T00:00:00"},
]


def _patch_recommendation(monkeypatch, llm_output, *, tags=("咖啡爱好者",), memories=MEMORIES, sent=()):
    monkeypatch.setattr(rec.repo, "list_user_tags", AsyncMock(return_value=list(tags)))
    monkeypatch.setattr(rec.admin_repo, "user_memories_for_gift", AsyncMock(return_value=list(memories)))
    monkeypatch.setattr(rec.admin_repo, "gift_names_for_user", AsyncMock(return_value=list(sent)))
    template = Mock()
    template.format = Mock(return_value="prompt")
    monkeypatch.setattr(rec, "get_prompt_text", AsyncMock(return_value=template))
    llm = AsyncMock(return_value=llm_output) if not isinstance(llm_output, Exception) else AsyncMock(side_effect=llm_output)
    monkeypatch.setattr(rec, "invoke_text", llm)
    monkeypatch.setattr(rec, "get_chat_model", lambda: object())
    return template


@pytest.mark.asyncio
async def test_recommendations_resolve_cited_evidence_and_drop_fabricated_refs(monkeypatch):
    output = """```json
    {"candidates": [
      {"gift_name": "手冲咖啡滤杯套装", "search_keywords": ["手冲滤杯", "咖啡滤杯"],
       "price_min_yuan": 40, "price_max_yuan": 400, "reason": "在学手冲", "gift_note": "冲一杯给自己",
       "evidence": [{"ref": "M1", "feature": "在学手冲咖啡"}, {"ref": "T1", "feature": "咖啡爱好者"},
                    {"ref": "M9", "feature": "编造的"}]},
      {"gift_name": "暖手宝", "reason": "手冰", "gift_note": "揣着", "evidence": [{"ref": "M99"}]},
      {"gift_name": "桂花乌龙茶", "reason": "已送过", "gift_note": "x", "evidence": [{"ref": "M1"}]},
      {"gift_name": "手冲咖啡滤杯套装", "reason": "重复", "gift_note": "x"},
      {"reason": "缺名字"}
    ]}
    ```"""
    template = _patch_recommendation(monkeypatch, output, sent=["桂花乌龙茶"])

    result = await rec.generate_recommendations(
        user_id="user-1", workspace=WORKSPACE, budget_min_yuan=30, budget_max_yuan=150,
        exclude_names=["香薰蜡烛"], count=5,
    )

    names = [c["gift_name"] for c in result["candidates"]]
    assert names == ["手冲咖啡滤杯套装", "暖手宝"]
    assert result["dropped_count"] == 3
    first = result["candidates"][0]
    assert [e["ref"] for e in first["evidence"]] == ["M1", "T1"]
    assert first["evidence"][0]["source_text"] == "最近在学做手冲咖啡"
    assert first["evidence"][1]["source_type"] == "tag"
    assert first["grounded"] is True
    assert (first["price_min_yuan"], first["price_max_yuan"]) == (40, 150)  # 超预算被收口
    assert result["candidates"][1]["grounded"] is False  # 引用全是不存在的编号
    rendered = template.format.call_args.kwargs
    assert "M1. [生活 / 爱好 · 2026-10-01] 最近在学做手冲咖啡" in rendered["memories"]
    assert "T1. 咖啡爱好者" in rendered["tags"]
    assert "桂花乌龙茶" in rendered["exclude"] and "香薰蜡烛" in rendered["exclude"]
    assert rendered["budget"] == "30-150 元"


@pytest.mark.asyncio
async def test_recommendations_distinguish_empty_data_from_failures(monkeypatch):
    _patch_recommendation(monkeypatch, "{}", tags=(), memories=())
    with pytest.raises(HTTPException) as empty:
        await rec.generate_recommendations(
            user_id="u", workspace=None, budget_min_yuan=None, budget_max_yuan=None, exclude_names=[], count=5,
        )
    assert empty.value.status_code == 422

    _patch_recommendation(monkeypatch, RuntimeError("provider down"))
    with pytest.raises(HTTPException) as failed:
        await rec.generate_recommendations(
            user_id="u", workspace=None, budget_min_yuan=None, budget_max_yuan=None, exclude_names=[], count=5,
        )
    assert failed.value.status_code == 502

    _patch_recommendation(monkeypatch, "抱歉我不知道")
    with pytest.raises(HTTPException) as unparsable:
        await rec.generate_recommendations(
            user_id="u", workspace=None, budget_min_yuan=None, budget_max_yuan=None, exclude_names=[], count=5,
        )
    assert unparsable.value.status_code == 502


# ── 手动寄送 ──────────────────────────────────────────────────────────────


def _create_request(**extra):
    data = {
        "request_id": "3f1c2a7e-8c1b-4b8e-9d55-0a2f6e3b9c11",
        "user_id": "user-1",
        "gift_name": " 桂花乌龙茶 ",
        "gift_note": "泡一杯",
        "paid_amount_yuan": 58.5,
        "carrier": "顺丰",
        "tracking_number": "SF1",
        "initial_status": "shipping",
    }
    data.update(extra)
    return AdminGiftCreateRequest(**data)


def _patch_create(monkeypatch, *, address=ADDRESS, inserted=(None, True)):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=None))
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", AsyncMock(return_value=False))
    monkeypatch.setattr(svc.admin_repo, "get_user", AsyncMock(return_value={"user_id": "user-1"}))
    monkeypatch.setattr(svc.admin_repo, "list_user_workspaces", AsyncMock(return_value=[WORKSPACE]))
    monkeypatch.setattr(svc.repo, "resolve_user_context", AsyncMock(return_value=CTX))
    monkeypatch.setattr(svc.gift_repo, "default_address", AsyncMock(return_value=address))
    insert = AsyncMock(return_value=inserted)
    monkeypatch.setattr(svc.admin_repo, "insert_manual_gift", insert)
    add_events = AsyncMock()
    monkeypatch.setattr(svc.gift_repo, "add_tracking_events", add_events)
    paid = AsyncMock()
    monkeypatch.setattr(svc.gift_repo, "update_last_gift_paid", paid)
    monkeypatch.setattr(svc, "gift_sent_message", AsyncMock(return_value="有个惊喜在路上"))
    emit = AsyncMock(return_value="msg-1")
    monkeypatch.setattr(svc, "emit_gift_card", emit)
    remember = Mock()
    monkeypatch.setattr(svc.gift_service, "remember_user_event", remember)
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(side_effect=lambda gift_id: {"id": gift_id}))
    monkeypatch.setattr(svc, "AdminGiftActionResponse", lambda **kw: kw)
    return insert, add_events, emit, paid, remember


@pytest.mark.asyncio
async def test_manual_gift_is_created_as_mystery_with_initial_tracking(monkeypatch):
    insert, add_events, emit, paid, remember = _patch_create(
        monkeypatch, inserted=(_admin_gift("shipping"), True),
    )

    result = await svc.create_manual_gift("admin-1", _create_request())

    data = insert.await_args.args[0]
    assert data["id"] == "3f1c2a7e8c1b4b8e9d550a2f6e3b9c11"
    assert data["gift_name"] == "桂花乌龙茶"
    assert data["paid_amount_cents"] == 5850
    assert data["logistics_provider"] == "顺丰"
    assert data["address_snapshot"] == ADDRESS
    assert data["product_snapshot"]["created_by_admin_id"] == "admin-1"
    assert [e["status"] for e in add_events.await_args.args[1]] == ["ordered", "shipping"]
    paid.assert_awaited_once()
    assert emit.await_args.kwargs["trigger_type"] == "gift_sent"
    assert "桂花" not in remember.call_args.kwargs["text"]
    assert result["notification"] == "sent"


@pytest.mark.asyncio
async def test_concurrent_duplicate_insert_discards_new_upload(monkeypatch):
    insert, add_events, emit, _, _ = _patch_create(monkeypatch, inserted=(_admin_gift(), False))
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", AsyncMock(return_value=True))
    monkeypatch.setattr(svc.activity_media_storage, "save_image_blob", Mock(return_value="user-1_gift_x.png"))
    deleted = Mock()
    monkeypatch.setattr(svc.activity_media_storage, "delete_media_file", deleted)
    png = base64.b64encode(b"\x89PNG fake").decode()

    result = await svc.create_manual_gift("admin-1", _create_request(image_base64=png, image_mime="image/png"))

    assert result["notification"] == "skipped"
    emit.assert_not_awaited()
    deleted.assert_called_once_with("user-1_gift_x.png")  # 抢输的那次上传不留孤儿文件


@pytest.mark.asyncio
async def test_replay_completes_interrupted_send_without_renotifying(monkeypatch):
    insert, add_events, emit, paid, remember = _patch_create(monkeypatch)
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift("shipping")))
    notice = AsyncMock(return_value=False)
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", notice)

    # 上次插入成功但通知前崩了：重放不再插入，补发通知与轨迹。
    result = await svc.create_manual_gift("admin-1", _create_request(gift_name="别的名字"))
    insert.assert_not_awaited()
    add_events.assert_awaited_once()
    emit.assert_awaited_once()
    assert result["notification"] == "sent"

    # 已经通知过：只补轨迹（按内容去重），不再打扰用户。
    emit.reset_mock()
    remember.reset_mock()
    notice.return_value = True
    result = await svc.create_manual_gift("admin-1", _create_request())
    emit.assert_not_awaited()
    remember.assert_not_called()
    assert result["notification"] == "skipped"

    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift(user_id="someone-else")))
    with pytest.raises(HTTPException) as conflict:
        await svc.create_manual_gift("admin-1", _create_request())
    assert conflict.value.status_code == 409


@pytest.mark.asyncio
async def test_manual_gift_requires_address_and_valid_request_id(monkeypatch):
    insert, *_ = _patch_create(monkeypatch, address=None)
    with pytest.raises(HTTPException) as no_address:
        await svc.create_manual_gift("admin-1", _create_request())
    assert no_address.value.status_code == 409
    insert.assert_not_awaited()

    with pytest.raises(HTTPException) as bad_id:
        await svc.create_manual_gift("admin-1", _create_request(request_id="x" * 32))
    assert bad_id.value.status_code == 422


def test_image_urls_are_scoped_to_the_recipient():
    assert svc._validate_image_url("user-1", "https://cbu01.alicdn.com/a.jpg") == "https://cbu01.alicdn.com/a.jpg"
    assert svc._validate_image_url("user-1", "/offline/media/user-1_image_a.jpg") == "/offline/media/user-1_image_a.jpg"
    assert svc._validate_image_url("user-1", "  ") is None
    for bad in ("/offline/media/user-2_image_a.jpg", "ftp://x/a.jpg", "javascript:alert(1)", "/offline/media/../x"):
        with pytest.raises(HTTPException):
            svc._validate_image_url("user-1", bad)


@pytest.mark.asyncio
async def test_update_only_touches_provided_fields_and_can_clear_image(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift(product_image_url="https://x/a.jpg")))
    update = AsyncMock(return_value=_admin_gift())
    monkeypatch.setattr(svc.admin_repo, "update_gift_details", update)
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))

    await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(carrier="", image_url=None))
    assert update.await_args.args[1] == {"logistics_provider": None, "product_image_url": None}

    await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(gift_note="新寄语"))
    assert update.await_args.args[1] == {"gift_note": "新寄语"}


# ── 物流录入 ──────────────────────────────────────────────────────────────


def _patch_tracking(monkeypatch, gift, events, *, inserted=True, notice_sent=False):
    """events 是插入/删除之后库里剩下的全部节点。"""
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=gift))
    monkeypatch.setattr(svc.admin_repo, "list_tracking_events", AsyncMock(return_value=events))
    insert = AsyncMock(return_value={"id": "new"} if inserted else None)
    monkeypatch.setattr(svc.admin_repo, "insert_tracking_event", insert)
    monkeypatch.setattr(svc.admin_repo, "delete_tracking_event", AsyncMock(return_value=True))
    set_status = AsyncMock(return_value=True)
    monkeypatch.setattr(svc.admin_repo, "set_transit_status", set_status)
    advance = AsyncMock(return_value=_admin_gift("out_for_delivery"))
    monkeypatch.setattr(svc.gift_service, "advance_to_out_for_delivery", advance)
    monkeypatch.setattr(svc.gift_repo, "gift_notice_sent", AsyncMock(return_value=notice_sent))
    notify = AsyncMock(return_value="msg-door")
    monkeypatch.setattr(svc.gift_service, "notify_out_for_delivery", notify)
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))
    monkeypatch.setattr(svc, "AdminGiftActionResponse", lambda **kw: kw)
    return insert, set_status, advance, notify


def _ev(status, occurred_at, created_at=None):
    return {**_event(status, occurred_at), "created_at": (created_at or occurred_at).isoformat()}


@pytest.mark.asyncio
async def test_first_out_for_delivery_event_moves_status_and_reminds_user(monkeypatch):
    insert, set_status, advance, notify = _patch_tracking(
        monkeypatch, _admin_gift("shipping"),
        [_ev("shipping", NOW - timedelta(days=1)), _ev("out_for_delivery", NOW)],
    )
    result = await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(
        status="out_for_delivery", title="快递员派送中", occurred_at=NOW,
    ))
    assert advance.await_args.kwargs["notify"] is False
    notify.assert_awaited_once()
    assert result["notification"] == "sent"
    set_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeat_out_for_delivery_does_not_remind_again(monkeypatch):
    _, _, advance, notify = _patch_tracking(
        monkeypatch, _admin_gift("shipping"),
        [_ev("out_for_delivery", NOW - timedelta(hours=3)), _ev("shipping", NOW - timedelta(hours=1)),
         _ev("out_for_delivery", NOW)],
        notice_sent=True,
    )
    result = await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(
        status="out_for_delivery", title="再次派送", occurred_at=NOW,
    ))
    advance.assert_awaited_once()
    notify.assert_not_awaited()
    assert result["notification"] == "skipped"


@pytest.mark.asyncio
async def test_backfilled_older_event_does_not_regress_status(monkeypatch):
    insert, set_status, advance, _ = _patch_tracking(
        monkeypatch, _admin_gift("shipping"),
        [_ev("ordered", NOW - timedelta(days=2), created_at=NOW), _ev("shipping", NOW, created_at=NOW - timedelta(hours=1))],
    )
    await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(
        status="ordered", title="补录下单", occurred_at=NOW - timedelta(days=2),
    ))
    insert.assert_awaited_once()
    set_status.assert_not_awaited()
    advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_minute_event_counts_as_latest(monkeypatch):
    # 后台时间选择器只精确到分钟：同一分钟内后录入的节点为准。
    _, set_status, _, _ = _patch_tracking(
        monkeypatch, _admin_gift("ordered"),
        [_ev("ordered", NOW.replace(second=40), created_at=NOW),
         _ev("shipping", NOW.replace(second=0), created_at=NOW + timedelta(seconds=5))],
    )
    await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(
        status="shipping", title="已发出", occurred_at=NOW.replace(second=0),
    ))
    set_status.assert_awaited_once_with("a" * 32, "shipping")


@pytest.mark.asyncio
async def test_deleting_mistaken_door_event_reverts_to_in_transit(monkeypatch):
    _, set_status, advance, notify = _patch_tracking(
        monkeypatch, _admin_gift("out_for_delivery"), [_ev("shipping", NOW - timedelta(hours=1))],
    )
    result = await svc.delete_tracking_event("admin-1", "g", "door-event")
    set_status.assert_awaited_once_with("a" * 32, "shipping")
    advance.assert_not_awaited()
    notify.assert_not_awaited()
    assert result["notification"] == "skipped"


@pytest.mark.asyncio
async def test_tracking_rejects_closed_gifts_and_duplicates(monkeypatch):
    insert, *_ = _patch_tracking(monkeypatch, _admin_gift("delivered"), [])
    with pytest.raises(HTTPException) as closed:
        await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(status="shipping", title="x"))
    assert closed.value.status_code == 409
    insert.assert_not_awaited()

    _patch_tracking(monkeypatch, _admin_gift("shipping"), [], inserted=False)
    with pytest.raises(HTTPException) as duplicate:
        await svc.add_tracking_event("admin-1", "g", AdminGiftTrackingRequest(status="shipping", title="x"))
    assert duplicate.value.status_code == 409


@pytest.mark.asyncio
async def test_admin_receipt_only_from_out_for_delivery(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift("shipping")))
    confirm = AsyncMock()
    monkeypatch.setattr(svc.gift_service, "confirm_receipt_as_admin", confirm)
    with pytest.raises(HTTPException) as early:
        await svc.confirm_receipt("admin-1", "g")
    assert early.value.status_code == 409
    confirm.assert_not_awaited()

    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift("out_for_delivery")))
    confirm.return_value = (_admin_gift("delivered"), "是桂花乌龙茶", False)
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))
    monkeypatch.setattr(svc, "AdminGiftActionResponse", lambda **kw: kw)
    result = await svc.confirm_receipt("admin-1", "g")
    confirm.assert_awaited_once_with("user-1", "g")
    assert result["notification"] == "sent"


@pytest.mark.asyncio
async def test_cancel_refuses_received_gifts(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value=_admin_gift("delivered")))
    monkeypatch.setattr(svc.admin_repo, "cancel_gift", AsyncMock(return_value=False))
    with pytest.raises(HTTPException) as exc:
        await svc.cancel_gift("admin-1", "g", "寄错了")
    assert exc.value.status_code == 409


# ── 鉴权 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method,path", [
    ("get", "/admin-api/gifts/users?q=a"),
    ("get", "/admin-api/gifts/users/user-1"),
    ("post", "/admin-api/gifts/users/user-1/recommendations"),
    ("get", "/admin-api/gifts"),
    ("post", "/admin-api/gifts"),
    ("get", "/admin-api/gifts/g"),
    ("patch", "/admin-api/gifts/g"),
    ("post", "/admin-api/gifts/g/tracking"),
    ("delete", "/admin-api/gifts/g/tracking/e"),
    ("post", "/admin-api/gifts/g/confirm-receipt"),
    ("post", "/admin-api/gifts/g/cancel"),
])
@pytest.mark.parametrize("role,status", [(None, 401), ("user", 403)])
def test_every_admin_gift_endpoint_requires_admin(api_client, auth_header, method, path, role, status):
    headers = auth_header("someone", role=role) if role else {}
    response = getattr(api_client, method)(path, headers=headers)
    assert response.status_code == status


def test_admin_create_uses_jwt_admin_as_operator(api_client, auth_header, monkeypatch):
    create = AsyncMock(return_value={"gift": {
        **_admin_gift("ordered"), "events": [], "messages": [],
    }, "notification": "sent"})
    monkeypatch.setattr(svc, "create_manual_gift", create)
    response = api_client.post(
        "/admin-api/gifts",
        json={"request_id": "3f1c2a7e-8c1b-4b8e-9d55-0a2f6e3b9c11", "user_id": "user-1", "gift_name": "茶"},
        headers=auth_header("admin-9", role="admin"),
    )
    assert response.status_code == 200, response.text
    assert create.await_args.args[0] == "admin-9"
    assert response.json()["gift"]["gift_name"] == "桂花乌龙茶"


def test_activity_cleanup_keeps_admin_uploaded_gift_images(tmp_path, monkeypatch):
    from app.services.offline import activity_media_storage as storage

    monkeypatch.setattr(storage, "_MEDIA_DIR", tmp_path)
    for name in ("u1_image_a.jpg", "u1_gift_b.jpg", "u2_gift_c.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert storage.delete_user_media_files("u1", keep_gift_images=True) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["u1_gift_b.jpg", "u2_gift_c.jpg"]
    assert storage.delete_user_media_files("u1") == 1  # 用户数据重置：全部删除
    key = storage.save_image_blob(user_id="u1", blob=b"img", mime="image/png", kind="gift")
    assert key.startswith("u1_gift_") and key.endswith(".png")
