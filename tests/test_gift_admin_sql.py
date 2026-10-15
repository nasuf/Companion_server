"""礼物状态迁移与后台礼物管理的真实 SQL 验证（隔离 PostgreSQL，绝不指向生产）。"""

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Json, Prisma

from app.services.offline import gift_admin_repository as admin_repo
from app.services.offline import gift_repository as gift_repo


@pytest.fixture
async def sql(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Requires isolated PostgreSQL via PROACTIVE_E2E_DATABASE_URL")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    db = Prisma(datasource={"url": url}, http={"trust_env": False})
    await db.connect()
    monkeypatch.setattr(gift_repo, "db", db)
    monkeypatch.setattr(admin_repo, "db", db)
    tag = uuid4().hex[:8]
    users, agents, spaces, convs = [], [], [], []
    try:
        for i in range(2):
            user = await db.user.create(data={"username": f"gift-admin-{tag}-{i}", "displayName": f"显示名{i}"})
            users.append(user.id)
            agent = await db.aiagent.create(data={"userId": user.id, "name": f"伙伴{i}"})
            agents.append(agent.id)
            space = await db.chatworkspace.create(data={"userId": user.id, "agentId": agent.id})
            spaces.append(space.id)
            conv = await db.conversation.create(data={"userId": user.id, "agentId": agent.id, "workspaceId": space.id})
            convs.append(conv.id)
        await db.authidentity.create(data={
            "userId": users[0], "provider": "wechat", "providerAccountId": f"wx-{tag}",
            "rawProfile": Json({"nickname": "微信昵称"}),
        })
        yield {"db": db, "users": users, "agents": agents, "spaces": spaces, "convs": convs, "tag": tag}
    finally:
        await db.realworldgift.delete_many(where={"userId": {"in": users}})
        await db.usermemory.delete_many(where={"userId": {"in": users}})
        await db.message.delete_many(where={"conversationId": {"in": convs}})
        await db.conversation.delete_many(where={"id": {"in": convs}})
        await db.chatworkspace.delete_many(where={"id": {"in": spaces}})
        await db.aiagent.delete_many(where={"id": {"in": agents}})
        await db.authidentity.delete_many(where={"userId": {"in": users}})
        await db.user.delete_many(where={"id": {"in": users}})
        await db.disconnect()


async def _manual_gift(sql, *, index=0, status="ordered", name="桂花乌龙茶", tracking="SF1"):
    gift, created = await admin_repo.insert_manual_gift({
        "id": uuid4().hex,
        "user_id": sql["users"][index],
        "agent_id": sql["agents"][index],
        "workspace_id": sql["spaces"][index],
        "conversation_id": sql["convs"][index],
        "status": status,
        "gift_name": name,
        "gift_reason": "内部理由",
        "gift_note": "寄语",
        "product_image_url": None,
        "product_snapshot": {"recommendation": {"reason": "r"}},
        "paid_amount_cents": 5800,
        "tracking_number": tracking,
        "logistics_provider": "顺丰",
        "address_snapshot": {"city": "上海"},
    })
    assert created and gift
    return gift


@pytest.mark.asyncio
async def test_manual_insert_is_idempotent_and_owner_scoped(sql):
    gift = await _manual_gift(sql, status="shipping")
    assert gift["provider"] == "manual"
    assert gift["trigger_type"] == "admin_manual"
    assert gift["shipped_at"] is not None
    assert gift["paid_amount_cents"] == 5800

    replay, created = await admin_repo.insert_manual_gift({
        "id": gift["id"], "user_id": sql["users"][0], "agent_id": sql["agents"][0],
        "status": "ordered", "gift_name": "别的名字",
    })
    assert created is False and replay["gift_name"] == "桂花乌龙茶"

    foreign, created = await admin_repo.insert_manual_gift({
        "id": gift["id"], "user_id": sql["users"][1], "agent_id": sql["agents"][1],
        "status": "ordered", "gift_name": "x",
    })
    assert foreign is None and created is False

    detail = await admin_repo.get_gift(gift["id"])
    assert detail["nickname"] == "微信昵称"
    assert detail["agent_name"] == "伙伴0"
    assert detail["recommendation"] == {"reason": "r"}
    assert detail["address_snapshot"] == {"city": "上海"}


@pytest.mark.asyncio
async def test_out_for_delivery_and_receipt_are_single_shot_cas(sql):
    user = sql["users"][0]
    gift = await _manual_gift(sql, status="shipping")

    assert await gift_repo.mark_gift_received(gift["id"], user) is None  # 运输中不能签收
    moved = await gift_repo.mark_gift_out_for_delivery(gift["id"], user)
    assert moved["status"] == "out_for_delivery"
    assert await gift_repo.mark_gift_out_for_delivery(gift["id"], user) is None
    assert await gift_repo.mark_gift_out_for_delivery(gift["id"], sql["users"][1]) is None

    received = await gift_repo.mark_gift_received(gift["id"], user)
    assert received["status"] == "delivered" and received["delivered_at"]
    assert await gift_repo.mark_gift_received(gift["id"], user) is None
    # 已签收后物流同步不能改状态。
    await gift_repo.update_tracking_snapshot(gift["id"], user, status="shipping")
    assert (await gift_repo.get_gift(gift["id"], user))["status"] == "delivered"
    assert await gift_repo.historical_gift_spend_cents(user) == 5800


@pytest.mark.asyncio
async def test_tracking_snapshot_only_advances_ordered_to_shipping(sql):
    user = sql["users"][0]
    gift = await _manual_gift(sql, status="ordered")
    await gift_repo.update_tracking_snapshot(gift["id"], user, status="shipping", tracking_number="T2")
    after = await gift_repo.get_gift(gift["id"], user)
    assert after["status"] == "shipping" and after["tracking_number"] == "T2"
    await gift_repo.mark_gift_out_for_delivery(gift["id"], user)
    await gift_repo.update_tracking_snapshot(gift["id"], user, status="shipping")
    assert (await gift_repo.get_gift(gift["id"], user))["status"] == "out_for_delivery"
    assert await gift_repo.historical_gift_spend_cents(user) == 5800


@pytest.mark.asyncio
async def test_tracking_insert_dedupes_and_sign_record_is_undeletable(sql):
    gift = await _manual_gift(sql)
    when = datetime.now(timezone.utc) - timedelta(hours=1)
    event = {"status": "shipping", "title": "已发出", "description": None, "location": "上海", "occurred_at": when}
    first = await admin_repo.insert_tracking_event(gift["id"], event)
    assert first and first["status"] == "shipping"
    assert await admin_repo.insert_tracking_event(gift["id"], event) is None
    await gift_repo.add_tracking_events(gift["id"], [{
        "status": "delivered", "title": "已签收", "description": "你已确认", "location": None, "occurred_at": when,
    }])
    events = await admin_repo.list_tracking_events(gift["id"])
    signed = next(e for e in events if e["status"] == "delivered")
    assert await admin_repo.delete_tracking_event(gift["id"], signed["id"]) is False
    assert await admin_repo.delete_tracking_event(gift["id"], first["id"]) is True
    assert [e["status"] for e in await admin_repo.list_tracking_events(gift["id"])] == ["delivered"]
    assert (await admin_repo.get_gift(gift["id"]))["last_event_title"] == "已签收"


@pytest.mark.asyncio
async def test_list_filters_search_and_totals(sql):
    tag = sql["tag"]  # 共享的隔离库里可能有别的数据，搜索词都带上本次的 tag
    ordered = await _manual_gift(sql, status="ordered", name=f"滤杯{tag}", tracking=f"YT{tag}")
    door = await _manual_gift(sql, status="shipping", name="暖手宝", tracking=f"SF{tag}")
    await gift_repo.mark_gift_out_for_delivery(door["id"], sql["users"][0])
    done = await _manual_gift(sql, status="shipping", name="帆布手账本", tracking="JD999")
    await gift_repo.mark_gift_out_for_delivery(done["id"], sql["users"][0])
    await gift_repo.mark_gift_received(done["id"], sql["users"][0])
    other = await _manual_gift(sql, index=1, status="ordered", name="香薰蜡烛", tracking="EMS1")
    await admin_repo.cancel_gift(other["id"], "寄错了")

    scoped = {"user_id": sql["users"][0]}
    in_transit, total = await admin_repo.list_gifts(status_filter="in_transit", **scoped)
    assert total == 2 and {g["id"] for g in in_transit} == {ordered["id"], door["id"]}
    assert [g["id"] for g in (await admin_repo.list_gifts(status_filter="out_for_delivery", **scoped))[0]] == [door["id"]]
    assert [g["id"] for g in (await admin_repo.list_gifts(status_filter="delivered", **scoped))[0]] == [done["id"]]
    closed, _ = await admin_repo.list_gifts(status_filter="closed", user_id=sql["users"][1])
    assert [g["id"] for g in closed] == [other["id"]] and closed[0]["failure_reason"] == "寄错了"

    assert [g["id"] for g in (await admin_repo.list_gifts(query=f"SF{tag}"))[0]] == [door["id"]]
    assert [g["id"] for g in (await admin_repo.list_gifts(query=f"滤杯{tag}"))[0]] == [ordered["id"]]
    assert {g["id"] for g in (await admin_repo.list_gifts(query=f"gift-admin-{sql['tag']}-1"))[0]} == {other["id"]}
    assert [g["id"] for g in (await admin_repo.list_gifts(query=done["id"]))[0]] == [done["id"]]

    page, total = await admin_repo.list_gifts(limit=1, offset=1, **scoped)
    assert total == 3 and len(page) == 1


@pytest.mark.asyncio
async def test_details_update_status_corrections_and_cancel_rules(sql):
    gift = await _manual_gift(sql, status="ordered")
    updated = await admin_repo.update_gift_details(gift["id"], {
        "gift_note": "新寄语", "logistics_provider": None, "paid_amount_cents": 100,
    })
    assert updated["gift_note"] == "新寄语" and updated["carrier"] is None and updated["paid_amount_cents"] == 100
    assert updated["gift_name"] == "桂花乌龙茶"

    assert await admin_repo.set_transit_status(gift["id"], "shipping") is True
    assert (await admin_repo.get_gift(gift["id"]))["shipped_at"] is not None
    await gift_repo.mark_gift_out_for_delivery(gift["id"], sql["users"][0])
    assert await admin_repo.set_transit_status(gift["id"], "shipping") is True  # 纠错回退
    await gift_repo.mark_gift_out_for_delivery(gift["id"], sql["users"][0])
    await gift_repo.mark_gift_received(gift["id"], sql["users"][0])
    assert await admin_repo.set_transit_status(gift["id"], "shipping") is False
    assert await admin_repo.cancel_gift(gift["id"], "x") is False

    other = await _manual_gift(sql)
    assert await admin_repo.cancel_gift(other["id"], "") is True
    assert await admin_repo.update_gift_details(other["id"], {"gift_note": "y"}) is None
    # 撤销后，进行中的自动下单流程也不能把它改回在途。
    assert await gift_repo.update_gift_order_details(other["id"], sql["users"][0], {"status": "shipping"}) is None
    assert await gift_repo.update_gift_status(other["id"], sql["users"][0], "failed") is None
    assert (await admin_repo.get_gift(other["id"]))["status"] == "cancelled"


@pytest.mark.asyncio
async def test_user_context_queries(sql):
    db, user, space = sql["db"], sql["users"][0], sql["spaces"][0]
    gift = await _manual_gift(sql)
    for metadata in (
        {"real_world_type": "gift", "source_id": gift["id"], "trigger_type": "gift_sent", "trace_id": "t1"},
        {"real_world_type": "gift", "source_id": "other"},
        {"real_world_type": "activity", "source_id": gift["id"]},
    ):
        await db.message.create(data={
            "conversationId": sql["convs"][0], "role": "assistant", "content": "有惊喜在路上", "metadata": Json(metadata),
        })
    messages = await admin_repo.gift_messages(gift["id"], user)
    assert [(m["trigger_type"], m["has_trace"]) for m in messages] == [("gift_sent", True)]
    assert await admin_repo.gift_messages(gift["id"], sql["users"][1]) == []
    assert await gift_repo.gift_notice_sent(gift["id"], user, "gift_sent") is True
    assert await gift_repo.gift_notice_sent(gift["id"], user, "gift_out_for_delivery") is False
    assert await gift_repo.gift_notice_sent(gift["id"], sql["users"][1], "gift_sent") is False

    for content, sub, archived in (("喜欢花香的茶", "爱好", False), ("周五提醒交报告", "提醒", False), ("旧事", "爱好", True)):
        await db.usermemory.create(data={
            "userId": user, "workspaceId": space, "mainCategory": "生活", "subCategory": sub,
            "content": content, "isArchived": archived,
        })
    memories = await admin_repo.user_memories_for_gift(user, space)
    assert [m["content"] for m in memories] == ["喜欢花香的茶"]
    assert memories[0]["category"] == "生活 / 爱好"
    assert await admin_repo.count_user_memories(user, space) == 2

    workspaces = await admin_repo.list_user_workspaces(user)
    assert workspaces == [{
        "workspace_id": space, "agent_id": sql["agents"][0], "agent_name": "伙伴0", "conversation_id": sql["convs"][0],
    }]
    assert (await admin_repo.get_user(user))["nickname"] == "微信昵称"
    assert await admin_repo.gift_names_for_user(user) == ["桂花乌龙茶"]
