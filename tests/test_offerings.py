from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import offerings
from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.prompting.defaults import (
    CHAT_RED_PACKET_REPLY_PROMPT,
    CHAT_RED_PACKET_USER_MESSAGE_PROMPT,
)
from app.services.prompting.registry import PROMPT_DEFINITIONS


class _FakeTx:
    def __init__(self, rows_by_query: list[list[dict]]):
        self.rows_by_query = list(rows_by_query)
        self.query_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []

    async def query_raw(self, query: str, *args):
        self.query_calls.append((query, args))
        return self.rows_by_query.pop(0)

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        return 1


class _TxContext:
    def __init__(self, tx: _FakeTx):
        self.tx = tx

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDb:
    def __init__(self, *, query_rows=None, tx_rows=None):
        self.query_rows = list(query_rows or [])
        self.query_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fake_tx = _FakeTx(tx_rows or [])

    async def query_raw(self, query: str, *args):
        self.query_calls.append((query, args))
        return self.query_rows.pop(0)

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        return 1

    def tx(self):
        return _TxContext(self.fake_tx)


class _FakeManager:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    async def send_event(self, conversation_id, event_type, payload):
        self.events.append((conversation_id, event_type, payload))


def _swallow_background(coro):
    coro.close()
    return None


def _offering_row(**overrides):
    meta = overrides.pop("metadata", {
        "offering_count": 1,
        "previous_summary": "",
        "agent_name": "小芜",
        "workspace_id": "ws-1",
    })
    if isinstance(meta, dict):
        meta = json.dumps(meta, ensure_ascii=False)
    row = {
        "id": "off-1",
        "user_id": "u1",
        "agent_id": "a1",
        "conversation_id": "c1",
        "message_id": None,
        "kind": "red_packet",
        "ticket_amount": 18,
        "agent_value_yuan": 18,
        "status": "sent",
        "blessing": None,
        "metadata": meta,
        "created_at": "2026-08-20T08:00:00+00:00",
        "received_at": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_send_red_packet_debits_and_builds_card(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[[{
            "id": "c1",
            "user_id": "u1",
            "agent_id": "a1",
            "workspace_id": "ws-1",
            "agent_name": "小芜",
        }]],
        tx_rows=[
            [],
            [],
            [{"n": 0}],
            [_offering_row()],
        ],
    )
    debit = AsyncMock(return_value={
        "ticket_balance": 82,
        "point_balance": 0,
        "achievement_points_synced": 0,
    })
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(offerings.wallet, "debit_tickets", debit)
    monkeypatch.setattr(offerings, "fire_background", _swallow_background)

    result = await offerings.send_red_packet(
        user_id="u1",
        conversation_id="c1",
        ticket_amount=18,
    )

    debit.assert_awaited_once()
    assert debit.await_args.kwargs["source"] == "red_packet"
    card = result["component_card"]
    assert card["type"] == "red_packet"
    assert card["payload"]["offering_id"] == "off-1"
    assert card["payload"]["ticket_amount"] == 18
    assert card["payload"]["status"] == "sent"
    assert card["subtitle"] == ""
    assert "¥" not in card["title"]
    assert result["wallet"]["ticket_balance"] == 82


@pytest.mark.asyncio
async def test_send_red_packet_insufficient_balance(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[[{
            "id": "c1",
            "user_id": "u1",
            "agent_id": "a1",
            "workspace_id": "ws-1",
            "agent_name": "小芜",
        }]],
        tx_rows=[[]],
    )
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock())

    async def _debit(*_args, **_kwargs):
        raise ValueError("insufficient_ticket_balance")

    monkeypatch.setattr(offerings.wallet, "debit_tickets", _debit)

    with pytest.raises(ValueError, match="insufficient_ticket_balance"):
        await offerings.send_red_packet(
            user_id="u1",
            conversation_id="c1",
            ticket_amount=18,
        )


@pytest.mark.asyncio
async def test_send_red_packet_rejects_invalid_amount():
    with pytest.raises(ValueError, match="invalid_amount"):
        await offerings.send_red_packet(
            user_id="u1",
            conversation_id="c1",
            ticket_amount=0,
        )


@pytest.mark.asyncio
async def test_authorize_replaces_client_card_and_rejects_replay(monkeypatch):
    fake_db = _FakeDb(query_rows=[[_offering_row()]])
    monkeypatch.setattr(offerings, "db", fake_db)

    card = await offerings.authorize_red_packet_card(
        {
            "type": "red_packet",
            "payload": {"offering_id": "off-1", "ticket_amount": 999},
        },
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
    )
    assert card["payload"]["ticket_amount"] == 18
    assert card["_offering"]["id"] == "off-1"

    fake_db.query_rows = [[_offering_row(message_id="msg-1")]]
    with pytest.raises(ValueError, match="offering_already_bound"):
        await offerings.authorize_red_packet_card(
            {"type": "red_packet", "payload": {"offering_id": "off-1"}},
            user_id="u1",
            agent_id="a1",
            conversation_id="c1",
        )


@pytest.mark.asyncio
async def test_authorize_rejects_foreign_user(monkeypatch):
    fake_db = _FakeDb(query_rows=[[_offering_row()]])
    monkeypatch.setattr(offerings, "db", fake_db)
    with pytest.raises(ValueError, match="offering_forbidden"):
        await offerings.authorize_red_packet_card(
            {"type": "red_packet", "payload": {"offering_id": "off-1"}},
            user_id="other",
            agent_id="a1",
            conversation_id="c1",
        )


@pytest.mark.asyncio
async def test_bind_red_packet_message(monkeypatch):
    fake_db = _FakeDb(query_rows=[[_offering_row(message_id="msg-9")]])
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings, "fire_background", _swallow_background)
    bound = await offerings.bind_red_packet_message(
        offering_id="off-1",
        message_id="msg-9",
        user_id="u1",
    )
    assert bound["message_id"] == "msg-9"
    sql = fake_db.query_calls[0][0]
    assert "WHERE id = $1 AND user_id = $3 AND message_id IS NULL" in sql
    assert "conversation_id = $4" in sql
    assert fake_db.query_calls[0][1][3] is None
    assert any("UPDATE messages" in sql for sql, _ in fake_db.execute_calls)
    assert any("元红包" in str(args) for _, args in fake_db.execute_calls)


@pytest.mark.asyncio
async def test_bind_rejects_mismatched_conversation(monkeypatch):
    fake_db = _FakeDb(query_rows=[[]])
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings, "fire_background", _swallow_background)
    with pytest.raises(ValueError, match="offering_already_bound"):
        await offerings.bind_offering_message(
            offering_id="off-1",
            message_id="msg-9",
            user_id="u1",
            conversation_id="c-other",
        )
    assert fake_db.query_calls[0][1][3] == "c-other"


@pytest.mark.asyncio
async def test_mark_received_credits_wallet_once(monkeypatch):
    received = _offering_row(
        status="received",
        message_id="msg-1",
        received_at="2026-08-20T08:01:00+00:00",
    )
    fake_db = _FakeDb(tx_rows=[
        [received],
        [{"id": "notice-1", "created_at": "2026-08-20T08:01:01+00:00"}],
    ])
    manager = _FakeManager()
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings, "fire_background", _swallow_background)
    monkeypatch.setattr(
        "app.services.runtime.ws_manager.manager",
        manager,
    )

    first = await offerings.mark_red_packet_received(
        offering_id="off-1",
        user_id="u1",
        conversation_id="c1",
    )
    assert first["offering"]["status"] == "received"
    assert first["component_card"]["subtitle"] == ""
    assert first["component_card"]["payload"]["status_label"] == "已领取"
    assert any("SET status" in sql for sql, _ in fake_db.fake_tx.query_calls)
    assert any("INSERT INTO agent_wallets" in sql for sql, _ in fake_db.fake_tx.execute_calls)
    assert manager.events[0][1] == "red_packet"
    assert manager.events[0][2]["notice"]["text"] == "小芜领取了你的红包"

    fake_db.fake_tx.rows_by_query = [[], [received]]
    fake_db.fake_tx.execute_calls.clear()
    fake_db.execute_calls.clear()
    manager.events.clear()
    second = await offerings.mark_red_packet_received(
        offering_id="off-1",
        user_id="u1",
        conversation_id="c1",
    )
    assert second["offering"]["status"] == "received"
    assert fake_db.fake_tx.execute_calls == []
    assert fake_db.execute_calls == []
    assert manager.events == []


def _offering_from_public():
    return offerings._offering_from_row(_offering_row())


@pytest.mark.asyncio
async def test_write_offering_memories_uses_gift_subcategory(monkeypatch):
    stored: list[dict] = []

    async def _store(_user_id, content, **kwargs):
        stored.append({"content": content, **kwargs})
        return "mem"

    monkeypatch.setattr(offerings, "store_memory", _store)
    await offerings._write_offering_memories(_offering_from_public())

    assert stored[0]["main_category"] == "生活"
    assert stored[0]["sub_category"] == "馈赠"
    assert stored[0]["source"] == "user"
    assert stored[0]["provenance"] == USER_STATED
    assert stored[0]["level"] == 2
    assert "18元" in stored[0]["content"]
    assert stored[0].get("skip_reconciliation") is True
    assert stored[1]["source"] == "ai"
    assert stored[1]["provenance"] == AI_AUTHORED
    assert "充值" not in stored[1]["content"]


@pytest.mark.asyncio
async def test_backfill_content_only_updates_messages(monkeypatch):
    row = _offering_row(message_id="msg-backfill", status="received")
    fake_db = _FakeDb(
        query_rows=[
            [row],
            [{"id": "msg-backfill"}],
        ],
    )
    monkeypatch.setattr(offerings, "db", fake_db)

    stats = await offerings.backfill_offering_memories_and_content(
        limit=10,
        content_only=True,
    )
    assert stats["scanned"] == 1
    assert stats["content_updated"] == 1
    assert stats["sent_memories"] == 0
    assert any("UPDATE messages" in sql for sql, _ in fake_db.query_calls)


@pytest.mark.asyncio
async def test_backfill_writes_missing_memories(monkeypatch):
    row = _offering_row(message_id="msg-mem", status="received")
    fake_db = _FakeDb(
        query_rows=[
            [row],
            [],
            [],
            [],
            [],
        ],
    )
    stored: list[str] = []

    async def _store(_uid, content, **kwargs):
        stored.append(content)
        return "mem-id"

    async def _missing(*args, **kwargs):
        return []

    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings, "store_memory", _store)
    monkeypatch.setattr(offerings, "_backfill_memory_if_missing", _missing)

    stats = await offerings.backfill_offering_memories_and_content(limit=10)
    assert stats["scanned"] == 1
    assert stats["content_updated"] == 0



@pytest.mark.asyncio
async def test_build_red_packet_user_message_uses_registry(monkeypatch):
    monkeypatch.setattr(
        offerings,
        "get_prompt_text_or_default",
        AsyncMock(return_value=CHAT_RED_PACKET_USER_MESSAGE_PROMPT),
    )
    text = await offerings.build_red_packet_user_message(_offering_from_public())
    assert "18" in text
    assert "充值" not in text
    assert "商城" not in text


def test_red_packet_prompts_are_registered_without_hardcoded_tiers():
    keys = {item.key for item in PROMPT_DEFINITIONS}
    assert "chat.red_packet_reply" in keys
    assert "chat.red_packet_user_message" in keys
    reply = CHAT_RED_PACKET_REPLY_PROMPT
    rewrite = CHAT_RED_PACKET_USER_MESSAGE_PROMPT
    for text in (reply, rewrite):
        assert "{ticket_amount}" in text
        assert "{agent_value_yuan}" in text
        assert "{offering_count}" in text
    assert "不要按档位念稿" in reply
    assert "不要报出数字" in reply
    assert "1-10" not in reply
    assert "小于10" not in reply
    assert "大于100" not in reply


def test_orchestrator_red_packet_forces_main_prompt_and_skips_memory():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app/services/chat/orchestrator.py").read_text()
    assert "skip_memory=bool(offering_context)" in src
    assert "meal_card_decision.state != \"none\" or bool(offering_context)" in src


@pytest.mark.asyncio
async def test_red_packet_prompt_skips_reengagement_section():
    """An 8-day gap must not inject 重逢感知 on a red-packet turn.

    2026-08-20: empty card bubble + reengagement_day made the model greet
    ('好久不见呀') and ignore the gift even though chat.red_packet_reply
    was present.
    """
    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    async def _prompt_text(key: str, **_kwargs) -> str:
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    diagnostics: dict = {}
    with (
        patch(
            "app.services.chat.prompt_builder.get_prompt_text",
            AsyncMock(side_effect=_prompt_text),
        ),
        patch(
            "app.services.chat.prompt_builder.get_prompt_text_or_default",
            AsyncMock(side_effect=_prompt_text),
        ),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="小伴", values={"gender": "female"}),
            reengagement_gap_seconds=8 * 86400,
            session_recap="上次在聊西甲票务",
            red_packet_context={
                "offering_id": "off-1",
                "ticket_amount": 100,
                "agent_value_yuan": 100,
                "offering_count": 1,
                "previous_summary": "",
                "blessing": "",
            },
            diagnostics=diagnostics,
        )

    assert "## 红包回应" in prompt
    assert "100" in prompt
    assert "## 重逢感知" not in prompt
    assert "没说话了" not in prompt
    assert "## 上次聊到" not in prompt
    assert "重逢感知" in diagnostics["empty_prompt_sections_removed"]
    assert "上次聊到" in diagnostics["empty_prompt_sections_removed"]


def _gift_offering_row(**overrides):
    meta = overrides.pop("metadata", {
        "offering_count": 1,
        "previous_summary": "",
        "agent_name": "小芜",
        "workspace_id": "ws-1",
        "product_kind": "gift_1",
        "product_title": "美式咖啡",
        "product_subcategory": "饮品",
        "product_asset_key": "1",
    })
    return _offering_row(
        kind="gift",
        ticket_amount=25,
        agent_value_yuan=25,
        metadata=meta,
        **overrides,
    )


@pytest.mark.asyncio
async def test_send_gift_consumes_inventory_and_builds_card(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[[{
            "id": "c1",
            "user_id": "u1",
            "agent_id": "a1",
            "workspace_id": "ws-1",
            "agent_name": "小芜",
        }]],
        tx_rows=[
            [],
            [],
            [{"n": 0}],
            [_gift_offering_row()],
        ],
    )
    consume = AsyncMock(return_value={
        "product_kind": "gift_1",
        "quantity": 0,
        "acquired_at": "2026-08-20T08:00:00+00:00",
        "updated_at": "2026-08-20T08:00:00+00:00",
    })
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock(return_value={
        "ticket_balance": 10,
        "point_balance": 80,
        "achievement_points_synced": 0,
    }))
    monkeypatch.setattr(offerings, "consume_inventory", consume)
    monkeypatch.setattr(offerings, "fire_background", _swallow_background)

    result = await offerings.send_gift(
        user_id="u1",
        conversation_id="c1",
        product_kind="gift_1",
    )

    consume.assert_awaited_once()
    assert consume.await_args.args[:2] == ("u1", "gift_1")
    assert consume.await_args.kwargs["client"] is fake_db.fake_tx
    assert consume.await_args.kwargs["quantity"] == 1
    card = result["component_card"]
    assert card["type"] == "gift"
    assert card["title"] == "美式咖啡"
    assert card["subtitle"] == ""
    assert card["payload"]["product_kind"] == "gift_1"
    assert card["payload"]["status"] == "sent"
    assert result["inventory_item"]["quantity"] == 0


@pytest.mark.asyncio
async def test_send_gift_rejects_non_gift_and_missing_inventory(monkeypatch):
    with pytest.raises(ValueError, match="not_giftable"):
        await offerings.send_gift(
            user_id="u1",
            conversation_id="c1",
            product_kind="outfit_checkin",
        )

    fake_db = _FakeDb(
        query_rows=[[{
            "id": "c1",
            "user_id": "u1",
            "agent_id": "a1",
            "workspace_id": "ws-1",
            "agent_name": "小芜",
        }]],
        tx_rows=[[]],
    )

    async def _empty(*args, **kwargs):
        raise ValueError("insufficient_inventory")

    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(offerings, "consume_inventory", _empty)
    with pytest.raises(ValueError, match="insufficient_inventory"):
        await offerings.send_gift(
            user_id="u1",
            conversation_id="c1",
            product_kind="gift_1",
        )


@pytest.mark.asyncio
async def test_authorize_gift_card_rejects_wrong_kind(monkeypatch):
    fake_db = _FakeDb(query_rows=[[_offering_row()]])
    monkeypatch.setattr(offerings, "db", fake_db)
    with pytest.raises(ValueError, match="offering_forbidden"):
        await offerings.authorize_gift_card(
            {"type": "gift", "payload": {"offering_id": "off-1"}},
            user_id="u1",
            agent_id="a1",
            conversation_id="c1",
        )


@pytest.mark.asyncio
async def test_mark_gift_received_skips_agent_wallet(monkeypatch):
    received = _gift_offering_row(
        status="received",
        message_id="msg-1",
        received_at="2026-08-21T08:00:00+00:00",
    )
    fake_db = _FakeDb(tx_rows=[
        [received],
        [{"id": "notice-g", "created_at": "2026-08-21T08:00:01+00:00"}],
    ])
    monkeypatch.setattr(offerings, "db", fake_db)
    manager = _FakeManager()
    monkeypatch.setattr(
        "app.services.runtime.ws_manager.manager",
        manager,
        raising=False,
    )

    result = await offerings.mark_gift_received(
        offering_id="off-1",
        user_id="u1",
        conversation_id="c1",
    )

    assert result["offering"]["status"] == "received"
    assert result["component_card"]["subtitle"] == ""
    assert result["component_card"]["payload"]["status_label"] == "已接收"
    assert manager.events[0][1] == "gift"
    assert manager.events[0][2]["notice"]["text"] == "小芜收下了你的礼物"
    assert not fake_db.fake_tx.execute_calls


@pytest.mark.asyncio
async def test_build_gift_user_message_uses_registry(monkeypatch):
    from app.services.prompting.defaults import CHAT_GIFT_USER_MESSAGE_PROMPT

    monkeypatch.setattr(
        offerings,
        "get_prompt_text_or_default",
        AsyncMock(return_value=CHAT_GIFT_USER_MESSAGE_PROMPT),
    )
    text = await offerings.build_gift_user_message(
        offerings._offering_from_row(_gift_offering_row()),
    )
    assert "美式咖啡" in text
    assert "25" in text
    assert "充值" not in text
    assert "商城" not in text


def test_gift_prompts_are_registered_without_hardcoded_tiers():
    from app.services.prompting.defaults import (
        CHAT_GIFT_REPLY_PROMPT,
        CHAT_GIFT_USER_MESSAGE_PROMPT,
    )

    keys = {item.key for item in PROMPT_DEFINITIONS}
    assert "chat.gift_reply" in keys
    assert "chat.gift_user_message" in keys
    reply = CHAT_GIFT_REPLY_PROMPT
    rewrite = CHAT_GIFT_USER_MESSAGE_PROMPT
    for text in (reply, rewrite):
        assert "{product_title}" in text
        assert "{agent_value_yuan}" in text
        assert "{offering_count}" in text
    assert "不要按档位念稿" in reply
    assert "不要报出价格" in reply
    assert "1-10" not in reply
    assert "小于10" not in reply


@pytest.mark.asyncio
async def test_gift_prompt_skips_reengagement_section():
    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    async def _prompt_text(key: str, **_kwargs) -> str:
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    diagnostics: dict = {}
    with (
        patch(
            "app.services.chat.prompt_builder.get_prompt_text",
            AsyncMock(side_effect=_prompt_text),
        ),
        patch(
            "app.services.chat.prompt_builder.get_prompt_text_or_default",
            AsyncMock(side_effect=_prompt_text),
        ),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="小伴", values={"gender": "female"}),
            reengagement_gap_seconds=8 * 86400,
            session_recap="上次在聊西甲票务",
            gift_context={
                "offering_id": "off-1",
                "product_title": "美式咖啡",
                "product_subcategory": "饮品",
                "agent_value_yuan": 25,
                "offering_count": 1,
                "previous_summary": "",
            },
            diagnostics=diagnostics,
        )

    assert "## 礼物回应" in prompt
    assert "美式咖啡" in prompt
    assert "## 重逢感知" not in prompt
    assert "## 上次聊到" not in prompt


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conv_row():
    return {
        "id": "c1",
        "user_id": "u1",
        "agent_id": "a1",
        "workspace_id": "ws-1",
        "agent_name": "小芜",
    }


@pytest.mark.asyncio
async def test_send_gift_reuses_fresh_unbound_without_consuming(monkeypatch):
    unbound = _gift_offering_row(created_at=_now_iso())
    fake_db = _FakeDb(
        query_rows=[[_conv_row()]],
        tx_rows=[
            [unbound],
            [{
                "product_kind": "gift_1",
                "quantity": 0,
                "acquired_at": None,
                "updated_at": None,
            }],
        ],
    )
    consume = AsyncMock()
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock(return_value={
        "ticket_balance": 10,
        "point_balance": 80,
        "achievement_points_synced": 0,
    }))
    monkeypatch.setattr(offerings, "consume_inventory", consume)

    result = await offerings.send_gift(
        user_id="u1",
        conversation_id="c1",
        product_kind="gift_1",
    )

    consume.assert_not_awaited()
    assert result["offering"]["id"] == "off-1"
    assert result["inventory_item"]["quantity"] == 0


@pytest.mark.asyncio
async def test_send_red_packet_reuses_fresh_unbound_without_debiting(monkeypatch):
    unbound = _offering_row(created_at=_now_iso())
    fake_db = _FakeDb(
        query_rows=[[_conv_row()]],
        tx_rows=[[unbound]],
    )
    debit = AsyncMock()
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock(return_value={
        "ticket_balance": 0,
        "point_balance": 0,
        "achievement_points_synced": 0,
    }))
    monkeypatch.setattr(offerings.wallet, "debit_tickets", debit)

    result = await offerings.send_red_packet(
        user_id="u1",
        conversation_id="c1",
        ticket_amount=18,
    )

    debit.assert_not_awaited()
    assert result["offering"]["id"] == "off-1"
    assert result["wallet"]["ticket_balance"] == 0


@pytest.mark.asyncio
async def test_send_gift_reclaims_stale_unbound_then_consumes(monkeypatch):
    stale = _gift_offering_row(created_at="2026-08-01T00:00:00+00:00")
    created = _gift_offering_row(id="off-new")
    fake_db = _FakeDb(
        query_rows=[[_conv_row()]],
        tx_rows=[
            [stale],
            [{"id": "off-1"}],
            [],
            [{"n": 0}],
            [created],
        ],
    )
    consume = AsyncMock(return_value={
        "product_kind": "gift_1",
        "quantity": 0,
        "acquired_at": None,
        "updated_at": None,
    })
    restore = AsyncMock(return_value={"product_kind": "gift_1", "quantity": 1})
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock(return_value={
        "ticket_balance": 10,
        "point_balance": 80,
        "achievement_points_synced": 0,
    }))
    monkeypatch.setattr(offerings, "consume_inventory", consume)
    monkeypatch.setattr(offerings, "add_inventory", restore)

    result = await offerings.send_gift(
        user_id="u1",
        conversation_id="c1",
        product_kind="gift_1",
    )

    restore.assert_awaited_once()
    consume.assert_awaited_once()
    assert result["offering"]["id"] == "off-new"


@pytest.mark.asyncio
async def test_send_gift_retargets_unbound_from_other_conversation(monkeypatch):
    unbound = _gift_offering_row(
        created_at=_now_iso(),
        conversation_id="c-other",
        agent_id="a-other",
    )
    retargeted = _gift_offering_row(created_at=_now_iso())
    fake_db = _FakeDb(
        query_rows=[[_conv_row()]],
        tx_rows=[
            [unbound],
            [retargeted],
            [{
                "product_kind": "gift_1",
                "quantity": 0,
                "acquired_at": None,
                "updated_at": None,
            }],
        ],
    )
    consume = AsyncMock()
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock(return_value={
        "ticket_balance": 10,
        "point_balance": 80,
        "achievement_points_synced": 0,
    }))
    monkeypatch.setattr(offerings, "consume_inventory", consume)

    result = await offerings.send_gift(
        user_id="u1",
        conversation_id="c1",
        product_kind="gift_1",
    )

    consume.assert_not_awaited()
    assert result["offering"]["conversation_id"] == "c1"
    assert "UPDATE user_offerings" in fake_db.fake_tx.query_calls[1][0]


@pytest.mark.asyncio
async def test_send_red_packet_reclaims_other_amount_then_debits(monkeypatch):
    leftover = _offering_row(created_at=_now_iso())
    created = _offering_row(id="off-new", ticket_amount=10, agent_value_yuan=10)
    fake_db = _FakeDb(
        query_rows=[[_conv_row()]],
        tx_rows=[
            [leftover],
            [{"id": "off-1"}],
            [],
            [{"n": 0}],
            [created],
        ],
    )
    debit = AsyncMock(return_value={
        "ticket_balance": 90,
        "point_balance": 0,
        "achievement_points_synced": 0,
    })
    credit = AsyncMock()
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(offerings.wallet, "debit_tickets", debit)
    monkeypatch.setattr(offerings.wallet, "credit_tickets", credit)

    result = await offerings.send_red_packet(
        user_id="u1",
        conversation_id="c1",
        ticket_amount=10,
    )

    credit.assert_awaited_once()
    assert credit.await_args.kwargs["source"] == "red_packet_unbound_refund"
    debit.assert_awaited_once()
    assert result["offering"]["id"] == "off-new"
    assert result["offering"]["ticket_amount"] == 10


@pytest.mark.asyncio
async def test_reclaim_stale_unbound_offerings_restores_gift(monkeypatch):
    stale = _gift_offering_row(created_at="2026-08-01T00:00:00+00:00")
    fake_db = _FakeDb(tx_rows=[[stale], [{"id": "off-1"}]])
    restore = AsyncMock(return_value={"product_kind": "gift_1", "quantity": 1})
    monkeypatch.setattr(offerings, "db", fake_db)
    monkeypatch.setattr(offerings, "add_inventory", restore)

    count = await offerings.reclaim_stale_unbound_offerings()

    assert count == 1
    restore.assert_awaited_once()
    assert restore.await_args.args[:2] == ("u1", "gift_1")


def test_is_offering_received_notice():
    from app.services.chat.message_utils import (
        _previous_assistant_message,
        is_offering_received_notice,
    )

    assert is_offering_received_notice({"offering_received": True})
    assert not is_offering_received_notice({"music_status": "started"})
    assert not is_offering_received_notice(None)

    notice = SimpleNamespace(
        id="n1",
        role="assistant",
        metadata={"offering_received": True},
    )
    real = SimpleNamespace(id="a1", role="assistant", metadata={})
    current = SimpleNamespace(id="u1", role="user", metadata={})
    assert _previous_assistant_message([real, notice, current], "u1") is real


def test_unbound_ttl_treats_old_rows_as_stale():
    now = datetime.now(timezone.utc)
    fresh = {"created_at": now.isoformat()}
    stale = {"created_at": (now - timedelta(seconds=121)).isoformat()}
    assert offerings._is_stale_unbound(fresh, now) is False
    assert offerings._is_stale_unbound(stale, now) is True
