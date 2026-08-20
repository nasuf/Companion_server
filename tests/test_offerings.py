from __future__ import annotations

import json
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
    assert card["subtitle"] == "待领取"
    assert "¥" not in card["title"]
    assert result["wallet"]["ticket_balance"] == 82


@pytest.mark.asyncio
async def test_send_red_packet_insufficient_balance(monkeypatch):
    fake_db = _FakeDb(query_rows=[[{
        "id": "c1",
        "user_id": "u1",
        "agent_id": "a1",
        "workspace_id": "ws-1",
        "agent_name": "小芜",
    }]])
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
    bound = await offerings.bind_red_packet_message(
        offering_id="off-1",
        message_id="msg-9",
        user_id="u1",
    )
    assert bound["message_id"] == "msg-9"
    assert "WHERE id = $1 AND user_id = $3 AND message_id IS NULL" in fake_db.query_calls[0][0]


@pytest.mark.asyncio
async def test_mark_received_credits_wallet_once(monkeypatch):
    received = _offering_row(
        status="received",
        message_id="msg-1",
        received_at="2026-08-20T08:01:00+00:00",
    )
    fake_db = _FakeDb(tx_rows=[[received]])
    manager = _FakeManager()
    monkeypatch.setattr(offerings, "db", fake_db)
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
    assert first["component_card"]["subtitle"] == "已领取"
    assert any("SET status" in sql for sql, _ in fake_db.fake_tx.query_calls)
    assert any("INSERT INTO agent_wallets" in sql for sql, _ in fake_db.fake_tx.execute_calls)
    assert manager.events[0][1] == "red_packet"

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
    assert "18钞票" in stored[0]["content"].replace(" ", "")
    assert stored[1]["source"] == "ai"
    assert stored[1]["provenance"] == AI_AUTHORED
    assert "人民币" not in stored[0]["content"]
    assert "充值" not in stored[1]["content"]


def _offering_from_public():
    return offerings._offering_from_row(_offering_row())


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
    assert "skip_memory=bool(red_packet_context)" in src
    assert "meal_card_decision.state != \"none\" or bool(red_packet_context)" in src


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
