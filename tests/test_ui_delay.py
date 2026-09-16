"""Tests for delayed-reply UI timing helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.api.realtime import ui_delay as ui_delay_mod
from app.services.interaction.user_turn_aggregation import UserMessageAggregationPlan


def _plan(route: str, *, context: dict | None = None) -> UserMessageAggregationPlan:
    return UserMessageAggregationPlan(
        route=route,
        agent_id="a1",
        user_id="u1",
        conversation_id="c1",
        text="hi",
        metadata={},
        final_message="hi",
        final_context=context,
        fallback_message="hi",
        fallback_context=context,
    )


@pytest.fixture
def reply_delay_enabled(monkeypatch):
    monkeypatch.setattr(
        "app.services.interaction.chat_management.reply_delay_enabled",
        lambda: True,
    )


def test_compute_ack_ui_timing_defers_during_aggregation():
    plan = _plan("fragment_window", context={"delay_seconds": 3.0})
    timing = ui_delay_mod.compute_ack_ui_timing(plan=plan)
    assert timing == {"ui_delay_seconds": 0.0, "defer_ui": True}


def test_compute_ack_ui_timing_returns_remaining_delay(reply_delay_enabled, monkeypatch):
    fixed_now = datetime(2026, 9, 16, 14, 0, 3, tzinfo=timezone.utc)
    received_at = datetime(2026, 9, 16, 14, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(ui_delay_mod, "_now_corrected", lambda: fixed_now)

    timing = ui_delay_mod.compute_ack_ui_timing(
        reply_context={"delay_seconds": 5.0, "received_at": received_at.isoformat()},
    )
    assert timing["defer_ui"] is False
    assert timing["ui_delay_seconds"] == pytest.approx(2.0)


def test_compute_ack_ui_timing_disabled_delay(reply_delay_enabled, monkeypatch):
    monkeypatch.setattr(
        "app.services.interaction.chat_management.reply_delay_enabled",
        lambda: False,
    )
    timing = ui_delay_mod.compute_ack_ui_timing(
        reply_context={"delay_seconds": 8.0, "received_at": "2026-09-16T14:00:00+00:00"},
    )
    assert timing == {"ui_delay_seconds": 0.0, "defer_ui": False}


@pytest.mark.asyncio
async def test_send_processing_event_payload():
    sent = []

    async def _send(payload):
        sent.append(payload)

    received_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await ui_delay_mod.send_processing_event(
        _send,
        conversation_id="conv-1",
        message_id="msg-1",
        reply_context={"delay_seconds": 4.0, "received_at": received_at},
    )

    assert len(sent) == 1
    assert sent[0]["type"] == "processing"
    data = sent[0]["data"]
    assert data["conversation_id"] == "conv-1"
    assert data["message_id"] == "msg-1"
    assert 0.0 <= data["ui_delay_seconds"] <= 4.0
