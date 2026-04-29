"""Tests for reply timing context service."""

from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.services.interaction.reply_context import (
    actual_delay_seconds,
    compute_delay_profile,
    merge_reply_contexts,
)


@pytest.fixture
def reply_delay_enabled(monkeypatch):
    """compute_delay_profile 在 settings.reply_delay_enabled=False 时短路返 0,
    spec §6.2 三档延迟测试需要先把 flag 打开."""
    monkeypatch.setattr(settings, "reply_delay_enabled", True)


def test_compute_delay_profile_conversation_mode(reply_delay_enabled):
    now = datetime.now(timezone.utc)
    profile = compute_delay_profile(
        last_reply_at=now - timedelta(minutes=5),
        received_at=now,
        received_status={"status": "busy"},
        user_emotion=None,
    )
    assert profile["interaction_mode"] == "conversation_mode"
    assert 1 <= profile["delay_seconds"] <= 5


def test_compute_delay_profile_high_emotion(reply_delay_enabled):
    now = datetime.now(timezone.utc)
    profile = compute_delay_profile(
        last_reply_at=now - timedelta(hours=2),
        received_at=now,
        received_status={"status": "very_busy"},
        user_emotion={"arousal": 0.9, "pleasure": -0.4},
    )
    assert profile["interaction_mode"] == "high_emotion"


def test_compute_delay_profile_disabled_by_default():
    """默认 settings.reply_delay_enabled=False → 短路返 disabled / 0s."""
    now = datetime.now(timezone.utc)
    profile = compute_delay_profile(
        last_reply_at=now - timedelta(minutes=5),
        received_at=now,
        received_status={"status": "busy"},
        user_emotion={"arousal": 0.9, "pleasure": -0.4},
    )
    assert profile["interaction_mode"] == "disabled"
    assert profile["delay_seconds"] == 0.0


def test_merge_reply_contexts_keeps_first_receipt():
    base = {
        "received_at": "2026-03-19T00:00:00+00:00",
        "received_status": {"activity": "工作", "status": "busy"},
        "delay_reason": "schedule_busy",
    }
    latest = {
        "received_at": "2026-03-19T00:00:05+00:00",
        "received_status": {"activity": "自由时间", "status": "idle"},
        "delay_reason": "conversation_mode",
    }
    merged = merge_reply_contexts(base, latest)
    assert merged["received_status"]["activity"] == "工作"
    assert merged["latest_received_at"] == latest["received_at"]


def test_actual_delay_seconds_works():
    now = datetime.now(timezone.utc)
    context = {"received_at": (now - timedelta(seconds=90)).isoformat()}
    value = actual_delay_seconds(context, now=now)
    assert value is not None
    assert 89 <= value <= 91
