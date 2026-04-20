"""Tests for reply timing context service."""

from datetime import datetime, timedelta, timezone

from app.services.interaction.reply_context import (
    actual_delay_seconds,
    compute_delay_profile,
    merge_reply_contexts,
)


def test_compute_delay_profile_conversation_mode():
    now = datetime.now(timezone.utc)
    profile = compute_delay_profile(
        last_reply_at=now - timedelta(minutes=5),
        received_at=now,
        received_status={"status": "busy"},
        user_emotion=None,
    )
    assert profile["interaction_mode"] == "conversation_mode"
    assert 1 <= profile["delay_seconds"] <= 5


def test_compute_delay_profile_high_emotion():
    now = datetime.now(timezone.utc)
    profile = compute_delay_profile(
        last_reply_at=now - timedelta(hours=2),
        received_at=now,
        received_status={"status": "very_busy"},
        user_emotion={"arousal": 0.9, "pleasure": -0.4},
    )
    assert profile["interaction_mode"] == "high_emotion"


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
