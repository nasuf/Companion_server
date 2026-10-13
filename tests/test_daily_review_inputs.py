"""Bounded projections and classification controls for derived daily memories."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from app.services.schedule_domain import review_inputs as inputs, review_write as writer
from app.services.memory.evidence_read import daily_summary_details


@pytest.mark.parametrize("result", [None, [], {"memories": "bad"}])
def test_invalid_classification_fails(result):
    with pytest.raises(ValueError):
        writer.candidates(result, 5)


@pytest.mark.parametrize(
    "item",
    [
        {},
        None,
        1,
        {"content": False},
        {"content": "x", "score": True},
        {"content": "x", "importance": True},
        {"content": "x", "importance": "bad"},
        {"content": "x", "score": float("nan")},
        {"content": "x", "score": 9},
        {"content": "x", "score": float("inf")},
        {"content": "x" * 10001},
        {"content": "x", "type": "身份"},
        {"content": "  "},
        {"content": "x", "type": "偏好"},
    ],
)
def test_nonfacts_and_invalid_scores_cannot_enter_candidates(item):
    assert writer.candidates({"memories": [item]}, 5) == []


def test_legacy_strings_compatible_and_budget_bounded():
    assert writer.candidates({"memories": ["甲", "甲", "乙", "丙"]}, 2) == [
        {"content": "甲", "main": "生活", "importance": 0.3},
        {"content": "乙", "main": "生活", "importance": 0.3},
    ]
    assert (
        writer.candidates(
            {"memories": [{"type": "情绪记忆", "content": "甲", "score": 10}]}, 5
        )[0]["importance"]
        == 0.1
    )
    assert writer.candidates({"memories": ["甲"]}, 0) == []


def sample():
    return {
        "local_date": "2026-10-12",
        "timezone": "Asia/Shanghai",
        "sampled_at": "2026-10-13T01:00:00+00:00",
        "schedule_slots": 2,
        "adjustments": 0,
        "proactive_messages": 5,
        "proactive_truncated": True,
        "games": 0,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("timezone", "Not/AZone"),
        ("local_date", "PRIVATE"),
        ("local_date", "2026-02-30"),
        ("local_date", None),
        ("sampled_at", "2026-10-13"),
        ("sampled_at", None),
        ("proactive_truncated", 0),
        ("games", -1),
        ("games", 201),
        ("schedule_slots", True),
        ("proactive_messages", None),
    ],
)
def test_unknown_details_cannot_be_presented_as_zero_or_success(key, value):
    assert daily_summary_details({**sample(), key: value}) is None


def test_public_details_strip_private_extra_fields_and_preserve_real_zero():
    assert (
        daily_summary_details(
            {**sample(), "prompt": "PRIVATE", "source_ids": ["secret"]}
        )
        == sample()
    )
    assert daily_summary_details(None) is None
    p = sample()
    del p["sampled_at"]
    assert daily_summary_details(p) is None


def test_game_receipts_do_not_invent_completed_results():
    assert "无游戏业务记录" in inputs.render_games([])
    text = inputs.render_games(
        [
            {"status": "started", "outcome": "win"},
            {"status": "settled", "outcome": "lose", "dur": 120, "title": "猜词"},
            {"status": "settled", "outcome": "draw"},
        ]
    )
    assert (
        "未结算1局" in text
        and "用户赢0局" in text
        and "AI赢1局" in text
        and "平局1局" in text
        and "记录时长2分钟" in text
        and "《猜词》" in text
    )


@pytest.mark.parametrize(
    "schedule",
    [
        {},
        [{}],
        [{"start": 1, "end": "10:00"}],
        [{"start": "09:00", "end": "10:00"}] * 97,
        [{"start": "09:00", "end": "10:00", "event": "x" * 262144}],
    ],
)
async def test_invalid_or_excess_schedule_rejected_before_model(schedule, monkeypatch):
    db = AsyncMock()
    monkeypatch.setattr(
        inputs, "memory_scope", AsyncMock(return_value={"agent_id": "a"})
    )
    db.query_raw.side_effect = [
        [],
        [{"name": "Agent"}],
        [{"id": "s", "schedule_data": schedule}],
    ]
    start = datetime(2026, 10, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    with pytest.raises(ValueError, match="invalid_schedule"):
        await inputs.collect_inputs(
            db,
            user_id="u",
            workspace_id="w",
            agent_id="a",
            start=start,
            end=start + timedelta(days=1),
        )


@pytest.mark.parametrize("overflow", ["adjustments", "games", "bytes"])
async def test_overflow_is_not_silently_truncated_or_reported_as_empty(
    overflow, monkeypatch
):
    db = AsyncMock()
    monkeypatch.setattr(
        inputs, "memory_scope", AsyncMock(return_value={"agent_id": "a"})
    )
    db.query_raw.side_effect = [
        [],
        [{"name": "Agent"}],
        [],
        [{}] * 101 if overflow == "adjustments" else [],
        [{"message": "x" * 524289}] if overflow == "bytes" else [],
        [{}] * 201 if overflow == "games" else [],
    ]
    start = datetime(2026, 10, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    with pytest.raises(ValueError, match="limit|too_large"):
        await inputs.collect_inputs(
            db,
            user_id="u",
            workspace_id="w",
            agent_id="a",
            start=start,
            end=start + timedelta(days=1),
        )


async def test_invalid_window_rejected():
    start = datetime(2026, 10, 12)
    with pytest.raises(ValueError, match="invalid_window"):
        await inputs.collect_inputs(
            AsyncMock(),
            user_id="u",
            workspace_id="w",
            agent_id="a",
            start=start,
            end=start + timedelta(days=1),
        )


@pytest.mark.parametrize(
    "workspace",
    [
        None,
        SimpleNamespace(userId="other", agentId="a"),
        SimpleNamespace(userId="u", agentId="other"),
    ],
)
async def test_daily_review_rejects_wrong_active_workspace_before_claim(
    workspace, monkeypatch
):
    from app.services.schedule_domain import schedule
    from app.services.workspace import workspaces

    database = AsyncMock()
    monkeypatch.setattr(schedule, "db", database)
    monkeypatch.setattr(
        workspaces, "get_active_workspace", AsyncMock(return_value=workspace)
    )
    assert await schedule.review_daily_schedule("a", "u") == []
    database.query_raw.assert_not_awaited()
