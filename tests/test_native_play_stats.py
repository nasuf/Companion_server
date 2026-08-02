"""Tests for the game hub's aggregate play counters.

The hub used to count a page of sessions client-side, which pinned "累计相伴"
at the page size. These cover the SQL-backed replacement: the query shape, the
UTC+8 day window it asks for, and the response mapping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services.games import native


class _CapturingDb:
    def __init__(self, row):
        self.row = row
        self.query = None
        self.args = None

    async def query_raw(self, query, *args):
        self.query = query
        self.args = args
        return [self.row] if self.row is not None else []


@pytest.mark.asyncio
async def test_play_stats_maps_the_aggregate_row(monkeypatch):
    fake = _CapturingDb(
        {"total_rounds": 37, "total_seconds": 4321, "today_seconds": 600}
    )
    monkeypatch.setattr(native, "db", fake)

    stats = await native.get_play_stats("user-1")

    assert stats == {
        "total_rounds": 37,
        "total_seconds": 4321,
        "today_seconds": 600,
    }


@pytest.mark.asyncio
async def test_play_stats_defaults_to_zero_when_no_rows(monkeypatch):
    monkeypatch.setattr(native, "db", _CapturingDb(None))

    assert await native.get_play_stats("user-1") == {
        "total_rounds": 0,
        "total_seconds": 0,
        "today_seconds": 0,
    }


@pytest.mark.asyncio
async def test_play_stats_counts_unfinished_sessions(monkeypatch):
    """No status filter: an abandoned match still counts as a round."""
    fake = _CapturingDb({"total_rounds": 1, "total_seconds": 0, "today_seconds": 0})
    monkeypatch.setattr(native, "db", fake)

    await native.get_play_stats("user-1")

    assert "status" not in fake.query
    assert "provider = 'native'" in fake.query
    # Sessions that never started contribute no time but are still counted.
    assert "ELSE 0" in fake.query


@pytest.mark.asyncio
async def test_play_stats_asks_for_the_local_day_window(monkeypatch):
    fake = _CapturingDb({"total_rounds": 0, "total_seconds": 0, "today_seconds": 0})
    monkeypatch.setattr(native, "db", fake)
    # 2026-08-02 09:30 UTC+8 → the window must be the whole local day, expressed
    # as the naive UTC values the timestamp columns store.
    local_now = datetime(
        2026, 8, 2, 9, 30, tzinfo=timezone(timedelta(hours=8))
    )
    monkeypatch.setattr(
        native.time_service,
        "get_current_time",
        lambda: SimpleNamespace(now=local_now),
    )

    await native.get_play_stats("user-1")

    user_id, day_start, day_end = fake.args
    assert user_id == "user-1"
    assert day_start == datetime(2026, 8, 1, 16, 0)
    assert day_end == datetime(2026, 8, 2, 16, 0)
    assert day_start.tzinfo is None and day_end.tzinfo is None
