"""Tests for per-game home-screen record stats.

The home used to sum the last page of sessions (16 on the client), so 总对局 /
胜利局 / 胜率 / 时长 all froze after that many rounds. These cover the SQL
replacement: the query shape, the scored-round filter, and the win-rate rule
that keeps 中途退出 in the denominator (quitting deducts points).
"""

from __future__ import annotations

import pytest

from app.services.games import native
from app.services.games.substance import action_floor


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
async def test_record_stats_maps_the_aggregate_row(monkeypatch):
    fake = _CapturingDb(
        {
            "total_rounds": 20,
            "wins": 8,
            "losses": 7,
            "draws": 2,
            "aborted": 3,
            "total_seconds": 5400,
        }
    )
    monkeypatch.setattr(native, "db", fake)

    stats = await native.get_record_stats("user-1", "gomoku")

    assert stats["total_rounds"] == 20
    assert stats["wins"] == 8
    assert stats["losses"] == 7
    assert stats["draws"] == 2
    assert stats["aborted"] == 3
    assert stats["total_seconds"] == 5400
    # 胜率 = 8 / 20 = 40; quits stay in the denominator.
    assert stats["win_rate"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_record_stats_quit_stays_in_win_rate_denominator(monkeypatch):
    fake = _CapturingDb(
        {
            "total_rounds": 2,
            "wins": 1,
            "losses": 0,
            "draws": 0,
            "aborted": 1,
            "total_seconds": 120,
        }
    )
    monkeypatch.setattr(native, "db", fake)

    stats = await native.get_record_stats("user-1", "gomoku")
    assert stats["win_rate"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_record_stats_win_rate_is_zero_when_every_round_escaped(monkeypatch):
    fake = _CapturingDb(
        {
            "total_rounds": 4,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "aborted": 4,
            "total_seconds": 90,
        }
    )
    monkeypatch.setattr(native, "db", fake)

    stats = await native.get_record_stats("user-1", "chess")
    assert stats["win_rate"] == 0.0
    assert stats["total_rounds"] == 4


@pytest.mark.asyncio
async def test_record_stats_defaults_to_zero_when_no_rows(monkeypatch):
    monkeypatch.setattr(native, "db", _CapturingDb(None))

    assert await native.get_record_stats("user-1", "reversi") == {
        "total_rounds": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "aborted": 0,
        "win_rate": 0.0,
        "total_seconds": 0,
    }


@pytest.mark.asyncio
async def test_record_stats_unknown_game_does_not_hit_the_database(monkeypatch):
    fake = _CapturingDb({"total_rounds": 9})
    monkeypatch.setattr(native, "db", fake)

    stats = await native.get_record_stats("user-1", "not_a_game")

    assert stats["total_rounds"] == 0
    assert fake.query is None


@pytest.mark.asyncio
async def test_record_stats_query_is_scoped_to_one_played_game(monkeypatch):
    fake = _CapturingDb(
        {
            "total_rounds": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "aborted": 0,
            "total_seconds": 0,
        }
    )
    monkeypatch.setattr(native, "db", fake)

    await native.get_record_stats("user-1", "xiangqi")

    assert fake.args[0] == "user-1"
    assert fake.args[1] == "xiangqi"
    assert fake.args[2] == action_floor("xiangqi")
    assert "AND game_key = $2" in fake.query
    assert "status IN ('settled', 'aborted')" in fake.query
    assert "duration_seconds IS NOT NULL" in fake.query
    assert "EXTRACT(EPOCH FROM (ended_at - started_at))" in fake.query
    # Quick-exit leftover aborts must not inflate 总对局; scored forfeits do.
    assert "IN ('win', 'lose', 'draw')" in fake.query
    assert ">= $3" in fake.query
    assert "AS escaped" in fake.query
