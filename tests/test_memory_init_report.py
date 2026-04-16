"""Unit tests for init_report — ensures report captures phases and survives errors."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.init_report import (
    InitReport,
    MainStats,
    init_report,
    phase_timer,
)


@pytest.mark.asyncio
async def test_report_persists_on_normal_exit(fake_redis):
    with patch("app.services.memory.init_report.get_redis",
               AsyncMock(return_value=fake_redis)):
        async with init_report("agent-1", profile_id="p1") as rep:
            rep.direct_count = 10
            rep.total_stored = 200
            rep.distinct_subs = 50

    stored_json = fake_redis._store["memgen:report:agent-1"]
    data = json.loads(stored_json)
    assert data["agent_id"] == "agent-1"
    assert data["profile_id"] == "p1"
    assert data["direct_count"] == 10
    assert data["total_stored"] == 200
    assert data["total_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_report_persists_even_on_exception(fake_redis):
    """If generation raises, we still write the partial report."""
    with patch("app.services.memory.init_report.get_redis",
               AsyncMock(return_value=fake_redis)):
        with pytest.raises(RuntimeError):
            async with init_report("agent-err") as rep:
                rep.direct_count = 5
                raise RuntimeError("mid-generation failure")

    assert "memgen:report:agent-err" in fake_redis._store
    data = json.loads(fake_redis._store["memgen:report:agent-err"])
    assert data["direct_count"] == 5


def test_phase_timer_records_duration():
    rep = InitReport(agent_id="x")
    with phase_timer(rep, "phase-x"):
        # Trivial work — timer records > 0 ms (may be 0 on very fast machines).
        for _ in range(1000):
            _ = [i for i in range(10)]
    assert "phase-x" in rep.phase_ms
    assert rep.phase_ms["phase-x"] >= 0


def test_main_stats_default_zeros():
    s = MainStats()
    assert s.tokens_in == 0
    assert s.tokens_out == 0
    assert s.duration_ms == 0
    assert s.produced == 0
    assert s.failed is False


def test_to_dict_includes_stats():
    rep = InitReport(agent_id="a")
    rep.main_stats = {"身份": MainStats(tokens_in=100, tokens_out=200, produced=5)}
    d = rep.to_dict()
    assert d["main_stats"]["身份"]["tokens_in"] == 100
    assert d["main_stats"]["身份"]["produced"] == 5


@pytest.mark.asyncio
async def test_persist_failure_swallowed():
    """If Redis write blows up, generation should not fail just because the
    observability sidecar couldn't write — report persistence is best-effort."""
    fake_redis_that_fails = AsyncMock()
    fake_redis_that_fails.set = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("app.services.memory.init_report.get_redis",
               AsyncMock(return_value=fake_redis_that_fails)):
        async with init_report("agent-z") as rep:
            rep.direct_count = 3

    # Exiting without raise is itself the assertion; verify we did attempt the
    # write (so the failure path actually ran) and user state survived.
    fake_redis_that_fails.set.assert_awaited_once()
    assert rep.direct_count == 3
    assert rep.ended_at is not None
