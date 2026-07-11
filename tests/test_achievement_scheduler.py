from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.services.achievements.rules import daily_rollup_rules
from jobs import scheduler as scheduler_module


@pytest.mark.asyncio
async def test_daily_rollup_isolates_one_pair_failure_from_later_pairs():
    pairs = [
        {"user_id": "u1", "agent_id": "a1"},
        {"user_id": "u2", "agent_id": "a2"},
    ]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=pairs)

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(
            daily_rollup_rules,
            "_run_daily_rollup_for_pair",
            AsyncMock(side_effect=[RuntimeError("broken pair"), None]),
        ) as run_pair,
    ):
        with pytest.raises(RuntimeError, match="1 pair"):
            await daily_rollup_rules.run_daily_rollup(
                datetime(2026, 6, 1, tzinfo=timezone.utc)
            )

    assert run_pair.await_count == 2
    assert run_pair.await_args_list[1].args[0] == pairs[1]


def test_achievement_rollup_job_uses_explicit_shanghai_timezone():
    fake_scheduler = MagicMock()
    with patch.object(scheduler_module, "scheduler", fake_scheduler):
        scheduler_module.setup_scheduler()

    achievement_call = next(
        call
        for call in fake_scheduler.add_job.call_args_list
        if call.kwargs.get("id") == "achievement_daily_rollup"
    )
    assert achievement_call.kwargs["hour"] == 0
    assert achievement_call.kwargs["minute"] == 5
    assert achievement_call.kwargs["timezone"] == "Asia/Shanghai"
    assert achievement_call.kwargs["coalesce"] is True
    assert achievement_call.kwargs["misfire_grace_time"] == 6 * 3600
    assert any(
        call.kwargs.get("id") == "achievement_daily_rollup_startup_catchup"
        for call in fake_scheduler.add_job.call_args_list
    )


@pytest.mark.asyncio
async def test_achievement_rollup_runner_executes_callback_and_saves_checkpoint():
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.set = AsyncMock()

    async def _run_distributed(_name, _ttl, callback):
        await callback()

    with (
        patch.object(
            scheduler_module,
            "_run_distributed_job",
            AsyncMock(side_effect=_run_distributed),
        ),
        patch.object(
            scheduler_module,
            "get_redis",
            AsyncMock(return_value=fake_redis),
        ),
        patch(
            "app.services.achievements.service.run_daily_rollup",
            AsyncMock(),
        ) as run_rollup,
    ):
        await scheduler_module._run_achievement_daily_rollup()

    run_rollup.assert_awaited_once()
    fake_redis.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_achievement_rollup_runner_catches_up_each_missed_day():
    target_day = (
        datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
    )
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(
        return_value=(target_day - timedelta(days=2)).isoformat()
    )
    fake_redis.set = AsyncMock()

    async def _run_distributed(_name, _ttl, callback):
        await callback()

    with (
        patch.object(
            scheduler_module,
            "_run_distributed_job",
            AsyncMock(side_effect=_run_distributed),
        ),
        patch.object(
            scheduler_module,
            "get_redis",
            AsyncMock(return_value=fake_redis),
        ),
        patch(
            "app.services.achievements.service.run_daily_rollup",
            AsyncMock(),
        ) as run_rollup,
    ):
        await scheduler_module._run_achievement_daily_rollup()

    assert run_rollup.await_count == 2
    assert fake_redis.set.await_count == 2
