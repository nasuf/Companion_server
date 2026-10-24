from unittest.mock import AsyncMock

import pytest

from app.services.ops import redis_capacity as c


@pytest.mark.parametrize("percent,previous,expected", [
    (0, "normal", "normal"), (69.99, "normal", "normal"), (70, "normal", "warning"),
    (69, "warning", "warning"), (65, "warning", "warning"), (64.99, "warning", "normal"),
    (None, "warning", "unknown"),
])
def test_threshold_and_hysteresis(percent, previous, expected):
    assert c.level_for({"available": True, "mem_percent": percent}, previous) == expected


@pytest.mark.parametrize("used,maximum,status,percent", [
    (0, 100, "measured", 0), (70, 100, "measured", 70),
    (5, 0, "unlimited", None), (None, 100, "invalid", None), (5, None, "invalid", None),
])
async def test_real_zero_distinct_from_unknown(monkeypatch, used, maximum, status, percent):
    redis = AsyncMock()
    redis.info.return_value = {"used_memory": used, "maxmemory": maximum}
    redis.dbsize.return_value = 0
    monkeypatch.setattr(c, "get_redis", AsyncMock(return_value=redis))
    sample = await c.collect_metrics()
    assert sample["capacity_status"] == status
    assert sample["mem_percent"] == percent
    assert sample["hit_rate"] is None


async def test_sampling_errors_are_redacted(monkeypatch):
    monkeypatch.setattr(c, "get_redis", AsyncMock(side_effect=RuntimeError("secret redis://credential")))
    sample = await c.collect_metrics()
    assert sample["available"] is False and sample["mem_percent"] is None
    assert "credential" not in str(sample)


async def test_missing_monitor_table_is_unavailable(monkeypatch):
    monkeypatch.setattr(c.db, "query_raw", AsyncMock(side_effect=RuntimeError("table unavailable")))
    assert (await c.read_status())["status"] == "unavailable"


async def test_unstarted_monitor_is_not_healthy_empty_data(monkeypatch):
    monkeypatch.setattr(c.db, "query_raw", AsyncMock(return_value=[]))
    assert (await c.read_status())["status"] == "not_sampled"


async def test_recipient_limit_rejects_unbounded_fanout(monkeypatch):
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value={"available": False}))
    monkeypatch.setattr(c.settings, "redis_memory_alert_user_ids", ",".join(str(i) for i in range(21)))
    with pytest.raises(ValueError, match="at most 20"):
        await c.sample_and_notify()


@pytest.mark.parametrize("monitor_fails", [False, True])
async def test_health_recheck_runs_sql_alert_even_when_redis_health_probe_fails(monkeypatch, monitor_fails):
    from jobs import scheduler
    from app import redis_client
    from unittest.mock import Mock
    async def run_local(name, body):
        await body()
    monkeypatch.setattr(scheduler, "_run_local_job", run_local)
    monkeypatch.setattr(redis_client, "recheck_redis_health", AsyncMock(side_effect=RuntimeError("unavailable")))
    sample = AsyncMock(side_effect=RuntimeError("SQL unavailable") if monitor_fails else None)
    dispatch = AsyncMock()
    failure = Mock()
    monkeypatch.setattr(c, "sample_and_notify", sample)
    monkeypatch.setattr(scheduler, "dispatch_due_notifications", dispatch)
    monkeypatch.setattr(scheduler, "_job_failed", failure)
    await scheduler._run_redis_health_recheck()
    sample.assert_awaited_once()
    if monitor_fails:
        dispatch.assert_not_awaited()
        assert [call.args[0] for call in failure.call_args_list] == ["Redis health recheck", "Redis capacity monitor"]
    else:
        dispatch.assert_awaited_once_with(limit=10, event_type="admin_redis_capacity")
