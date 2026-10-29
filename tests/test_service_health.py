"""Service-level alert rules: thresholds, unknown handling, incident decisions and wording."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.ops import service_health as h


def sample(*alerting, unknown=()):
    checks = {n: {"status": "measured", "count": 0, "threshold": 1, "window_minutes": 15, "after_minutes": 3,
                  "replies": 0, "jobs": [], "alerting": n in alerting} for n in h.CHECKS}
    for n in alerting:
        checks[n].update(count=3, jobs=["daily_schedule"])
    for n in unknown:
        checks[n] = {"status": "unknown", "error_type": "TimeoutError"}
    return {"sampled_at": "2026-10-27T00:00:00+00:00", "checks": checks}


def test_any_alerting_check_is_a_warning_and_unmeasured_never_clears_one():
    assert h.level_for(sample("reply_failed"), "normal") == "warning"
    assert h.level_for(sample(), "normal") == "normal"
    assert h.level_for(sample(unknown=["memory_blocked"]), "normal") == "unknown"
    assert h.level_for(sample(unknown=["memory_blocked"]), "warning") == "warning"
    assert h.level_for(sample("unanswered", unknown=["memory_blocked"]), "normal") == "warning"


N, W, U = "normal", "warning", "unknown"


@pytest.mark.parametrize("notified,level,failing,streak,push,after", [
    ((N, []), N, [], 0, False, (N, [])),                    # a quiet start sends nothing
    ((N, []), W, ["a"], 0, True, (W, ["a"])),                # first failure pushes
    ((W, ["a"]), W, ["a"], 0, False, (W, ["a"])),            # the same incident does not repeat
    ((W, ["a"]), W, ["a", "b"], 0, True, (W, ["a", "b"])),   # a newly failing check pushes
    ((W, ["a", "b"]), W, ["b"], 0, False, (W, ["a", "b"])),  # a shrinking set does not
    ((W, ["a", "b"]), W, [], 0, False, (W, ["a", "b"])),     # latched by an unknown sample
    ((W, ["a"]), N, [], 0, False, (W, ["a"])),              # first healthy sample only starts recovery
    ((N, []), U, [], 2, False, (N, [])),                     # unknown must persist first
    ((N, []), U, [], 3, True, (U, [])),
    ((U, []), U, [], 9, False, (U, [])),
    ((U, []), N, [], 0, False, (U, [])),
])
def test_one_push_per_incident(notified, level, failing, streak, push, after):
    current = {"checks": {name: {"status": "measured", "alerting": True} for name in failing}}
    state = {"level": notified[0], "checks": notified[1]}
    actual_push, actual = h.decide(level, current, state, streak)
    assert actual_push == push
    if level == N and notified[0] != N:
        assert actual.pop("recovery")["samples"] == 1
    assert actual == {"level": after[0], "checks": after[1]}


def test_messages_name_each_failing_check_and_never_call_unknown_healthy():
    current = sample("unanswered", "cron_failing")
    current["checks"]["unanswered"]["after_minutes"] = 3
    title, body = h.message_for("warning", current)
    assert title == "服务告警" and "3条用户消息超过3分钟未回复" in body and "daily_schedule" in body
    awaiting = sample("reply_failed")
    awaiting["checks"]["reply_failed"].update(count=1, threshold=2)
    assert "尚未出现成功回复" in h.message_for("warning", awaiting)[1]
    title, body = h.message_for("unknown", sample(unknown=["memory_blocked"]))
    assert title == "服务巡检异常" and "记忆写入受阻" in body and "不代表服务正常" in body
    assert h.message_for("normal", sample())[0] == "服务告警恢复"


async def test_unmeasured_checks_keep_their_evidence_and_thresholds_are_inclusive(monkeypatch):
    seen = {}

    async def ok(now, carry):
        seen["ok"] = carry
        return {"count": 2, "threshold": 2, "alerting": True, "_carry": {"unanswered_ids": ["m1", "m2"]}}

    async def down(now, carry):
        raise RuntimeError("db down")

    monkeypatch.setattr(h, "CHECKS", {"unanswered": ok, "reply_failed": down})
    previous = {"unanswered": {"unanswered_ids": ["m0"]}, "reply_failed": {"last_failed_at": "2026-10-27T00:00:00"}}
    result = await h.collect_metrics(datetime(2026, 10, 27, tzinfo=UTC), previous)
    assert seen["ok"] == {"unanswered_ids": ["m0"]}
    assert result["checks"]["reply_failed"] == {"status": "unknown", "error_type": "RuntimeError"}
    assert "_carry" not in result["checks"]["unanswered"]
    # Fresh evidence replaces the old; an unmeasured check keeps what it had.
    assert result["carry"] == {"unanswered": {"unanswered_ids": ["m1", "m2"]},
                               "reply_failed": {"last_failed_at": "2026-10-27T00:00:00"}}


async def test_cron_check_reads_health_itself_and_unreadable_is_unknown(monkeypatch):
    from app import redis_client
    from jobs import scheduler
    redis = SimpleNamespace(hgetall=AsyncMock(return_value={}))
    monkeypatch.setattr(redis_client, "get_redis", AsyncMock(return_value=redis))
    with pytest.raises(RuntimeError, match="cron_health_unavailable"):
        await h._cron_failing(datetime(2026, 10, 27, tzinfo=UTC), {})
    redis.hgetall = AsyncMock(side_effect=ConnectionError("down"))
    result = await h.collect_metrics(datetime(2026, 10, 27, tzinfo=UTC))
    assert result["checks"]["cron_failing"]["status"] == "unknown"
    # A readable record reports only failing jobs.
    redis.hgetall = AsyncMock(return_value={"x": "y"})
    monkeypatch.setattr("app.services.ops.cron_health.build_report", lambda raw, jobs, now: SimpleNamespace(
        jobs=[SimpleNamespace(job_id="daily_schedule", verdict="failing"), SimpleNamespace(job_id="b", verdict="stale")]))
    monkeypatch.setattr(scheduler.scheduler, "get_jobs", lambda: [])
    result = await h._cron_failing(datetime(2026, 10, 27, tzinfo=UTC), {})
    assert result["count"] == 1 and result["jobs"] == ["daily_schedule"] and result["alerting"]


@pytest.mark.parametrize("enabled,max_seconds,minutes", [(False, 3600, 3), (True, 30, 3), (True, 300, 7), (True, 3600, 62)])
def test_intentional_reply_delays_are_not_unanswered(monkeypatch, enabled, max_seconds, minutes):
    monkeypatch.setattr("app.services.runtime_config.resolve_config_sync",
                        lambda: SimpleNamespace(reply_delay_enabled=enabled, reply_delay_max_seconds=max_seconds))
    assert h.unanswered_after().total_seconds() == minutes * 60


def test_the_monitor_rides_the_local_redis_job():
    import inspect
    from jobs import scheduler
    source = inspect.getsource(scheduler._run_redis_health_recheck)
    assert "sample_service_health()" in source and 'event_type="admin_service_health"' in source
    assert "service_health_monitor" not in {j.id for j in scheduler.scheduler_job_definitions()}
