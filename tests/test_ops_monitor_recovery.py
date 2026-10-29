from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.services.ops import (
    recovery,
    reply_slow,
    worker_health,
    redis_capacity,
    task_backlog,
    service_health,
)

NOW = datetime(2026, 10, 29, tzinfo=UTC)


def test_recovery_requires_elapsed_time_and_samples_and_gaps_reset():
    state = {"level": "warning", "checks": ["task_backlog"]}
    for minute in (0, 2, 4, 6, 8):
        notify, state = recovery.advance(
            "normal", state, NOW + timedelta(minutes=minute), interval=120
        )
        assert not notify
    notify, state = recovery.advance(
        "normal", state, NOW + timedelta(minutes=10), interval=120
    )
    assert notify and state["level"] == "normal"
    state = {"level": "warning"}
    _, state = recovery.advance("normal", state, NOW, interval=120)
    _, state = recovery.advance(
        "normal", state, NOW + timedelta(minutes=8), interval=120
    )
    assert state["recovery"]["samples"] == 1
    _, state = recovery.advance(
        "unknown", state, NOW + timedelta(minutes=10), interval=120
    )
    assert "recovery" not in state


def test_same_failure_in_recovery_grace_is_not_a_new_incident_but_other_check_is():
    prior = {
        "level": "warning",
        "checks": ["task_backlog"],
        "recovery": {
            "since": NOW.isoformat(),
            "last_at": NOW.isoformat(),
            "samples": 1,
        },
    }
    sample = {
        "sampled_at": NOW.isoformat(),
        "checks": {"task_backlog": {"status": "measured", "alerting": True}},
    }
    push, state = service_health.decide("warning", sample, prior, 0)
    assert not push and "recovery" not in state
    sample["checks"]["reply_slow"] = {"status": "measured", "alerting": True}
    assert service_health.decide("warning", sample, prior, 0)[0]


def test_worker_start_grace_missing_replacement_graceful_exit_and_recovery():
    base = NOW.timestamp()
    old = {"started_at": base, "last_seen": base + 200, "closed": False}
    new = {"started_at": base + 290, "last_seen": base + 300, "closed": False}
    assert not worker_health.classify([old], base, base + 100, {})["alerting"]
    deficit = worker_health.classify([old], base, base + 250, {})
    assert not deficit["alerting"] and deficit["bad_samples"] == 1
    assert worker_health.classify([old], base, base + 280, deficit["_carry"])[
        "alerting"
    ]
    crash = worker_health.classify([old, new], base, base + 300, {})
    assert crash["abnormal_exits"] == 1 and crash["alerting"]
    closed = {**old, "closed": True}
    assert (
        worker_health.classify([closed, new], base, base + 300, {})["abnormal_exits"]
        == 0
    )
    healthy = [
        {**new, "last_seen": base + 1300},
        {**new, "started_at": base + 1000, "last_seen": base + 1300},
    ]
    good = worker_health.classify(healthy, base, base + 1300, crash["_carry"])
    assert good["alerting"]
    assert not worker_health.classify(healthy, base, base + 1330, good["_carry"])[
        "alerting"
    ]


def timed(offset, ms):
    return {"id": str(offset), "at": NOW + timedelta(seconds=offset), "ms": ms}


def test_slow_reply_empty_window_cannot_recover_but_three_new_fast_turns_can():
    rows = [timed(-10, 25000), timed(-9, 24000), timed(-8, 23000)]
    first = reply_slow.classify(rows, NOW, {})
    assert first["alerting"] and first["count"] == 3
    empty = reply_slow.classify([], NOW + timedelta(hours=1), first["_carry"])
    assert empty["sample_state"] == "no_sample" and empty["alerting"]
    rows += [timed(1, 1000), timed(2, 19000), timed(3, 20000)]
    ok = reply_slow.classify(rows, NOW + timedelta(seconds=4), first["_carry"])
    assert not ok["alerting"] and ok["recovery_turns"] == 3
    # Old failed turns still in the window cannot re-open the same incident.
    assert not reply_slow.classify(rows, NOW + timedelta(seconds=6), ok["_carry"])[
        "alerting"
    ]
    rows += [timed(7, 25000), timed(8, 25000), timed(9, 25000)]
    assert reply_slow.classify(rows, NOW + timedelta(seconds=10), ok["_carry"])[
        "alerting"
    ]


def test_slow_p90_requires_twenty_samples():
    assert not reply_slow.classify([timed(-i, 16000) for i in range(19)], NOW, {})[
        "alerting"
    ]
    assert reply_slow.classify([timed(-i, 16000) for i in range(20)], NOW, {})[
        "alerting"
    ]


def test_redis_error_delta_restarts_and_unavailable_never_fabricate_counts():
    first = {
        "run_id": "a",
        "errorstats": {"status": "measured", "counts": {"ERR": 5429}},
    }
    redis_capacity.error_deltas(first, {}, NOW)
    assert (
        not first["error_events"]
        and first["errorstats"]["delta_state"] == "baseline_only"
    )
    second = {
        "run_id": "a",
        "errorstats": {"status": "measured", "counts": {"ERR": 5431, "OOM": 1}},
    }
    redis_capacity.error_deltas(second, first, NOW + timedelta(seconds=30))
    assert second["error_events"][-1]["counts"] == {"ERR": 2, "OOM": 1}
    missing = {"errorstats": {"status": "unknown"}}
    redis_capacity.error_deltas(missing, second, NOW + timedelta(seconds=60))
    assert missing["_error_baseline"] == second["_error_baseline"]
    restart = {
        "run_id": "b",
        "errorstats": {"status": "measured", "counts": {"ERR": 1}},
    }
    redis_capacity.error_deltas(restart, missing, NOW + timedelta(seconds=90))
    assert len(restart["error_events"]) == 1


async def test_backlog_keeps_failed_tasks_until_terminal_and_marks_missing_unknown(
    monkeypatch,
):
    async def query(sql, *args):
        if "ANY($1::text[])" in sql:
            return [{"id": "old", "status": "failed"}]
        return []

    monkeypatch.setattr(service_health, "_query", query)
    monkeypatch.setattr(
        task_backlog,
        "read_monitor_jobs",
        AsyncMock(return_value={"items": [], "truncated": False}),
    )
    r = await task_backlog.measure(NOW, {"ids": {"runtime_jobs": ["old"]}})
    assert r["alerting"] and r["count"] == 1
    monkeypatch.setattr(service_health, "_query", AsyncMock(return_value=[]))
    r = await task_backlog.measure(NOW, r["_carry"])
    assert r["status"] == "unknown" and r["_carry"]["ids"]["runtime_jobs"] == ["old"]


async def test_unknown_check_carry_never_leaks_to_public_sample(monkeypatch):
    monkeypatch.setattr(
        service_health,
        "CHECKS",
        {
            "task_backlog": AsyncMock(
                return_value={
                    "status": "unknown",
                    "_carry": {"ids": {"runtime_jobs": ["secret-message"]}},
                }
            )
        },
    )
    r = await service_health.collect_metrics(NOW)
    assert "secret-message" not in str(r["checks"])
    assert r["carry"]["task_backlog"]["ids"]
