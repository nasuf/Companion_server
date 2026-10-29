import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit
from unittest.mock import AsyncMock

import pytest
from prisma import Prisma
from app.services.ops import provider_failing as f, service_health as h


@pytest.fixture
async def database(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("isolated PostgreSQL required")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    db = Prisma(datasource={"url": url}, http={"trust_env": False})
    await db.connect()
    monkeypatch.setattr(f, "db", db)
    monkeypatch.setattr(f, "_pending", [])
    monkeypatch.setattr(f, "_batch", None)
    monkeypatch.setattr(f, "_lost", 0)
    monkeypatch.setattr(f, "_lock", None)
    monkeypatch.setattr(f, "_sequence", 0)
    await db.execute_raw("DELETE FROM ops_provider_health WHERE id='primary'")
    try:
        yield db
    finally:
        await db.execute_raw("DELETE FROM ops_provider_health WHERE id='primary'")
        await db.disconnect()


def batch(id, events, at):
    return {"worker": id, "sequence": 1, "events": events, "at": at, "lost": 0}


def event(at, kind="failure", **kw):
    return dict(
        {
            "route": "ark/m/chat",
            "request": str(at),
            "kind": kind,
            "at": at,
            "started": at - 1,
        },
        **kw,
    )


async def test_real_sql_two_workers_merge_and_retries_do_not_double_count(database):
    now = datetime.now(UTC)
    t = now.timestamp()
    a = batch("worker-A", [event(t - 8 + i) for i in range(3)], t)
    b = batch("worker-B", [event(t - 5 + i) for i in range(3)], t)
    await asyncio.gather(f._persist_batch(a), f._persist_batch(b))
    await f._persist_batch(a)  # ambiguous network outcome: same frozen ID
    check = await f.measure(now, {})
    assert (
        check["count"] == 1
        and check["routes"][0]["calls"] == 6
        and check["routes"][0]["failures"] == 6
    )
    assert check["status"] == "measured"
    # Persisted incident survives an empty window and fresh worker buffers.
    await database.execute_raw(
        "UPDATE ops_provider_health SET sampled_at=$1::timestamptz",
        (now + timedelta(minutes=20)).isoformat(),
    )
    assert (await f.measure(now + timedelta(minutes=20), {}))["alerting"]
    await f._persist_batch(
        batch("old-success", [event(t, "success", started=t - 20)], t)
    )
    assert (await f.measure(now, {}))["alerting"]
    await f._persist_batch(
        batch("new-success", [event(t + 2, "success", started=t + 1)], t)
    )
    assert not (await f.measure(now + timedelta(seconds=3), {}))["alerting"]
    assert (await f.measure(now + timedelta(minutes=25), {}))[
        "status"
    ] == "unknown"  # collector stale


async def test_atomic_failure_retains_buffer_and_new_events_arrive_during_flush(
    database, monkeypatch
):
    f.Call("ark", "m").record("failure", f._at())
    real = f._persist_batch

    async def outage(batch, **kwargs):
        f.Call("ark", "m").record("failure", f._at())
        raise RuntimeError("synthetic SQL outage")

    monkeypatch.setattr(f, "_persist_batch", outage)
    with pytest.raises(RuntimeError):
        await f.flush()
    assert f._batch and len(f._pending) == 1
    monkeypatch.setattr(f, "_persist_batch", real)
    await f.flush()
    await f.flush()
    assert (await f.measure(datetime.now(UTC), {}))["routes"][0]["calls"] == 2
    assert not f._pending and f._batch is None


async def test_not_collected_no_sample_and_overflow_are_distinct(database, monkeypatch):
    assert (await f.measure(datetime.now(UTC), {}))["reason"] == "not_collected"
    await f.flush()
    result = await f.measure(datetime.now(UTC), {})
    assert result["status"] == "measured" and result["sample_state"] == "no_sample"
    monkeypatch.setattr(f.settings, "provider_failing_buffer_limit", 1)
    for _ in range(2):
        f.Call("ark", "m").record("failure", f._at())
    await f.flush()
    result = await f.measure(datetime.now(UTC), {})
    assert result["status"] == "unknown" and result["lost_observations"] == 1
    # Restarting local state cannot erase the durable data-loss marker.
    monkeypatch.setattr(f, "_lost", 0)
    assert (await f.measure(datetime.now(UTC), {}))["incomplete"]


async def test_expired_batch_cannot_replay_as_fresh_evidence(database):
    t = f._at()
    old = t - f.settings.provider_failing_receipt_seconds - 1
    await f._persist_batch(batch("too-old", [event(old)], old))
    result = await f.measure(datetime.now(UTC), {})
    assert result["incomplete"] and not result["routes"]


async def test_real_outbox_dedup_with_available_provider_evidence(
    database, monkeypatch
):
    monkeypatch.setattr(h, "db", database)
    monkeypatch.setattr(
        "app.services.notifications.service.db", database, raising=False
    )
    monkeypatch.setattr(h.provider_budget, "refresh_active", AsyncMock())
    admin = await database.user.create(
        data={"username": "provider-health-test-" + str(f._at()), "role": "admin"}
    )
    monkeypatch.setattr(h.settings, "redis_memory_alert_user_ids", admin.id)
    await database.execute_raw("DELETE FROM ops_service_health WHERE id='primary'")
    for _ in range(5):
        f.Call("ark", "m", "utility").record("failure", f._at())
    await f.flush()

    async def collect(*args):
        return {
            "checks": {"provider_failing": await f.measure(datetime.now(UTC), {})},
            "carry": {},
        }

    monkeypatch.setattr(h, "collect_metrics", collect)
    try:
        await asyncio.gather(h.sample_and_notify(), h.sample_and_notify())
        await database.execute_raw(
            "UPDATE ops_service_health SET sampled_at=NOW()-INTERVAL '5 minutes'"
        )
        await h.sample_and_notify()
        rows = await database.query_raw(
            "SELECT type,title,body FROM notification_events WHERE user_id=$1", admin.id
        )
        assert len(rows) == 1 and rows[0]["type"] == "admin_service_health"
        assert "小模型（ark/m）" in rows[0]["body"] and "5次/5次" in rows[0]["body"]
        f.Call("ark", "m", "utility").record("success", f._at() + 1)
        await f.flush()
        await database.execute_raw(
            "UPDATE ops_service_health SET sampled_at=NOW()-INTERVAL '5 minutes'"
        )
        await h.sample_and_notify()
        assert (
            len(
                await database.query_raw(
                    "SELECT id FROM notification_events WHERE user_id=$1", admin.id
                )
            )
            == 2
        )
    finally:
        await database.execute_raw(
            "DELETE FROM notification_events WHERE user_id=$1", admin.id
        )
        await database.user.delete(where={"id": admin.id})
        await database.execute_raw("DELETE FROM ops_service_health WHERE id='primary'")


async def test_receipts_are_bounded_and_partial_monitor_cannot_look_healthy(
    database, monkeypatch
):
    monkeypatch.setattr(f.settings, "provider_failing_receipt_limit", 1)
    t = f._at()
    await f._persist_batch(batch("one", [], t))
    await f._persist_batch(batch("two", [], t))
    row = (
        await database.query_raw(
            "SELECT snapshot FROM ops_provider_health WHERE id='primary'"
        )
    )[0]["snapshot"]
    if isinstance(row, str):
        row = json.loads(row)
    assert len(row["receipts"]) == 1 and row["gap_at"]
    assert (await f.measure(datetime.now(UTC), {}))["status"] == "unknown"


async def test_late_other_worker_denominator_is_merged_before_alert_latching(database):
    t = f._at()
    await f._persist_batch(
        batch("failure-worker", [event(t - 10 + i) for i in range(5)], t)
    )
    # Older successes arrived in a later flush. They are NOT recovery evidence,
    # but belong in the denominator (5 / 25 < 50%). No sample has alerted yet.
    await f._persist_batch(
        batch(
            "success-worker",
            [event(t - 30 + i, "success", started=t - 60) for i in range(20)],
            t,
        )
    )
    check = await f.measure(datetime.now(UTC), {})
    assert check["count"] == 0 and not check["alerting"]
    assert check["routes"][0]["calls"] == 25 and check["routes"][0]["failures"] == 5


async def test_healthy_peer_cannot_mask_a_stale_peer_or_latch_partial_denominator(
    database, monkeypatch
):
    now = datetime.now(UTC)
    t = now.timestamp()
    await f._persist_batch(batch("lagging", [], t - 121))
    await f._persist_batch(batch("healthy", [event(t - 10 + i) for i in range(5)], t))
    check = await f.measure(now, {})
    assert check["status"] == "unknown" and check["peer_collection_incomplete"]
    assert check["observed_workers"] == 2
    assert not check["alerting"]  # denominator may be missing successful calls
    recovered = batch(
        "lagging", [event(t - 30 + i, "success", started=t - 60) for i in range(20)], t
    )
    recovered["sequence"] = 2
    await f._persist_batch(recovered)
    check = await f.measure(now, {})
    assert check["status"] == "measured" and not check["alerting"]
    assert check["routes"][0]["calls"] == 25


async def test_collection_gap_clears_after_one_window(database):
    now = datetime.now(UTC)
    t = now.timestamp()
    await f._persist_batch({**batch("lossy", [], t), "lost": 1})
    check = await f.measure(now, {})
    assert check["status"] == "unknown" and check["gap_until"]
    later = now + timedelta(seconds=f.settings.provider_failing_window_seconds + 5)
    await database.execute_raw(
        "UPDATE ops_provider_health SET sampled_at=$1::timestamptz", later.isoformat()
    )
    check = await f.measure(later, {})
    assert check["status"] == "measured" and check["gap_until"] is None
    assert check["lost_observations"] == 1


async def test_legacy_permanent_incomplete_flag_is_dropped(database):
    t = f._at()
    await database.execute_raw(
        "INSERT INTO ops_provider_health(id, snapshot, sampled_at) "
        "VALUES('primary', $1::jsonb, now())",
        json.dumps({"routes": {}, "receipts": {}, "incomplete": True}),
    )
    await f._persist_batch(batch("upgraded", [], t))
    check = await f.measure(datetime.now(UTC), {})
    assert check["status"] == "measured" and not check["incomplete"]


async def test_gracefully_stopped_worker_is_not_a_lagging_peer(database, monkeypatch):
    # Every deploy stops both workers. Their last receipts must not make the
    # next workers' samples unknown (a false "服务巡检异常" and then "恢复").
    f.Call("ark", "m").record("success", f._at())
    await f.retire()
    row = (
        await database.query_raw("SELECT snapshot FROM ops_provider_health WHERE id='primary'")
    )[0]["snapshot"]
    if isinstance(row, str):
        row = json.loads(row)
    receipt = row["receipts"][f._worker_id]
    assert receipt["closed"] and receipt["sequence"] == 2 and not f._pending
    now = datetime.now(UTC)
    later = now + timedelta(seconds=f.settings.provider_failing_stale_seconds + 30)
    successor = batch("successor", [], later.timestamp())
    await f._persist_batch(successor)
    await database.execute_raw(
        "UPDATE ops_provider_health SET sampled_at=$1::timestamptz", later.isoformat()
    )
    check = await f.measure(later, {})
    assert check["status"] == "measured" and not check["peer_collection_incomplete"]
    assert check["observed_workers"] == 1
    # A worker that died without retiring still counts as lagging.
    await f._persist_batch(batch("crashed", [], now.timestamp()))
    await f._persist_batch({**successor, "sequence": 2})
    check = await f.measure(later, {})
    assert check["status"] == "unknown" and check["peer_collection_incomplete"]


async def test_retire_persists_a_frozen_retry_batch_before_closing(database, monkeypatch):
    f.Call("ark", "m").record("failure", f._at())
    real = f._persist_batch

    async def outage(batch, **kwargs):
        raise RuntimeError("synthetic SQL outage")

    monkeypatch.setattr(f, "_persist_batch", outage)
    with pytest.raises(RuntimeError):
        await f.flush()
    f.Call("ark", "m").record("failure", f._at())
    monkeypatch.setattr(f, "_persist_batch", real)
    await f.retire()
    check = await f.measure(datetime.now(UTC), {})
    assert check["routes"][0]["calls"] == 2 and f._batch is None and not f._pending
