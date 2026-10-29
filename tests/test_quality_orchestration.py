"""Admin polling, immutable snapshots and independent executor failure paths."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tests.test_quality_postgres import quality, running
from tests.test_service_health_postgres import world
from app.services.quality import jobs, snapshots, triggers, worker, regression
from app.services import runtime_config


async def test_prompt_debounce_route_retry_and_daily_catchup(quality, monkeypatch):
    database, *_ = quality
    config = SimpleNamespace(
        remote_chat_provider="ark",
        remote_chat_model="v1",
        remote_small_provider="dashscope",
        remote_small_model="flash",
        online_model=True,
    )
    monkeypatch.setattr(runtime_config, "ensure_loaded", AsyncMock())
    monkeypatch.setattr(runtime_config, "resolve_config_sync", lambda _: config)
    monkeypatch.setattr(
        snapshots,
        "capture",
        AsyncMock(return_value={"prompts": [], "route": ["ark", "v2"]}),
    )
    versions = [dict(id="v1", prompt_key="synthetic", change_type="manual_save")]
    failing = [False]

    async def query(sql, *args):
        if sql.startswith("SELECT DISTINCT ON"):
            return versions
        return await database.query_raw(sql, *args)

    async def execute(sql, *args):
        if failing[0] and sql.startswith("UPDATE quality_jobs SET snapshot="):
            failing[0] = False
            raise RuntimeError("synthetic_cursor_write_failure")
        return await database.execute_raw(sql, *args)

    proxy = SimpleNamespace(query_raw=query, execute_raw=execute)
    monkeypatch.setattr(triggers, "db", proxy)
    monkeypatch.setattr(triggers, "CURSOR", "cursor-" + uuid4().hex)
    now = datetime(2024, 5, 1, 20, 44, tzinfo=UTC)  # local 04:44
    await triggers.poll(now)
    enqueue = jobs.enqueue
    claimed_dates = []

    async def claim_immediately(kind, *args, **kwargs):
        ident = await enqueue(kind, *args, **kwargs)
        if kind == "daily":
            claimed = await jobs.claim(job_id=ident)
            if claimed:
                claimed_dates.append(str(claimed["local_date"])[:10])
                await jobs.finish(claimed, "completed", {"test": "immediate_claim"})
        return ident

    # Simulate the other worker claiming before poll() resumes after enqueue.
    monkeypatch.setattr(jobs, "enqueue", claim_immediately)
    versions[0] = {**versions[0], "id": "v2"}
    config.remote_chat_model = "v2"
    failing[0] = True
    with pytest.raises(RuntimeError, match="cursor"):
        await triggers.poll(now + timedelta(minutes=1))
    await triggers.poll(now + timedelta(minutes=2))
    assert claimed_dates == ["2024-05-01"]
    rows = await database.query_raw(
        "SELECT kind,source_key,source_version,status FROM quality_jobs WHERE source_key IN ('synthetic','global_model_route','daily')"
    )
    assert sum(r["source_key"] == "global_model_route" for r in rows) == 1
    assert any(
        r["kind"] == "daily" and r["source_version"] == "2024-05-01" for r in rows
    )
    versions[0] = {**versions[0], "id": "v3"}
    await triggers.poll(now + timedelta(minutes=3))
    rows = await database.query_raw(
        "SELECT source_version,status FROM quality_jobs WHERE source_key='synthetic' ORDER BY source_version"
    )
    assert rows == [
        {"source_version": "v2", "status": "skipped"},
        {"source_version": "v3", "status": "pending"},
    ]
    versions[:] = [
        dict(id=str(i), prompt_key=str(i), change_type="manual_save")
        for i in range(501)
    ]
    with pytest.raises(ValueError, match="bound"):
        await triggers.poll(now)


async def test_snapshot_pins_content_enabled_route_and_prices(monkeypatch):
    config = SimpleNamespace(
        remote_chat_provider="ark", remote_chat_model="chat", online_model=True
    )
    monkeypatch.setattr(runtime_config, "ensure_loaded", AsyncMock())
    monkeypatch.setattr(runtime_config, "resolve_config_sync", lambda _: config)
    monkeypatch.setattr(
        runtime_config,
        "chat_context_policy_overrides",
        lambda: {"rounds": 8, "main_history_tokens": 5000},
    )
    monkeypatch.setattr(
        runtime_config,
        "get_pricing",
        lambda _: {"input": 0.8, "output": 2, "cached_input": 0.16},
    )
    rows = [
        dict(
            key="x", content="当前版本", is_enabled=False, updated_at=datetime.now(UTC)
        )
    ]
    tx = SimpleNamespace(
        execute_raw=AsyncMock(), query_raw=AsyncMock(side_effect=lambda *_: rows)
    )

    class Context:
        async def __aenter__(self):
            return tx

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(snapshots, "db", SimpleNamespace(tx=lambda **_: Context()))
    first = await snapshots.capture()
    assert first["prompts"][0]["is_enabled"] is False
    assert first["prices"]["ark/chat"]["cached_input"] == 0.16
    assert first["context_policy"] == {"rounds": 8, "main_history_tokens": 5000}
    rows[0]["content"] = "新版本"
    assert (await snapshots.capture())["hash"] != first["hash"]
    config.online_model = False
    with pytest.raises(ValueError, match="route"):
        await snapshots.capture()
    config.online_model = True
    rows[:] = rows * 501
    with pytest.raises(ValueError, match="bound"):
        await snapshots.capture()


@pytest.mark.parametrize("mode", ["success", "failed", "cancelled", "lost_lease"])
async def test_worker_execution_cancellation_and_terminal_signal(monkeypatch, mode):
    finish = AsyncMock()
    monkeypatch.setattr(jobs, "finish", finish)
    entered = asyncio.Event()
    closed = []

    async def work(_):
        try:
            entered.set()
            if mode == "failed":
                raise ValueError("synthetic")
            if mode in {"cancelled", "lost_lease"}:
                await asyncio.Event().wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(worker.daily, "run", work)
    monkeypatch.setattr(worker.regression, "run", work)
    job = {
        "id": "j",
        "lease_token": "l",
        "kind": "daily" if mode != "success" else "routing",
    }
    if mode == "lost_lease":

        async def sleep(_):
            await entered.wait()

        monkeypatch.setattr(
            worker, "asyncio", SimpleNamespace(**{**asyncio.__dict__, "sleep": sleep})
        )
        monkeypatch.setattr(
            jobs, "renew", AsyncMock(side_effect=jobs.LostLease("lost"))
        )
        finish.side_effect = jobs.LostLease("lost")
    task = asyncio.create_task(worker.execute(job))
    await entered.wait()
    if mode == "cancelled":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finish.call_args.args[1] == "interrupted"
    else:
        await task
    assert closed == [True]
    if mode == "failed":
        assert finish.call_args.args[1] == "failed"
    if mode == "success":
        finish.assert_not_awaited()


async def test_tick_disabled_active_and_sql_dispatch(quality, monkeypatch):
    database, *_ = quality
    import app.db as app_db

    monkeypatch.setattr(app_db, "db", database)
    monkeypatch.setattr(worker, "_task", None)
    monkeypatch.setattr(worker.settings, "quality_enabled", False)
    poll = AsyncMock()
    monkeypatch.setattr(triggers, "poll", poll)
    await worker.tick()
    poll.assert_not_awaited()
    monkeypatch.setattr(worker.settings, "quality_enabled", True)
    claim = AsyncMock(return_value={"id": "synthetic", "kind": "daily"})
    monkeypatch.setattr(jobs, "claim", claim)
    entered = asyncio.Event()

    async def execute(_):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "execute", execute)
    await worker.tick()
    await entered.wait()
    await worker.tick()
    assert poll.await_count == 1 and claim.await_count == 1
    await worker.stop()
    assert worker._task.done()
