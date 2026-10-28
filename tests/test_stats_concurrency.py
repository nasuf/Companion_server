"""Real pooled connection regression: background facts must not starve readers."""

import asyncio
import os
from datetime import timedelta
from urllib.parse import urlsplit
from unittest.mock import AsyncMock
import pytest
from app import db as database_module
from app.services.effect import daily_facts
from app.services.runtime.distributed_lock import DistributedLockNotAcquired


async def test_transaction_progresses_while_pool_admission_is_saturated(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Requires isolated PostgreSQL")
    parsed = urlsplit(url)
    assert (
        parsed.hostname in ("localhost", "127.0.0.1")
        and parsed.path == "/companion_proactive_e2e"
    )
    database = database_module.ThrottledPrisma(
        datasource={"url": url.split("?")[0] + "?connection_limit=1&pool_timeout=5"},
        http={"trust_env": False},
    )
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(database_module, "_query_semaphore", semaphore)
    await database.connect()
    pending = None
    try:
        async with database.tx(timeout=timedelta(seconds=5)) as tx:
            assert tx._tx_id is not None
            pending = asyncio.create_task(database.query_raw("SELECT 2::int AS value"))
            async with asyncio.timeout(1):
                while not semaphore.locked():
                    await asyncio.sleep(0.005)
            # The ordinary query holds the only admission permit while waiting
            # for this transaction's connection. The tx must still finish.
            result = await asyncio.wait_for(tx.query_raw("SELECT 1::int AS value"), 1)
            assert result == [{"value": 1}]
        assert await asyncio.wait_for(pending, 1) == [{"value": 2}]
    finally:
        if pending and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await database.disconnect()


async def test_recent_refresh_does_not_open_transaction_when_heal_lease_is_owned(
    monkeypatch,
):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def occupied(*args, **kwargs):
        raise DistributedLockNotAcquired("synthetic occupied")
        yield

    monkeypatch.setattr(
        "app.services.runtime.distributed_lock.distributed_lock", occupied
    )
    compute = AsyncMock()
    monkeypatch.setattr(daily_facts, "compute_day", compute)
    await asyncio.gather(
        *(
            daily_facts._refresh_recent([daily_facts.local_today()], None)
            for _ in range(20)
        )
    )
    compute.assert_not_awaited()
