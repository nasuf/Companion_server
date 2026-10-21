"""Deterministic heartbeat shutdown races, without a database or wall-clock lease."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.runtime.sql_job_consumer import PreparedJobResult, SqlJobConsumer
from app.services.runtime.sql_job_contracts import HandlerSpec, LeasePolicy, WorkerStopRequired


def fixture_queue(*, cancellation_seconds=1):
    claim = SimpleNamespace(handler='chat.execute.v1', deadline_at=None, revoked=False)
    def revoke():
        claim.revoked = True
    claim.revoke = revoke
    queue = SimpleNamespace(
        specs=[HandlerSpec(name=claim.handler, job_key='chat:0', versions=(('v1', 1),))],
        policy=LeasePolicy(heartbeat_seconds=.01, cancellation_seconds=cancellation_seconds),
        claim_next=AsyncMock(return_value=claim), finish=AsyncMock(return_value=True),
        fail=AsyncMock(),
    )
    return queue, claim


@pytest.mark.parametrize('outcome', ['success', 'handler_failure', 'cancellation'])
async def test_inflight_heartbeat_settles_without_transaction_cancellation(outcome):
    queue, claim = fixture_queue()
    entered, release, terminal = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled = []
    async def heartbeat(_):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    queue.heartbeat = heartbeat
    async def terminal_result(*args, **kwargs):
        terminal.set()
        return True
    queue.finish.side_effect = queue.fail.side_effect = terminal_result
    async def handler(_):
        await entered.wait()
        if outcome == 'handler_failure':
            raise ValueError('preparation failed')
        if outcome == 'cancellation':
            try:
                await asyncio.Event().wait()
            finally:
                terminal.set()
        return PreparedJobResult({})
    consumer = SqlJobConsumer(queue, {claim.handler: handler}, 'worker')
    polling = asyncio.create_task(consumer.poll_once())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if outcome == 'cancellation':
            polling.cancel()
        await asyncio.wait_for(terminal.wait(), 1)
        # Let the already scheduled cleanup run before allowing the DB call to
        # settle. This observes cancellation, rather than relying on sleep/TTL.
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        if outcome == 'cancellation':
            with pytest.raises(asyncio.CancelledError):
                await polling
        else:
            assert await polling
        assert not cancelled
        assert claim.revoked
    finally:
        release.set()
        await asyncio.gather(polling, return_exceptions=True)


async def test_stuck_heartbeat_requires_worker_stop_without_interrupting_database():
    queue, claim = fixture_queue(cancellation_seconds=.02)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled = []
    async def heartbeat(_):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        finally:
            finished.set()
    queue.heartbeat = heartbeat
    async def handler(_):
        await entered.wait()
        return PreparedJobResult({})
    consumer = SqlJobConsumer(queue, {claim.handler: handler}, 'worker')
    try:
        with pytest.raises(WorkerStopRequired, match='Heartbeat'):
            await consumer.poll_once()
        assert claim.revoked and not cancelled
        with pytest.raises(WorkerStopRequired):
            await consumer.poll_once()
        assert queue.claim_next.await_count == 1
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)


async def test_stop_wakes_idle_heartbeat_without_waiting_for_renewal_interval():
    queue, claim = fixture_queue()
    queue.policy = LeasePolicy(heartbeat_seconds=15)
    queue.heartbeat = AsyncMock()
    async def handler(_):
        return PreparedJobResult({})
    consumer = SqlJobConsumer(queue, {claim.handler: handler}, 'worker')
    assert await asyncio.wait_for(consumer.poll_once(), .5)
    queue.heartbeat.assert_not_awaited()


async def test_interrupted_heartbeat_drain_fails_closed_and_observes_late_error():
    queue, claim = fixture_queue()
    async def handler(_):
        return PreparedJobResult({})
    consumer = SqlJobConsumer(queue, {claim.handler: handler}, 'worker')
    release, stop = asyncio.Event(), asyncio.Event()
    async def database_operation():
        await release.wait()
        raise ConnectionError('late database result')
    heartbeat = asyncio.create_task(database_operation())
    draining = asyncio.create_task(consumer._settle_heartbeat(heartbeat, stop))
    try:
        await stop.wait()
        draining.cancel()
        with pytest.raises(WorkerStopRequired):
            await draining
        assert not heartbeat.cancelled() and not heartbeat.done()
        with pytest.raises(WorkerStopRequired):
            await consumer.poll_once()
    finally:
        release.set()
        await asyncio.gather(draining, heartbeat, return_exceptions=True)


async def test_stop_at_renewal_boundary_does_not_open_another_database_call(monkeypatch):
    queue, claim = fixture_queue()
    queue.heartbeat = AsyncMock()
    async def handler(_):
        return PreparedJobResult({})
    consumer = SqlJobConsumer(queue, {claim.handler: handler}, 'worker')
    stop = asyncio.Event()
    async def timeout_after_stop(waiter, timeout):
        waiter.close()
        stop.set()
        raise TimeoutError()
    monkeypatch.setattr(asyncio, 'wait_for', timeout_after_stop)
    await consumer._heartbeat(claim, stop)
    queue.heartbeat.assert_not_awaited()
    await consumer._heartbeat(claim, stop)
