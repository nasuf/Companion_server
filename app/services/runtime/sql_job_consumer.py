"""Explicit, bounded SQL consumer; never booted by API/scheduler imports.

Handlers return prepared results and an optional short SQL commit callback.
They must not stream replies or perform unguarded business/provider effects.
External actions require the R03 action protocol; reply/Outbox adapters are R01.05.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from app.services.runtime.execution_scope import ExecutionScopeUnavailable
from app.services.runtime.sql_job_contracts import (
    ClaimedJob,
    LeaseLost,
    WorkerStopRequired,
)
from app.services.runtime.sql_job_queue import SqlJobQueue


@dataclass(frozen=True, slots=True)
class PreparedJobResult:
    result: dict
    commit: Callable[[Any], Awaitable[None]] | None = None


Handler = Callable[[ClaimedJob], Awaitable[PreparedJobResult]]


class SqlJobConsumer:
    def __init__(
        self, queue: SqlJobQueue, handlers: Mapping[str, Handler], worker_id: str
    ):
        if set(handlers) != {s.name for s in queue.specs} or any(
            not (
                inspect.iscoroutinefunction(f)
                or inspect.iscoroutinefunction(getattr(f, "__call__", None))
            )
            for f in handlers.values()
        ):
            raise ValueError("Handler registry and claim compatibility must agree")
        self.queue = queue
        self.handlers = MappingProxyType(dict(handlers))
        self.worker_id = worker_id
        self._polling = False
        self._must_stop = False

    async def _heartbeat(self, claim: ClaimedJob) -> None:
        while True:
            await asyncio.sleep(self.queue.policy.heartbeat_seconds)
            await self.queue.heartbeat(claim)

    async def _stop_handler(self, task: asyncio.Task, claim: ClaimedJob) -> None:
        claim.revoke()  # Late SQL commits cannot use this local authority.
        task.cancel()
        done, _ = await asyncio.wait(
            {task}, timeout=self.queue.policy.cancellation_seconds
        )
        if not done:
            self._must_stop = True
            # Observe a later failure without pretending the handler stopped.
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            raise WorkerStopRequired(
                "Handler ignored cancellation; stop the owning worker"
            )
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def poll_once(self, *, queue: str = "foreground") -> bool:
        """One primary job at a time; DB errors propagate without Redis fallback."""
        if self._polling or self._must_stop:
            raise WorkerStopRequired("Consumer is already busy or must stop")
        self._polling = True
        try:
            claim = await self.queue.claim_next(self.worker_id, queue=queue)
            if claim is None:
                return False
            spec = next(s for s in self.queue.specs if s.name == claim.handler)
            timeout = spec.max_execution_seconds
            if claim.deadline_at is not None:
                timeout = min(
                    timeout,
                    (claim.deadline_at - datetime.now(timezone.utc)).total_seconds(),
                )
            if timeout <= 0:
                claim.revoke()
                return True
            # SQL clock is authoritative at all fences. Local time only bounds
            # preparation; clock skew cannot grant extra lease/commit authority.
            execution = asyncio.create_task(self.handlers[claim.handler](claim))
            heartbeat = asyncio.create_task(self._heartbeat(claim))
            try:
                done, _ = await asyncio.wait(
                    {execution, heartbeat},
                    timeout=max(0, timeout),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat in done:
                    await self._stop_handler(execution, claim)
                    await heartbeat  # Surface renewal DB/scope/lease failures.
                if execution not in done:
                    # Revoke and leave the lease to expire. Requeue only after
                    # fencing/reconciliation checks; never while work still runs.
                    await self._stop_handler(execution, claim)
                    return True
                try:
                    prepared = execution.result()
                    if not isinstance(prepared, PreparedJobResult):
                        raise ValueError("Handler must return a prepared SQL result")
                except Exception:
                    await self.queue.fail(claim, code="handler_failed", retryable=True)
                    return True
                await self.queue.finish(claim, prepared.result, commit=prepared.commit)
                return True
            except WorkerStopRequired:
                raise
            except (LeaseLost, ExecutionScopeUnavailable):
                await self._stop_handler(execution, claim)
                return True
            except BaseException:
                await self._stop_handler(execution, claim)
                raise
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
                claim.revoke()
        finally:
            self._polling = False

    async def run(
        self, stop: asyncio.Event, *, queue: str = "foreground", poll_seconds: float = 1
    ) -> None:
        if type(poll_seconds) not in {int, float} or not 0 < poll_seconds <= 60:
            raise ValueError("Invalid poll interval")
        while not stop.is_set():
            polling = asyncio.create_task(self.poll_once(queue=queue))
            stopping = asyncio.create_task(stop.wait())
            try:
                done, _ = await asyncio.wait(
                    {polling, stopping}, return_when=asyncio.FIRST_COMPLETED
                )
                if stopping in done:
                    polling.cancel()
                    result = await asyncio.gather(polling, return_exceptions=True)
                    if isinstance(result[0], WorkerStopRequired):
                        raise result[0]
                    return
                await polling
            finally:
                stopping.cancel()
                if not polling.done():
                    polling.cancel()
                await asyncio.gather(stopping, polling, return_exceptions=True)
            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
                except TimeoutError:
                    pass
