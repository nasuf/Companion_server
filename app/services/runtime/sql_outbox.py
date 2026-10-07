"""Acknowledgement-driven, at-least-once client delivery on the migrated schema.

Explicit use only: imports never register workers or change Redis scheduling.
A successful socket/pubsub send is NOT an acknowledgement. Replay drains unacked
records; clients deduplicate by event ID and reconcile replies by message ID.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
import json
from typing import Any

from app.services.runtime.chat_ingress import _object, _time
from app.services.runtime.chat_ingress_contracts import _identity
from app.services.runtime.execution_scope import (
    ExecutionScope,
    ExecutionScopeUnavailable,
    revalidate_scope,
    scoped_transaction,
)
from app.services.runtime.sql_job_contracts import (
    LeaseLost,
    LeasePolicy,
    WorkerStopRequired,
    _seconds,
)
from app.services.runtime.sql_job_queue import _scope, _conversation_lock, _now

_ACTIVE = "('pending','delivering')"
# Publication order (one commit batch per Run), then sequence within a batch.
# No cursor: a client ACK lost in transit leaves the event discoverable forever.
_HEAD = """NOT EXISTS (
 SELECT 1 FROM runtime_outbox earlier JOIN agent_runs er ON er.id=earlier.run_id
 WHERE er.conversation_id=r.conversation_id AND earlier.status IN ('pending','delivering')
 AND (earlier.created_at,earlier.run_id,earlier.sequence)<(o.created_at,o.run_id,o.sequence))"""


@dataclass(frozen=True, slots=True)
class ClaimedDelivery:
    id: str
    run_id: str
    scope: ExecutionScope
    worker_id: str
    token: int
    attempts: int
    sequence: int
    event_type: str
    payload_json: str

    def envelope(self) -> dict:
        return {
            "type": self.event_type,
            "data": json.loads(self.payload_json),
            "event_id": self.id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "delivery_token": str(self.token),
        }


class SqlOutbox:
    def __init__(self, database: Any, *, policy: LeasePolicy | None = None):
        self.database = database
        self.policy = policy or LeasePolicy()

    async def _row(self, tx, event_id, *, skip=False):
        rows = await tx.query_raw(
            "SELECT o.* FROM runtime_outbox o WHERE o.id=$1 FOR UPDATE"
            + (" SKIP LOCKED" if skip else ""),
            event_id,
        )
        return rows[0] if rows else None

    async def _retire(self, candidate):
        scope = _scope(candidate)
        async with self.database.tx(
            max_wait=timedelta(seconds=2), timeout=timedelta(seconds=5)
        ) as tx:
            await tx.execute_raw("SET LOCAL lock_timeout='1s'")
            await tx.execute_raw("SET LOCAL statement_timeout='2500ms'")
            await tx.query_raw(
                "SELECT id FROM users WHERE id IN ($1,$2) ORDER BY id FOR SHARE",
                scope.actor_user_id,
                scope.owner_user_id,
            )
            for table, identity in (
                ("ai_agents", scope.agent_id),
                ("chat_workspaces", scope.workspace_id),
                ("conversations", scope.conversation_id),
            ):
                await tx.query_raw(
                    f"SELECT id FROM {table} WHERE id=$1 FOR SHARE", identity
                )
            if not await _conversation_lock(tx, scope.conversation_id, skip=True):
                return
            try:
                await revalidate_scope(scope, database=tx)
            except ExecutionScopeUnavailable:
                await tx.execute_raw(
                    "UPDATE runtime_outbox SET status='cancelled',lease_owner=NULL,lease_expires_at=NULL,error='{\"code\":\"scope_unavailable\"}',updated_at=clock_timestamp() WHERE run_id=$1 AND status IN "
                    + _ACTIVE,
                    candidate["run_id"],
                )

    async def claim_next(
        self,
        worker_id: str,
        *,
        scope: ExecutionScope | None = None,
        reconnect: bool = False,
    ) -> ClaimedDelivery | None:
        _identity(worker_id)
        if type(reconnect) is not bool or (reconnect and scope is None):
            raise ValueError("Reconnect requires an authenticated bound scope")
        if scope is not None:
            await revalidate_scope(scope, database=self.database)
        candidates = await self.database.query_raw(
            "SELECT r.*,o.id AS event_id,o.run_id FROM runtime_outbox o JOIN agent_runs r ON r.id=o.run_id "
            "WHERE r.status='succeeded' AND r.parent_run_id IS NULL AND o.event_type IN ('reply','done') "
            "AND (($1::text IS NULL) OR r.conversation_id=$1) "
            "AND ((o.status='pending' AND ($2 OR o.available_at<=clock_timestamp())) "
            "OR (o.status='delivering' AND o.lease_expires_at<=clock_timestamp())) AND "
            + _HEAD
            + " ORDER BY o.created_at,o.run_id,o.sequence LIMIT $3",
            scope.conversation_id if scope else None,
            reconnect,
            self.policy.scan_limit,
        )
        for candidate in candidates:
            stored = _scope(candidate)
            # The Run actor may be an admin. Authenticate delivery against the
            # current socket actor, comparing resource generations independently.
            if scope is not None and any(
                getattr(stored, n) != getattr(scope, n)
                for n in (
                    "owner_user_id",
                    "agent_id",
                    "workspace_id",
                    "conversation_id",
                    "owner_generation",
                    "agent_generation",
                    "workspace_generation",
                    "conversation_generation",
                )
            ):
                await self._retire(candidate)
                continue
            try:
                async with scoped_transaction(stored, database=self.database) as tx:
                    if scope is not None:
                        await revalidate_scope(scope, database=tx)
                    if not await _conversation_lock(
                        tx, stored.conversation_id, skip=True
                    ):
                        continue
                    run = await tx.query_raw(
                        "SELECT status FROM agent_runs WHERE id=$1 FOR UPDATE SKIP LOCKED",
                        candidate["run_id"],
                    )
                    if not run or run[0]["status"] != "succeeded":
                        continue
                    row = await self._row(tx, candidate["event_id"], skip=True)
                    now = await _now(tx)
                    if (
                        not row
                        or row["status"] not in {"pending", "delivering"}
                        or (
                            row["status"] == "delivering"
                            and _time(row["lease_expires_at"]) > now
                        )
                        or (
                            not reconnect
                            and row["status"] == "pending"
                            and _time(row["available_at"]) > now
                        )
                    ):
                        continue
                    if not await tx.query_raw(
                        "SELECT o.id FROM runtime_outbox o JOIN agent_runs r ON r.id=o.run_id WHERE o.id=$1 AND "
                        + _HEAD,
                        row["id"],
                    ):
                        continue
                    leased = (
                        await tx.query_raw(
                            "UPDATE runtime_outbox SET status='delivering',attempts=attempts+1,fencing_token=fencing_token+1,lease_owner=$2,lease_expires_at=clock_timestamp()+$3::double precision*interval '1 second',error='{}',updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
                            row["id"],
                            worker_id,
                            self.policy.lease_seconds,
                        )
                    )[0]
                    claim = ClaimedDelivery(
                        row["id"],
                        row["run_id"],
                        stored,
                        worker_id,
                        int(leased["fencing_token"]),
                        leased["attempts"],
                        row["sequence"],
                        row["event_type"],
                        json.dumps(_object(row["payload"])),
                    )
                return claim
            except ExecutionScopeUnavailable:
                await self._retire(candidate)
        return None

    async def _owned(self, tx, claim, *, allow_delivered=False):
        runs = await tx.query_raw(
            "SELECT * FROM agent_runs WHERE id=$1 FOR UPDATE", claim.run_id
        )
        if (
            not runs
            or _scope(runs[0]) != claim.scope
            or runs[0]["status"] != "succeeded"
            or runs[0]["parent_run_id"] is not None
        ):
            raise LeaseLost()
        row = await self._row(tx, claim.id)
        if (
            row
            and row["run_id"] == claim.run_id
            and row["status"] == "delivered"
            and allow_delivered
        ):
            return row
        if (
            not row
            or row["run_id"] != claim.run_id
            or row["status"] != "delivering"
            or row["lease_owner"] != claim.worker_id
            or int(row["fencing_token"]) != claim.token
            or row["attempts"] != claim.attempts
            or _time(row["lease_expires_at"]) <= await _now(tx)
        ):
            raise LeaseLost()
        return row

    async def retry(self, claim: ClaimedDelivery, *, code: str = "ack_pending") -> bool:
        _identity(code, maximum=64)
        async with scoped_transaction(claim.scope, database=self.database) as tx:
            await _conversation_lock(tx, claim.scope.conversation_id)
            row = await self._owned(tx, claim, allow_delivered=True)
            if row["status"] == "delivered":
                return False  # A client may ACK before the sender returns.
            changed = await tx.execute_raw(
                "UPDATE runtime_outbox SET status='pending',lease_owner=NULL,lease_expires_at=NULL,available_at=clock_timestamp()+$2::double precision*interval '1 second',error=jsonb_build_object('code',$3::text),updated_at=clock_timestamp() WHERE id=$1 AND lease_expires_at>clock_timestamp()",
                claim.id,
                min(60, self.policy.retry_seconds * 2 ** min(claim.attempts - 1, 4)),
                code,
            )
            if changed != 1:
                raise LeaseLost()
            # No maximum delivery attempts: an offline client must not lose data.
            return True

    async def acknowledge(
        self, scope: ExecutionScope, event_id: str, token: str
    ) -> bool:
        _identity(event_id)
        if (
            type(token) is not str
            or not token.isascii()
            or not token.isdecimal()
            or token.startswith("0")
            or len(token) > 19
            or int(token) > 9223372036854775807
        ):
            raise ValueError("Invalid delivery token")
        async with scoped_transaction(scope, database=self.database) as tx:
            await _conversation_lock(tx, scope.conversation_id)
            runs = await tx.query_raw(
                "SELECT r.* FROM agent_runs r JOIN runtime_outbox o ON o.run_id=r.id WHERE o.id=$1 AND r.conversation_id=$2 FOR UPDATE OF r",
                event_id,
                scope.conversation_id,
            )
            if not runs:
                return False
            stored = _scope(runs[0])
            await revalidate_scope(stored, database=tx)
            row = await self._row(tx, event_id)
            if not row:
                return False
            if row["status"] == "delivered":
                return True
            if (
                row["status"] not in {"pending", "delivering"}
                or int(row["fencing_token"]) != int(token)
                or row["attempts"] < 1
            ):
                return False
            # A client receipt may arrive after timeout; only a NEW claim token
            # makes an old ACK stale. A socket send never marks delivered itself.
            await tx.execute_raw(
                "UPDATE runtime_outbox SET status='delivered',delivered_at=clock_timestamp(),lease_owner=NULL,lease_expires_at=NULL,error='{}',updated_at=clock_timestamp() WHERE id=$1",
                event_id,
            )
            return True

    async def has_pending(self, scope: ExecutionScope) -> bool:
        await revalidate_scope(scope, database=self.database)
        return bool(
            await self.database.query_raw(
                "SELECT o.id FROM runtime_outbox o JOIN agent_runs r ON r.id=o.run_id WHERE r.conversation_id=$1 AND r.status='succeeded' AND r.parent_run_id IS NULL AND o.status IN "
                + _ACTIVE
                + " LIMIT 1",
                scope.conversation_id,
            )
        )

    async def deliver_once(
        self,
        worker_id: str,
        send,
        *,
        scope: ExecutionScope | None = None,
        reconnect: bool = False,
        timeout_seconds: float = 5,
    ) -> bool:
        _seconds(timeout_seconds, 60)
        if timeout_seconds * 3 > self.policy.lease_seconds:
            raise ValueError("Send timeout must leave lease recovery margin")
        claim = await self.claim_next(worker_id, scope=scope, reconnect=reconnect)
        if claim is None:
            return False
        code = "ack_pending"
        try:
            async with scoped_transaction(claim.scope, database=self.database) as tx:
                await _conversation_lock(tx, claim.scope.conversation_id)
                await self._owned(tx, claim)
            if scope is not None:
                await revalidate_scope(scope, database=self.database)
            sending = asyncio.create_task(
                send(claim.scope.conversation_id, claim.envelope())
            )
            try:
                done, _ = await asyncio.wait({sending}, timeout=timeout_seconds)
                if not done:
                    raise TimeoutError()
                await sending
            finally:
                if not sending.done():
                    sending.cancel()
                    done, _ = await asyncio.wait(
                        {sending}, timeout=self.policy.cancellation_seconds
                    )
                    if not done:
                        sending.add_done_callback(
                            lambda t: None if t.cancelled() else t.exception()
                        )
                        raise WorkerStopRequired(
                            "Delivery ignored cancellation; stop the owning worker"
                        )
                    await asyncio.gather(sending, return_exceptions=True)
        except WorkerStopRequired:
            raise
        except asyncio.CancelledError:
            raise  # Leave lease for takeover; no premature concurrent sender.
        except ExecutionScopeUnavailable:
            raise
        except Exception:
            code = "send_unavailable"
        try:
            await self.retry(claim, code=code)
        except LeaseLost:
            pass  # Reclaimer owns it; the stable event will be redelivered.
        return True
