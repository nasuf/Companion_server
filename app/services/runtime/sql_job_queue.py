"""Staged SQL primary-job queue. Importing it never starts a consumer.

One primary job per Run is supported here. Child graphs and multi-job joins need
their own explicit coordination, rather than silently completing a shared Run.
All claims/commits take scope -> conversation -> Run -> Job locks. Provider/model
I/O belongs outside these transactions; direct streaming/business writes bypass
the fence and are forbidden for handlers registered with this queue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from dataclasses import fields
from typing import Any

from app.services.runtime.chat_ingress import (
    ChatIngressCorrupt,
    _ingress,
    _object,
    _seal,
    _time,
)
from app.services.runtime.chat_ingress_contracts import (
    MAX_TURN_MESSAGES,
    _identity,
    canonical_object,
)
from app.services.runtime.execution_scope import (
    ExecutionScope,
    ExecutionScopeUnavailable,
    revalidate_scope,
    scoped_transaction,
)
from app.services.runtime.sql_job_contracts import (
    ClaimedJob,
    HandlerSpec,
    LeaseLost,
    LeasePolicy,
)

_SCOPE_NAMES = tuple(field.name for field in fields(ExecutionScope))
_ACTIVE = "('pending','retry','running')"
_TERMINAL = {"succeeded", "failed", "cancelled"}


def _scope(row: dict) -> ExecutionScope:
    return ExecutionScope(**{name: row[name] for name in _SCOPE_NAMES})


async def _conversation_lock(
    tx: Any, conversation_id: str, *, skip: bool = False
) -> bool:
    fn = "pg_try_advisory_xact_lock" if skip else "pg_advisory_xact_lock"
    value = (
        await tx.query_raw(
            f"SELECT {fn}(hashtextextended($1,0))::text AS held",
            "chat-ingress:" + conversation_id,
        )
    )[0]["held"]
    return not skip or value == "true"


async def _now(tx: Any):
    return _time((await tx.query_raw("SELECT clock_timestamp() AS now"))[0]["now"])


async def _rows(tx: Any, run_id: str, job_id: str, *, skip: bool = False):
    # Separate locks establish Run before Job rather than relying on join order.
    suffix = " FOR UPDATE" + (" SKIP LOCKED" if skip else "")
    runs = await tx.query_raw("SELECT * FROM agent_runs WHERE id=$1" + suffix, run_id)
    if not runs:
        return None
    jobs = await tx.query_raw(
        "SELECT * FROM runtime_jobs WHERE id=$1 AND run_id=$2" + suffix, job_id, run_id
    )
    return (runs[0], jobs[0]) if jobs else None


async def _terminal(tx: Any, run: dict, status: str, error: dict) -> None:
    encoded = canonical_object(error)
    await tx.execute_raw(
        "UPDATE runtime_jobs SET status=$2,error=$3::jsonb,lease_owner=NULL,"
        "lease_expires_at=NULL,finished_at=clock_timestamp(),updated_at=clock_timestamp() "
        "WHERE run_id=$1 AND status IN " + _ACTIVE,
        run["id"],
        status,
        encoded,
    )
    await tx.execute_raw(
        "UPDATE agent_runs SET status=$2,error=$3::jsonb,"
        "finished_at=clock_timestamp(),updated_at=clock_timestamp() "
        "WHERE id=$1 AND status IN ('queued','running','waiting')",
        run["id"],
        status,
        encoded,
    )


class SqlJobQueue:
    def __init__(
        self,
        database: Any,
        specs: Sequence[HandlerSpec],
        *,
        policy: LeasePolicy | None = None,
    ):
        self.database = database
        self.policy = policy or LeasePolicy()
        if (
            not specs
            or len(specs) > 32
            or any(not isinstance(s, HandlerSpec) for s in specs)
        ):
            raise ValueError("Explicit bounded handler registry required")
        self.specs = tuple(specs)
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate handler registration")

    def _compatible(self, run: dict, job: dict) -> HandlerSpec | None:
        spec = self._specs.get(job["handler"])
        if (
            spec is None
            or run["parent_run_id"] is not None
            or job["job_key"] != spec.job_key
            or run["kind"] != spec.kind
            or job["payload_version"] != spec.payload_version
            or (run["graph_version"], run["state_version"]) not in spec.versions
        ):
            return None
        if (
            run["kind"] == "chat"
            and _object(run["state"]).get("ingress", {}).get("executor")
            != spec.executor
        ):
            return None
        return spec

    async def _candidates(self, queue: str) -> list[dict]:
        clauses, params = [], [queue]
        for spec in self.specs:
            for graph, state in spec.versions:
                start = len(params) + 1
                clauses.append(
                    "(j.handler=$%d AND j.job_key=$%d AND j.payload_version=$%d "
                    "AND r.kind=$%d AND r.graph_version=$%d AND r.state_version=$%d "
                    "AND (r.kind<>'chat' OR r.state #>> '{ingress,executor}'=$%d))"
                    % tuple(range(start, start + 7))
                )
                params.extend(
                    (
                        spec.name,
                        spec.job_key,
                        spec.payload_version,
                        spec.kind,
                        graph,
                        state,
                        spec.executor,
                    )
                )
        params.append(self.policy.scan_limit)
        return await self.database.query_raw(
            "SELECT r.*,j.id AS job_id FROM runtime_jobs j JOIN agent_runs r ON r.id=j.run_id "
            "WHERE j.queue=$1 AND r.parent_run_id IS NULL AND r.status IN ('queued','running','waiting') "
            "AND ((j.status IN ('pending','retry') AND j.available_at<=clock_timestamp()) "
            "OR (j.status='running' AND j.lease_expires_at<=clock_timestamp()) "
            "OR r.deadline_at<=clock_timestamp()) "
            "AND (" + " OR ".join(clauses) + ") "
            "AND (r.deadline_at<=clock_timestamp() OR NOT (r.status='waiting' "
            "AND j.status='retry' AND COALESCE(j.error->>'code','')='reconciliation_required' "
            "AND EXISTS (SELECT 1 FROM agent_actions a WHERE a.run_id=r.id AND a.status='unknown') "
            "AND NOT EXISTS (SELECT 1 FROM agent_actions a WHERE a.run_id=r.id AND a.status='started'))) "
            "AND (r.kind<>'chat' OR NOT EXISTS (SELECT 1 FROM agent_runs earlier "
            "WHERE earlier.conversation_id=r.conversation_id AND earlier.id<>r.id "
            "AND earlier.kind='chat' AND earlier.status IN ('queued','running') "
            "AND (earlier.status='running' OR (SELECT min(ordinal) FROM chat_ingress_receipts WHERE run_id=earlier.id) < "
            "(SELECT min(ordinal) FROM chat_ingress_receipts WHERE run_id=r.id) "
            "OR NOT EXISTS (SELECT 1 FROM chat_ingress_receipts WHERE run_id=earlier.id)))) "
            "ORDER BY j.priority,j.available_at,j.id LIMIT $" + str(len(params)),
            *params,
        )

    async def _retire_invalid_scope(self, candidate: dict) -> None:
        """Only invalidate internal runtime records, never write business data.

        Invalid/inactive scopes cannot enter scoped_transaction. Lock existing
        resource rows in the same parent-before-conversation order, revalidate
        inside that transaction, and cancel only if authority is still invalid.
        Deletion already cascades; no stored payload can authorize execution.
        """
        scope = _scope(candidate)
        async with self.database.tx(
            max_wait=timedelta(seconds=2), timeout=timedelta(seconds=5)
        ) as tx:
            await tx.execute_raw("SET LOCAL lock_timeout = '1s'")
            await tx.execute_raw("SET LOCAL statement_timeout = '2500ms'")
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
            pair = await _rows(tx, candidate["id"], candidate["job_id"], skip=True)
            if not pair or pair[0]["status"] in _TERMINAL:
                return
            try:
                await revalidate_scope(_scope(pair[0]), database=tx)
            except ExecutionScopeUnavailable:
                await _terminal(tx, pair[0], "cancelled", {"code": "scope_unavailable"})

    async def _earlier_chat(self, tx: Any, run: dict) -> bool:
        if run["kind"] != "chat":
            return False
        # Acceptance ordinal is durable server order, not client/created_at time.
        ordinal = (
            await tx.query_raw(
                "SELECT min(ordinal) AS ordinal FROM chat_ingress_receipts WHERE run_id=$1",
                run["id"],
            )
        )[0]["ordinal"]
        rows = await tx.query_raw(
            "SELECT id FROM agent_runs WHERE conversation_id=$1 AND id<>$2 "
            "AND kind='chat' AND status IN ('queued','running') "
            "AND (status='running' OR (SELECT min(ordinal) FROM chat_ingress_receipts WHERE run_id=agent_runs.id) < $3 "
            "OR NOT EXISTS (SELECT 1 FROM chat_ingress_receipts WHERE run_id=agent_runs.id)) LIMIT 1",
            run["conversation_id"],
            run["id"],
            ordinal,
        )
        return bool(rows)

    async def _validate_chat(self, tx: Any, run: dict, job: dict) -> None:
        ingress = _ingress(run)
        source = await tx.query_raw(
            "SELECT count(*)::int AS n,min(q.ordinal) AS first,max(q.ordinal) AS last,"
            "bool_and(m.conversation_id=$2 AND m.role='user') AS valid "
            "FROM chat_ingress_receipts q LEFT JOIN messages m ON m.id=q.message_id WHERE q.run_id=$1",
            run["id"],
            run["conversation_id"],
        )
        row = source[0]
        payload = _object(job["payload"])
        if (
            not 1 <= row["n"] <= MAX_TURN_MESSAGES
            or any(
                type(ingress.get(key)) is not int
                for key in ("message_count", "first_ordinal", "last_ordinal")
            )
            or row["n"] != ingress.get("message_count")
            or row["first"] != ingress.get("first_ordinal")
            or row["last"] != ingress.get("last_ordinal")
            or row["valid"] is not True
            or type(payload.get("schema_version")) is not int
            or payload
            != {
                "schema_version": 1,
                "run_id": run["id"],
                "input_source": "chat_ingress_receipts",
            }
        ):
            raise ChatIngressCorrupt()

    async def _hold_unknown(self, tx: Any, run: dict, job: dict) -> bool:
        unknown = await tx.query_raw(
            "SELECT status FROM agent_actions WHERE run_id=$1 "
            "AND status IN ('started','unknown') ORDER BY (status='started') DESC LIMIT 1",
            run["id"],
        )
        if not unknown:
            return False
        if (
            run["status"] == "waiting"
            and job["status"] == "retry"
            and _object(job["error"]).get("code") == "reconciliation_required"
            and unknown[0]["status"] == "unknown"
        ):
            return True
        await tx.execute_raw(
            "UPDATE agent_actions SET status='unknown',updated_at=clock_timestamp() "
            "WHERE run_id=$1 AND status='started'",
            run["id"],
        )
        await tx.execute_raw(
            "UPDATE runtime_jobs SET status='retry',lease_owner=NULL,lease_expires_at=NULL,"
            "error=$2::jsonb,updated_at=clock_timestamp() WHERE id=$1",
            job["id"],
            canonical_object({"code": "reconciliation_required"}),
        )
        await tx.execute_raw(
            "UPDATE agent_runs SET status='waiting',updated_at=clock_timestamp() WHERE id=$1",
            run["id"],
        )
        return True

    async def claim_next(
        self, worker_id: str, *, queue: str = "foreground"
    ) -> ClaimedJob | None:
        _identity(worker_id)
        if queue not in {"foreground", "background"}:
            raise ValueError("Unsupported queue")
        for candidate in await self._candidates(queue):
            scope = _scope(candidate)
            try:
                async with scoped_transaction(scope, database=self.database) as tx:
                    if not await _conversation_lock(
                        tx, scope.conversation_id, skip=True
                    ):
                        continue
                    pair = await _rows(
                        tx, candidate["id"], candidate["job_id"], skip=True
                    )
                    if not pair:
                        continue
                    run, job = pair
                    now = await _now(tx)
                    spec = self._compatible(run, job)
                    if (
                        not spec
                        or job["queue"] != queue
                        or run["status"] in _TERMINAL
                        or _scope(run) != scope
                        or job["status"] not in {"pending", "retry", "running"}
                    ):
                        continue
                    if run["kind"] == "chat":
                        try:
                            await self._validate_chat(tx, run, job)
                        except ChatIngressCorrupt:
                            await _terminal(
                                tx, run, "failed", {"code": "input_unavailable"}
                            )
                            continue
                    if job["status"] == "running":
                        if _time(job["lease_expires_at"]) > now and (
                            not run["deadline_at"] or _time(run["deadline_at"]) > now
                        ):
                            continue
                    elif _time(job["available_at"]) > now and (
                        not run["deadline_at"] or _time(run["deadline_at"]) > now
                    ):
                        continue
                    if run["deadline_at"] and _time(run["deadline_at"]) <= now:
                        await _terminal(
                            tx, run, "failed", {"code": "deadline_exceeded"}
                        )
                        continue
                    # A future multi-job Run must supply a coordinator, not use
                    # this primary-job algorithm to terminate unrelated work.
                    if (
                        await tx.query_raw(
                            "SELECT count(*)::int AS n FROM runtime_jobs WHERE run_id=$1",
                            run["id"],
                        )
                    )[0]["n"] != 1:
                        continue
                    if await self._hold_unknown(tx, run, job):
                        continue
                    if job["attempts"] >= job["max_attempts"]:
                        await _terminal(
                            tx, run, "failed", {"code": "attempts_exhausted"}
                        )
                        continue
                    if job["attempts"] and not spec.retry_safe:
                        await _terminal(
                            tx, run, "failed", {"code": "retry_not_authorized"}
                        )
                        continue
                    if await self._earlier_chat(tx, run):
                        continue
                    if run["kind"] == "chat":
                        ingress = _ingress(run)
                        if ingress["phase"] == "collecting":
                            if (
                                job["status"] != "pending"
                                or job["attempts"] != 0
                                or _time(ingress["window_due_at"]) > now
                            ):
                                continue
                            await _seal(
                                tx,
                                {**run, "job_id": job["id"]},
                                _time(ingress["window_due_at"]),
                            )
                    await tx.execute_raw(
                        "UPDATE agent_runs SET status='running',error='{}',updated_at=clock_timestamp() WHERE id=$1",
                        run["id"],
                    )
                    leases = await tx.query_raw(
                        "UPDATE runtime_jobs SET status='running',attempts=attempts+1,"
                        "fencing_token=fencing_token+1,lease_owner=$2,"
                        "lease_expires_at=LEAST(clock_timestamp()+$3::double precision*interval '1 second',"
                        "COALESCE($4::timestamptz,'infinity'::timestamptz)),error='{}',updated_at=clock_timestamp() "
                        "WHERE id=$1 AND ($4::timestamptz IS NULL OR $4::timestamptz>clock_timestamp()) RETURNING *",
                        job["id"],
                        worker_id,
                        self.policy.lease_seconds,
                        (
                            _time(run["deadline_at"]).isoformat()
                            if run["deadline_at"]
                            else None
                        ),
                    )
                    if not leases:
                        raise LeaseLost()
                    leased = leases[0]
                    claim = ClaimedJob(
                        job["id"],
                        run["id"],
                        scope,
                        spec.name,
                        worker_id,
                        int(leased["fencing_token"]),
                        leased["attempts"],
                        _time(leased["lease_expires_at"]),
                        _time(run["deadline_at"]) if run["deadline_at"] else None,
                        canonical_object(_object(job["payload"])),
                        canonical_object(_object(run["config_snapshot"])),
                        canonical_object(_object(run["prompt_snapshot"])),
                        canonical_object(_object(run["budget_snapshot"])),
                        run["graph_version"],
                        run["state_version"],
                    )
                return claim  # Only expose authority after the claim commits.
            except ExecutionScopeUnavailable:
                await self._retire_invalid_scope(candidate)
            except LeaseLost:
                # Deadline crossed during the short claim; rollback both rows.
                # A subsequent scan terminates it without preparing any reply.
                continue
        return None

    async def _owned(self, tx: Any, claim: ClaimedJob) -> tuple[dict, dict]:
        claim.require_active()
        pair = await _rows(tx, claim.run_id, claim.id)
        if not pair:
            raise LeaseLost()
        run, job = pair
        now = await _now(tx)
        if (
            run["status"] != "running"
            or _scope(run) != claim.scope
            or job["status"] != "running"
            or job["lease_owner"] != claim.worker_id
            or int(job["fencing_token"]) != claim.fencing_token
            or job["handler"] != claim.handler
            or job["attempts"] != claim.attempts
            or run["graph_version"] != claim.graph_version
            or run["state_version"] != claim.state_version
            or self._compatible(run, job) is None
            or _time(job["lease_expires_at"]) <= now
            or (run["deadline_at"] and _time(run["deadline_at"]) <= now)
        ):
            raise LeaseLost()
        return run, job

    @asynccontextmanager
    async def fenced_transaction(self, claim: ClaimedJob) -> AsyncIterator[Any]:
        """Short business commit fence; never model/provider I/O or lease-state edits."""
        claim.require_active()
        async with scoped_transaction(claim.scope, database=self.database) as tx:
            await _conversation_lock(tx, claim.scope.conversation_id)
            await self._owned(tx, claim)
            yield tx
            await self._owned(tx, claim)  # Wall clock/deadline must still hold at exit.

    async def heartbeat(self, claim: ClaimedJob) -> None:
        async with self.fenced_transaction(claim) as tx:
            await tx.execute_raw(
                "UPDATE runtime_jobs SET lease_expires_at=LEAST("
                "clock_timestamp()+$2::double precision*interval '1 second',"
                "COALESCE($3::timestamptz,'infinity'::timestamptz)),updated_at=clock_timestamp() WHERE id=$1",
                claim.id,
                self.policy.lease_seconds,
                claim.deadline_at.isoformat() if claim.deadline_at else None,
            )

    async def finish(self, claim: ClaimedJob, result: dict, *, commit=None) -> bool:
        """Commit prepared SQL business writes + Job/Run result together.

        commit(tx) is SQL-only and bounded by scoped_transaction. R01.05 will
        supply reply+Outbox writes here; this release has no such adapter.
        Returns False without invoking commit if an unresolved action requires
        reconciliation. Record known provider outcomes before final reply commit.
        """
        encoded = canonical_object(result)
        claim.require_active()
        async with scoped_transaction(claim.scope, database=self.database) as tx:
            await _conversation_lock(tx, claim.scope.conversation_id)
            run, job = await self._owned(tx, claim)
            if await self._hold_unknown(tx, run, job):
                return False
            if commit is not None:
                await commit(tx)
            await self._owned(tx, claim)
            changed = await tx.execute_raw(
                "UPDATE runtime_jobs SET status='succeeded',result=$2::jsonb,lease_owner=NULL,"
                "lease_expires_at=NULL,finished_at=clock_timestamp(),updated_at=clock_timestamp() "
                "WHERE id=$1 AND status='running' AND lease_owner=$3 AND fencing_token=$4 "
                "AND lease_expires_at>clock_timestamp() AND ($5::timestamptz IS NULL OR $5::timestamptz>clock_timestamp())",
                claim.id,
                encoded,
                claim.worker_id,
                claim.fencing_token,
                claim.deadline_at.isoformat() if claim.deadline_at else None,
            )
            if changed != 1:
                raise LeaseLost()
            await tx.execute_raw(
                "UPDATE agent_runs SET status='succeeded',result=$2::jsonb,"
                "finished_at=clock_timestamp(),updated_at=clock_timestamp() WHERE id=$1",
                claim.run_id,
                encoded,
            )
        return True

    async def fail(
        self, claim: ClaimedJob, *, code: str, retryable: bool = False
    ) -> None:
        _identity(code, maximum=64)
        if type(retryable) is not bool:
            raise ValueError("Invalid retry decision")
        claim.require_active()
        async with scoped_transaction(claim.scope, database=self.database) as tx:
            await _conversation_lock(tx, claim.scope.conversation_id)
            run, job = await self._owned(tx, claim)
            if await self._hold_unknown(tx, run, job):
                return
            if (
                retryable
                and self._specs[claim.handler].retry_safe
                and job["attempts"] < job["max_attempts"]
            ):
                await tx.execute_raw(
                    "UPDATE runtime_jobs SET status='retry',lease_owner=NULL,lease_expires_at=NULL,"
                    "available_at=clock_timestamp()+$2::double precision*interval '1 second',error=$3::jsonb,"
                    "updated_at=clock_timestamp() WHERE id=$1",
                    claim.id,
                    self.policy.retry_seconds,
                    canonical_object({"code": code}),
                )
                await tx.execute_raw(
                    "UPDATE agent_runs SET status='queued',error=$2::jsonb,updated_at=clock_timestamp() WHERE id=$1",
                    claim.run_id,
                    canonical_object({"code": code}),
                )
            else:
                await _terminal(tx, run, "failed", {"code": code})

    async def cancel_run(self, scope: ExecutionScope, run_id: str) -> bool:
        async with scoped_transaction(scope, database=self.database) as tx:
            await _conversation_lock(tx, scope.conversation_id)
            rows = await tx.query_raw(
                "SELECT * FROM agent_runs WHERE id=$1 AND conversation_id=$2 FOR UPDATE",
                run_id,
                scope.conversation_id,
            )
            if not rows or _scope(rows[0]) != scope:
                raise ExecutionScopeUnavailable()
            if rows[0]["status"] in _TERMINAL:
                return False
            await _terminal(tx, rows[0], "cancelled", {"code": "cancelled"})
        return True
