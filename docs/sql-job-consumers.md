# SQL primary-job consumers (R01.04, staged)

This batch implements parameterized SQL claims, leases, fencing and the explicit
consumer loop on the existing migrated runtime schema. There is no new DDL,
production handler registration or API/scheduler startup hook. Production still
uses the existing Redis scheduling and fully enabled chat graph. SQL ingress,
consumption and LangGraph checkpoint recovery remain disabled until their
respective activation gates are met. Web/Flutter event contracts are unchanged.

## Claiming and order

`SqlJobQueue` requires a bounded registry of `HandlerSpec` entries: primary job
key, payload version, Run kind, chat executor and explicit graph/state compatibility pairs.
Unknown versions are never executed under the newest handler. Only one primary
Job per root Run is supported; child Runs are excluded until ancestor cancellation
and child coordination are implemented. Multiple jobs and child-graph joins require an explicit
coordinator. Foreground/background selection is explicit, priorities ascend.
Consumer registration accepts async handlers, rejecting synchronous functions
and async generators before claiming any job. Use an explicit preparation adapter.

Candidate discovery is a bounded, unlocked read. Claims take the existing scope
SHARE locks, the same conversation advisory lock as ingress, then Run and Job
row locks with `SKIP LOCKED`. Busy conversations/rows are skipped. Chat order
comes from immutable receipt ordinals, not client timestamps, JSON state or job
priority. A later turn cannot overtake a queued/running earlier turn even when
the earlier job is delayed or its handler version is unsupported. The discovery
query excludes those later turns so a busy conversation cannot fill the scan
with blocked followers. A waiting Run releases the primary chat slot, consistent
with the storage contract; its eventual reply ordering belongs to R01.05/R03.
Already-held unknown actions do not occupy the scan budget; newly started
actions and expired deadlines still enter the reconciliation/termination checks.
A reconciled older chat must wait while any other turn holds the running slot.

Due aggregation windows are sealed inside the claim transaction. Receipt counts,
ordinals, message association and source payload must remain consistent. The
claim then updates Run/Job together, increments attempts and fencing token and
returns only after commit. A failed/ambiguous claim never grants permission to
prepare a reply. Wait for lease expiry before reclaiming an ambiguous claim.

## Authority and commits

The default lease is 60 seconds with heartbeat every 15 seconds. SQL wall clock
is authoritative; renewal is capped by the immutable Run deadline. Expired leases
cannot be renewed. Reclaim increments the token even for the same worker ID.
Every renewal and business commit checks the current server-bound resource scope,
Run/Job status, worker ID, token, lease expiry and deadline under short SQL locks.

`fenced_transaction(claim)` guards short SQL business writes and checks again at
exit. `finish(claim, result, commit=...)` executes a prepared SQL callback and
marks Job/Run succeeded in the same transaction, with a final lease/deadline CAS.
Callback exceptions, lease loss, cancellation and lifecycle invalidation roll
back all writes. Never call providers/models, use the global DB, stream replies
or nest transactions inside these blocks. Fencing does not make external effects
exactly once: R03 must provide action-specific idempotency and reconciliation.

Claimed scope and fencing fields are internal state, not client credentials.
Never build them from a request, model result or checkpoint. Invalid resource
scopes cannot perform business commits. The queue may cancel stale internal
Run/Job records only after locking and revalidating their persisted resources;
it does not use stale authority to write messages, quota or other business data.

## Recovery, cancellation and budgets

Attempts include the first execution, with the immutable Job maximum. An expired
lease or handler failure may retry only if its registered handler explicitly
declares `retry_safe=True`; the default is false. Retry keeps input and snapshots,
uses bounded backoff, and retains FIFO order. Exhausted attempts or deadlines
terminate the Run/Job. Cancellation clears leases; terminal results cannot revive.

An unresolved `started`/`unknown` external Action is converted/kept as unknown and
the Run waits for reconciliation. It is neither automatically re-executed nor
reported as a successful reply. `finish` returns False without invoking its
business callback in that case. Known provider outcomes must be durably recorded
before preparing final reply commits. Ordinary retries do not reconcile them.

The consumer caps preparation by the handler's execution time budget and the Run
deadline. It stops preparing on renewal failure, cancellation or shutdown. Local
authority is revoked before cancelling a handler, blocking late SQL commits.
A handler that ignores cancellation forces `WorkerStopRequired`: the supervisor
must stop that worker process, rather than poll another job in the same process.
No lease is released while work may still run. SQL failures stop the polling loop;
there is no Redis fallback or duplicate submission. Per-model/token/tool budgets,
checkpoint compatibility and interrupted graph resume remain R02 requirements.

## Activation and verification

R01.05 must supply reply/Outbox transactions and reconnect delivery; R01.06 must
wire explicit process roles/health/supervision; R01.07 must hand over Redis jobs
and define rollback after SQL activation. R01.03.8 must bind real endpoint/domain
idempotency. These are activation dependencies, not accomplished by importing the
queue. Do not register existing unfenced streaming handlers as if they already
satisfy this contract. A Redis-only rollback is still valid for this staged batch;
retain the additive tables and previous migration history.

CI covers strict contracts and real migrated loopback PostgreSQL: competing
workers, SKIP LOCKED and scan fairness, receipt FIFO, collecting windows, renewal,
expired takeover, old-worker rejection, retries/unknown effects, deadlines,
scope reset, cancellation, write rollback and ambiguous commit acknowledgement.
CI also kills a disposable worker process group in six commit windows, including
its Prisma engine, and verifies rollback, takeover fencing and one stored reply.
Consumer tests exercise live heartbeat, timeout, DB failure, stop events and
cancellation refusal. Existing authenticated chat protocol and two-worker gates
remain mandatory. Production acceptance is read-only; process-kill rehearsals
use disposable containers and synthetic databases only.
