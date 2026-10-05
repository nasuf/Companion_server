# Runtime job queue: atomic transitions and leases

The queue currently runs `agent_initialization`. Chat turn execution and the
LangGraph selector are unchanged. No database schema or prompt changes are
required. Existing Redis keys, payloads, status names and admin actions remain
compatible; new fields are optional for older readers.

## State and ownership

| Transition | Atomic result |
| --- | --- |
| Enqueue | Record, seven-day TTL, ready/delayed index and optional idempotency binding |
| Claim | Remove ready duplicates, acquire the existing lock name, increment attempt and record running lease |
| Renew | Compare attempt and random lease token; extend lock, record and running deadline |
| Finish | Compare live lease ownership; update result and its indexes; release only the owner's lock |
| Promote | Remove a due delayed entry and add one ready entry |
| Recover | After lock/lease expiry, requeue once or dead-letter an exhausted attempt budget |

All transitions use fixed Lua scripts with all accessed keys supplied explicitly.
Key types are checked before writes, because a Lua runtime error does not undo
earlier writes. Redis `TIME` controls scheduling and lease decisions. Each claim
has a 60-second lease, renewed every 15 seconds with a five-second renewal timeout.
An expired lease cannot renew or commit. Cancellation or renewal failure cancels
the handler and leaves its running record for recovery. Handlers must cooperate
with cancellation; a request already sent to an external provider may finish.

Completion and recovery do not extend the original seven-day retention. Running
indexes hold lease deadlines; legacy records without a token retain the original
15-minute stale threshold and cannot recover while their old lock is present.
Manual retry/resolve retains N04's state conflicts and cumulative attempt count;
manual retry permits a new attempt even when the previous automatic budget ended.

Initialization also holds a separate memory-generation lock for 30 minutes.
Its registration therefore adds a 30-minute recovery grace after the runtime
lease expires, allowing that domain lock to expire before another attempt.
The policy is persisted in new records; registration supplies it for legacy
records. The greater of the stored and registered policy wins. Recovery uses
the delayed index and records `not_before`; manual retry cannot bypass it.
Other handlers default to immediate recovery after their runtime lease expires.
This grace handles abandoned locks, not partial business effects: initialization
still needs the R03 work on step idempotency and internal error propagation.

## Old and interrupted data

Each scheduler tick scans one bounded page, retains oversized SCAN pages for later
ticks and restores missing indexes without changing payloads, terminal states or
TTLs. Recovery pages rotate so live legacy locks cannot hide later expired leases.
`not_before` protects delayed jobs even if the delayed index is lost while a stale
ready entry remains. Unknown legacy delays are held for operator review. Only the
audited `agent_initialization` registration allows a missing legacy delay to mean
immediate execution: its old enqueue caller never supplied a delay.

An idempotency binding without a matching record raises `RuntimeJobOrphaned`.
An error obtaining an EVAL result raises `RuntimeJobEnqueueUncertain`, because
the server may already have committed. Neither condition starts a local replay;
initialization progress exposes a failed state and a reconciliation message only
while still queued. This update compares the stage atomically, so a late enqueue
error cannot overwrite a worker's started or completed progress.
Redis failure before EVAL retains the existing local fallback.

For these errors, inspect the task, initialization progress and actual agent data
before deciding to retry. With an intact binding, a repeat enqueue using the same
business idempotency key returns the original record. A missing record cannot be
restored from the binding alone. Do not delete a binding or recreate a payload
until actual effects have been checked. No automatic bulk replay or Redis reset
is part of this release. Wrong global index types fail closed and require repair.

## Admin diagnostics

`GET /admin-api/runtime-jobs/diagnostics` requires an admin JWT and is read-only.
Parameters are `cursor`, `idempotency_cursor` and `limit` (1–200). Follow both
returned cursors; `truncated=true` means SCAN exceeded the requested page size,
so the response cannot establish a complete inventory. Repeat with a larger limit
or use a bounded operator inventory before declaring the queue clean. SCAN is a
best-effort inventory during concurrent changes, not a transaction over the queue.

Diagnostics report missing indexes, unverified legacy delays and missing bound
records without exposing payloads, idempotency key names or ownership tokens.
Job detail also returns optional queue version, lease state/deadline, heartbeat,
recovery count, recovery grace and recorded due time. Redis outages return 503. The Web workspace
keeps its existing UI; these added fields do not create a new UI panel.
Counts are raw index sizes: expired records can leave history entries, so counts
are not guaranteed to equal the number of retained detail records.

## Release, rollback and limits

Deploy only after an inventory confirms no old running initialization jobs; stop
old workers before starting this revision. A rolling mixture of old and new
workers is not a qualified release mode. Rollback needs the same quiescence and
stopped-worker boundary. Existing hashes/indexes remain readable by the previous
version; no migration, TTL reset or Redis clearing is needed. The previous version
does not provide renewable leases, and rolling back reintroduces its known gaps.

The queue provides recoverable, at-least-once execution while its retained Redis
records survive. A crash after a business effect but before completion may invoke
the handler again. Queue result fencing does not make external effects exactly
once; handlers need their own stable business idempotency. Production currently
uses shared Redis with `allkeys-lru`; eviction, total Redis loss and retention
expiry remain R01/R03 work. This batch does not alter that production policy.

Tests use random isolated Redis namespaces and synthetic effects. They cover
concurrent creators/claimers, renewal loss, stale results, legacy compatibility,
manual conflicts, delayed index loss, and real subprocess SIGKILL before/after
enqueue, claim, effect, completion and promotion. Fault workers reject production
mode and remote Redis before connecting. They do not call models or production DB.
