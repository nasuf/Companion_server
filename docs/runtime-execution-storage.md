# Durable execution storage (R01.02)

This release adds `AgentRun`, `RuntimeJob`, `AgentAction` and `RuntimeOutbox`
through Prisma migration `20261007013000_runtime_execution_foundation`.
It does **not** activate SQL enqueueing, SQL consumers, delivery workers or a
LangGraph checkpointer. Existing Redis scheduling and client frames are unchanged.

## Client generation and memory

The four runtime models and their new inverse fields use Prisma `@@ignore` /
`@ignore` for **client generation only**. Tables, indexes and foreign keys remain
Prisma migration-managed; a schema diff with/without those attributes produces
no DDL. SQL workers will use parameterized SQL for `SKIP LOCKED`, lease and CAS
operations rather than recursive ORM relations. Existing ORM models/query depth
are preserved. This avoids adding the new graph to the legacy recursive Python
client (a measured ~288 MiB additional import peak per process before isolation).
Tests write runtime records through a test-only SQL fixture on real Prisma
transactions; it is not an application repository or an authorization API.
Runtime SQL writers must provide stable IDs and update timestamps explicitly.

## Identity and transaction boundary

An AgentRun is one logical execution, not an entire conversation. Its request key
is unique within the conversation. Store the original canonical input alongside
its SHA-256 fingerprint: uniqueness alone does not distinguish a legitimate retry
from conflicting content; R01.03 must compare both before returning an existing run.

Actor, owner, agent, workspace, conversation and their generations are fixed.
Foreign keys enforce existence, **not authorization** or current ownership. All
future creation and business commits must use the server-bound ExecutionScope
and `scoped_transaction` described in [execution-scope.md](execution-scope.md).
Never decode a client/model-provided scope as authorization. Lifecycle changes
invalidate stored generations; they do not rewrite historical Run snapshots.
This first schema requires a conversation; agent-only maintenance/provisioning
will need its own explicit server scope before integration.

Graph/state versions, input, config/prompt/budget snapshots and the deadline are
immutable. R02 must capture real versioned snapshots and validate compatibility;
this schema does not supply them, establish checkpoint recovery, or validate
arbitrary snapshot contents. Never store secrets, raw media or live clients here.
Child runs retain the complete parent scope, including generations. Parent IDs
cannot be changed; deleting a parent removes its children. Only one `running`
chat run may occupy a conversation. `waiting` releases that slot; task/background
runs can overlap. Claiming and resumption order are still R01.04 responsibilities.

## Storage contracts

| Record | Stable uniqueness | Mutable execution state |
| --- | --- | --- |
| Run | conversation + request key | status, state, result, error, finish time |
| Job | run + job key | status, attempts, lease/fencing, due time, result/error |
| Action | run + action key; global idempotency key | status, provider reference, result/error, start/finish time |
| Outbox | run + event key; run + sequence; primary ID | status, attempts, lease/fencing, due/delivery time, error |

Jobs have foreground/background queues and ascending numeric priority; smaller
numbers run first. `max_attempts` defaults to 3 **total executions**, including the
first attempt. A running job/delivering event requires a nonempty worker, expiry,
positive attempt and fencing token. Other states must clear both lease fields.
Attempt/fencing counters cannot decrease. These checks do not yet implement
`SKIP LOCKED`, lease ownership CAS, expiry reclamation or stale-worker rejection;
R01.04/R01.05 must add and test those algorithms. Default lease/heartbeat policy
remains 60s/15s in the roadmap, not an implicit database default.

An Action in `unknown` may only be reconciled, completed, failed or cancelled; it
cannot transition back to planned/started automatically. An idempotency key does
not by itself guarantee an external provider honours it. R03 must handle provider
lookup, result-unknown cases and source-business idempotency across runs.

Outbox IDs are stable delivery event IDs. Payload and message association are
immutable. An associated message must belong to the Run's conversation at insert.
Message writers must preserve conversation ownership (R01.05 integration); this
insert guard does not fence unconverted writers that move existing messages.
Outbox acknowledgement means delivery, not user consumption. Delivery may repeat;
R01.05 must implement client deduplication and reconnect catch-up. Sequence is
per Run; it does not establish ordering across the conversation by itself.

All JSON state/payload/snapshot columns require objects. Terminal Run/Job/Action
records and delivered/failed/cancelled Outbox records cannot be edited or revived.
Idempotent callers must read and return them rather than call update on them
(Prisma's `updatedAt` is itself a change). Recovery uses a new explicitly identified
operation after reconciliation, not a silent reset of terminal records. Physical
resource deletion cascades to the new storage; archival and reset cancellation
remain R03 work and must revalidate generations before any business commit.

## Migration and rollback

The migration is additive and transactional, with bounded lock/statement timeouts.
No existing columns, indexes, rows, prompts or old migration files are changed.
Existing generated clients keep reading/writing old tables. Roll back application
code by using the verified previous artifact **and retain the expanded schema**;
do not DROP these tables. Once SQL ingress/consumers are enabled later, a rollback
to a Redis-only application will need a different explicit handover boundary.

CHECK constraints, the partial chat index and snapshot/association triggers are
managed only by Prisma migrations. Prisma's schema language does not represent
all of them: `db push` or runtime auto-DDL is not a supported deployment method.
Tests use only a named disposable loopback DB, mock providers and synthetic data.
Migration verification includes empty install, historic upgrade, repeated deploy,
old generated-client read/write and synthetic backup restore. No production test
records or prompt publications are part of this release.
