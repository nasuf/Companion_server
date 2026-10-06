# R01 execution scope foundation

This batch implements the R01.01 foundation. It does not switch the current chat
executor, queue, delivery or checkpoint. Existing writers do not yet universally
use these fences. Integration belongs to R01.03–R01.07 and deletion cancellation
belongs to R03. The full G03 business/quality/actual rollback gates remain open.

`bind_conversation_scope` takes the actor ID from verified authentication or a
trusted internal entry point and the conversation ID from the requested resource.
One SQL snapshot binds actor, actual resource owner, Agent, workspace and conversation.
All ownership edges must agree and resources must be active/unarchived. Admin
permission is read from the actor's current database role; an admin acts on the
owner's resources, never changes their owner to the admin. There is no default
workspace, cross-workspace fallback or lookup by client-supplied owner/Agent IDs.
Provisioning and archived browsing are not covered by this conversation scope.

The frozen scope can be serialized as internal schema version 1. It is not an
auth token and must never be accepted from clients or LLM/tool output. Future Run
records bind it on the server. Unknown/malformed records are rejected. Resume
must revalidate the saved scope, never bind a new one to make stale work pass.

Migration `20261006160000_execution_scope_generations` adds a server-only random
generation to users, Agents, workspaces and conversations. Database triggers rotate
on relevant ownership, lifecycle and actor role changes, including archive→restore.
Old clients need not know the new columns. Insert always generates a fresh identity,
so delete/recreate of the same resource ID cannot resurrect a historical execution.
Unrelated profile, title and timestamp updates do not rotate it. Explicit runtime
reset can rotate `execution_generation` in its transaction; consumers must use this
before destructive cleanup. No existing audit or prompt versions are rewritten.

Before preparation/resume call `revalidate_scope`. Before a SQL business commit use
`async with scoped_transaction(scope, database=database) as tx`, then write only
through `tx`. The helper starts its own short transaction, locks the current parent
rows with FOR SHARE and compares all generations. Lock timeout is 1s, statement
budget 2.5s, transaction budget 5s. Do not call models, external services or network
I/O, or nest transactions in this block. Lock/DB errors propagate and all writes
roll back; callers must not retry without revalidation or fall back to an unguarded
write. Shared parent locks permit parallel fenced readers; unrelated writes can
wait briefly. Lock conflicts/deadlocks fail closed, rather than silently completing
stale work. FOR KEY SHARE is insufficient because it permits lifecycle updates.

These guarantees cover converted SQL transactions only. External side effects,
writer lease fencing, idempotency, runtime resets before cleanup and cancellation
of already-running work require the subsequent Run/Action/Outbox integration.
A checkpoint is not a business idempotency mechanism.

## Validation and release

The CI gate runs unit contracts and real disposable PostgreSQL tests, including
admin demotion, mixed-owner foreign keys, lifecycle restore, same-ID recreation,
legacy writers, transaction rollback, real row locks, deletion during lock wait,
and bounded lock timeout. Test connections require a named loopback test database.
Migration rehearsal covers empty schema, upgrade from the previous migrations,
repeat deploy, concurrent old writers and old application read/write compatibility.
The migration is transactional with bounded DDL waits. Do not edit historical
migrations, use db push, or execute fault tests against production.

Before deployment capture lifecycle/ownership/status and generation absence from
production using read-only aggregates; after Prisma migrate deploy verify the
new migration checksum, complete unique generations, triggers, unchanged previous
schema/data hashes, source/image consistency, DB/Redis health and existing executor.
Do not drop the added columns to roll back the application: the previous release
can run with them present. This project currently stops its old container before
starting the new one; this foundation does not eliminate that deployment gap.
