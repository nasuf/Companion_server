# Transactional chat ingress (R01.03, staged)

`chat_ingress.accept_chat_message` commits a user message, immutable input receipt
and its Run/Job in one short, server-scoped PostgreSQL transaction. Multiple
receipts may share a Run/Job when they belong to one aggregation window.
Migration `20261007060000_chat_ingress_receipts` adds only the receipt table,
sequence, association guards and indexes; historical migrations are unchanged.
The model and inverse relations remain client-ignored to preserve the old ORM.

## Deployment boundary

The HTTP and ticket-authenticated WebSocket entrypoints now contain staged SQL
adapters. `CHAT_INGRESS_BACKEND` defaults to `redis`; requesting `sql` fails at API
startup and the shared request gate while `SQL_CHAT_ACTIVATION_READY` is false.
Production does **not** call the SQL writer, claim SQL jobs, deliver an Outbox
automatically or enable checkpoints.
Current Redis scheduling and graph rollout remain unchanged. SQL ingress must
not be enabled before R01.04-.07 provide consumers, delivery, role isolation and
the explicit Redis handover/rollback boundary. No shadow jobs are submitted:
recording pending SQL jobs while also executing Redis jobs would risk duplicate
replies when SQL consumers start later.

Endpoint/domain integration remains a separate activation requirement within
R01.03: preserve quota/payment rules, validate and bind media/link/offering rows,
and publish ack/pending only after commit. Existing Redis aggregation planning
mutates Redis while flushing; it must not run before a SQL transaction as if it
were a pure planner. Reminder, achievement, location and proactive hooks also
need durable source identity; this repository does not execute those hooks.

### R01.03.8.1: atomic domain effects (staged)

The optional server-only `ChatIngressEffects` adapter now commits actual quota
and wallet writes, attachment bindings and red-packet/gift bindings with the
message/receipt/Run/Job. Receipt deduplication under the conversation lock occurs
before these effects. Same-ID retries return the original message without
charging or binding again; conflicting content fails before any effects.
Every later write failure rolls back the entire acceptance, including payment
ledgers. A lost commit acknowledgement is recovered with the same request ID.

VIP status is read from SQL for the scoped owner. Payment confirmation is a
strict boolean; it is intentionally outside immutable request identity so a
rejected, uncharged draft can be resubmitted after confirmation. Usage is counted
per original message, before aggregation. Gift/red-packet cards are exempt only
after checking the authoritative offering owner, agent, conversation, kind,
`sent` state and unbound message under its row lock. Preparation metadata must
match the server-issued card. Their previous purchase is not charged again;
this adapter does not implement purchase, receipt or refund operations.

All attachment IDs must be unbound resources of the scoped owner/conversation,
and the prepared metadata preserves their original order. All requested bindings
must succeed. No model, vision, provider call, nested transaction or legacy
offering background hook runs during acceptance. Prepared text/metadata/context
still come from trusted preparation services; they are not request schemas.
Committed message metadata records the effect version and quota outcome, and a
paid ledger records that message's ID as its audit source.

The production Redis endpoints do **not** use this adapter yet. Their legacy quota wrapper
reuses the same transaction helper, with existing prices and periods; unexpected
wallet/ledger failures now propagate to roll back instead of being reported as
insufficient funds. The legacy separation between charging and saving a chat
message remains until the qualified topology activates atomic acceptance.

Link, music, location and other cards fail closed in this staged adapter until
their dedicated domain effects and durable follow-ups are implemented. The
legacy endpoints continue supporting them. This release must not enable SQL
ingress/consumers: R01.03.8.2-.6 and R01.06.7/.8, R01.07/.08 remain required.
An omitted adapter retains the storage-only API for foundation tests; SQL
endpoint adapters must explicitly supply it and must not retry through that
storage-only path on error. No schema or prompt content changes in this batch.

### R01.03.8.2: authenticated endpoint adapters (staged)

The shared receiver parses original text/card, ordered attachment IDs (maximum
three, without truncation or deduplication) and mandatory stable `client_id`.
HTTP accepts an optional `client_id` for existing Redis clients; it becomes
mandatory on the SQL path. HTTP and WS share `client:<client_id>` so a retry
can change transport without creating a second source. Original whitespace is
part of retry identity; separately prepared text uses the chat normalization.
Payment confirmation must be a JSON boolean. It is outside request identity so
an unaccepted draft can be confirmed and retried with the original ID. Legacy
WS also rejects strings/numbers instead of treating `"false"` as paid consent.

Each SQL request binds current actor/owner/agent/workspace/conversation from SQL,
using the authenticated JWT or consumed socket-ticket principal. Payload fields
cannot impersonate an owner, select a backend or supply a scope. Existing
terminal receipts return current status without starting another job. Duplicate
lookup precedes provider preparation; if an identical concurrent commit binds a
resource during preparation, the receiver looks up its receipt again. The final
transaction still owns deduplication, quota authorization and resource binding.

Trusted preparation reads owned attachments and server-issued offerings, renders
their model input, captures reply timing and chooses aggregation policy without
calling Redis execution-queue reads/flush/enqueue operations. Pending deletion
and contradiction checks retain the existing read-only bypass rules. SQL decides
window membership/order. Complete inputs close a collecting fragment window;
ordinary turn-window timings use the same 1.2s/4s constants, fragments use 5s,
and offerings bypass aggregation/delay.

Preparation binds a complete configuration/prompt view, serializes resolved
models/options, prices, effective prompt content/enabled flags/content hashes,
executor/graph/state versions and timeout budgets. Environment credentials and
raw SystemConfig rows are excluded. The first accepted turn keeps its captured
snapshot; capture remains bounded by existing JSON limits. Applying these values
to a real SQL foreground worker and replay is still R01.06.7/R02 work. Media
analysis/cache and daily schedule preparation occur outside the SQL transaction;
their provider-call deduplication is not promised by the business receipt.

Only a successful transaction exit produces an `ack`, followed by `pending`
when execution is nonterminal. ACK means accepted persistence, not successful
reply generation; it never exposes stored result/error details. UI defer/timer
fields preserve the existing collecting/delayed behavior. HTTP closes its SSE
request with an `acceptance_only` done event; durable reply delivery is separate.
Request conflicts, invalid resources, quota blocks and unavailable storage
produce distinct errors, with no ACK, Redis execution or legacy fallback. A
lost commit/socket ACK must be retried with the **same** source ID.

The official-account H5 entry uses the same authenticated HTTP/WS transport;
there is no independent provider chat-message callback to migrate in this repo.
The internal `from_wechat` contract does not expose an unsigned MsgId route.
Links, music, location and other cards remain refused on this staged SQL path
until R01.03.8.3 supplies their domain effects/follow-ups. Existing Redis clients
retain their current behavior. R01.03.8.3-.6, R01.06.7/.8 and R01.07/.08 still
block SQL activation; this batch changes neither prompts nor database schema.

## Trusted contracts and retries

- Bind `ExecutionScope` from authenticated actor + conversation on the server.
  A serialized scope, provider payload or model output is not authorization.
- `ChatRequestInput` captures original text, ordered attachment IDs and card
  input as strict canonical JSON. JSON object key order is irrelevant; text and
  attachment order are preserved. Requests are unique by conversation + key.
- WebSocket/HTTP share `client:<client_id>`. WeChat uses `wechat:<verified MsgId>`.
  Missing IDs are rejected by this staged API; legacy endpoints are unaffected.
  Do not manufacture a new ID for each retry or trust an unsigned provider ID.
- `PreparedChatMessage` contains separately rendered text, domain-validated
  metadata/context and a timezone-aware server receipt time. Construction does
  not authorize attachments, offerings or a quota charge.
- The first Run fixes executor/graph/state/config/prompt/budget/deadline values.
  Capture real values before transaction; never store secrets, clients or raw
  media. No model or provider I/O is permitted inside `scoped_transaction`.
- Same identity + identical original body returns original IDs, current status
  and stored result/error, without rewriting input, rendering or snapshots.
  Different body raises `ChatRequestConflict`; changed generations fail closed.
  Terminal retries return stored state without restarting a job or replaying replies.
- `lookup_chat_request` permits an early duplicate check; a missing result is
  **not** a reservation. Moving quota or offering mutations after this lookup
  alone would still allow concurrent duplicate charges. The final adapter must
  include their idempotent reservation/transaction semantics before activation.

Only return/publish an accepted receipt after the transaction context exits.
A commit acknowledgement may be lost after a successful commit; retry with the
same ID to discover the result. DB errors never fall back to an unguarded write
or a parallel Redis execution.

## Aggregation and ordering

Each source receipt is immutable and retains original/prepared input. Run input
is the **first original request**, not a rewritten aggregate. Run state tracks
the open window, policy and source counters; `load_chat_turn` reconstructs the
ordered execution input from receipts. Physical source deletion fails closed.

The DB-generated receipt ordinal defines **server acceptance order**, independent
of client timestamps. Gaps from rolled-back inserts are normal. It establishes
durable input order, but SQL job execution ordering still belongs to R01.04.

The caller supplies the existing domain decision: fragment/normal/immediate,
quiet window, maximum wait, delay and whether joining is allowed. Fragment joins
use direct concatenation; normal turns use the existing read-only query coalescer
and newline joins. A fragment arriving during a normal window retains that
normal policy; a complete input following fragments closes their window.
Bypass inputs do not join a normal window. Card/urgent decisions can explicitly
disallow joining. This module does not change intent classification prompts.

Joining preserves the first policy, snapshot and receipt context, refreshing
only latest receipt/emotion fields. Retries do not refresh windows. Expired
windows close before a new input can join. Different executor/graph/state versions
start a new turn; current configuration changes cannot rewrite an open turn.
Maximum input count is 32; a conservative 32,768-character aggregate budget
reserves newline separators. Reaching either boundary closes the old turn and
starts another without deleting source inputs. Unschedulable deadlines reject
the whole transaction; a rejected append leaves the previous window unchanged.

`seal_due_chat_turn` is idempotent; it does not claim or execute the job. Delay is
added to the window due time. Jobs contain an immutable Run reference, not a
mutable copy of the merged payload. `load_chat_turn` is a scoped read, **not** a
lease or authority to execute side effects.

## Locking, lifecycle and rollback

The lock order is scoped parent SHARE locks -> conversation advisory lock -> Run
-> Job -> source message/receipt. Acceptance/sealing are serialized per
conversation. Locks last only for the bounded transaction: 1s lock timeout,
2.5s statement budget, 5s transaction budget. Consumers must acquire the same
conversation lock, close collecting windows before claiming, enforce due/order,
and reject stale worker commits using the R01.04 lease/fencing protocol.

Association triggers require a user message and a pending/unattempted chat Job
in the same conversation/Run. Closed Runs reject additional receipts. These
checks prove association, not resource authority. Resource deletion cascades;
archive/reset invalidate generations. Full task cancellation and preserving
message ownership against unconverted writers remain R03/R01.05 requirements.

Before SQL activation, the prior application can run with the expanded schema;
keep added tables when rolling back. After activation, a Redis-only rollback
requires handover and cannot be inferred from this additive migration.

## Verification

CI runs strict input/snapshot contracts and migrated loopback PostgreSQL tests:
cross-client retry/conflict, ordered aggregation and expiry, deadline/bounds,
every-write rollback, lost commit acknowledgement, stale scopes, raw-writer
immutability/association, lock timeout and source deletion. Migration rehearsal
uses synthetic historic data, old generated-client reads/writes and backup restore.
Existing authenticated HTTP/WS/Redis graph E2E and two-worker startup gates remain
mandatory. Test cases must never use application DATABASE_URL or production data.
