# Reply commits and client Outbox (R01.05, staged)

`prepare_chat_result` captures a bounded immutable reply batch and returns the
SQL-only callback accepted by `SqlJobQueue.finish`. Assistant messages, their
reply events, one `done` event and Run/Job success commit together. The prepared callback is bound to its exact
originating claim; it cannot be reused under a different Run or reclaimed lease.
UUIDs derive
from Run and event key, so a retry/ambiguous commit has the same IDs. All existing
scope, lease, cancellation, unknown-action and deadline fences still apply.
Never invoke the callback directly or stream prepared output before commit.
There is no new DDL, prompt publication or production handler registration.
The old streaming chat handler is not a SQL preparation adapter: application
side effects/background work require idempotent domain jobs at R01.03.8 activation.

`SqlOutbox` claims short leases with SKIP LOCKED and conversation serialization.
A bounded socket send runs outside SQL locks. A process crash leaves the lease
for takeover; send failure or absent client ACK schedules bounded backoff. There
is deliberately no attempt maximum that could discard an offline user's reply.
Expired/stale senders cannot change delivery state; a new claim increments the
token even when the worker ID is reused. Importing this module boots no worker.
R01.06 must register an explicit delivery role and its health/supervision.

Client frames remain `ack`, `pending`, `reply`, `done`. Durable reply/done frames
add **top-level** `event_id`, `run_id`, `sequence` and decimal-string
`delivery_token`. Their `data.message_id` identifies the committed assistant
message. The new client sends `delivery_ack` with event ID/token only after an
active subscriber accepts the event. Sending a frame or Redis publication is
not delivery proof. A late ACK for the current token is valid after its timeout;
an ACK from an older reclaimed attempt is ignored. ACKs are scoped by the
current ticket-authenticated actor and conversation; IDs never grant authority.

`delivery_resume` returns at most one event and `delivery_status` per request.
The Web client drains unacknowledged records at a bounded two-second pace and
checks again on keepalive. ACK lost in transit means replay of the same ID.
No timestamp/high-water cursor can skip an older transaction that committed
late. Publication batches share SQL commit-preparation time and are serialized
by conversation; events within a batch follow sequence. Earlier pending events
block later batches until acknowledged/cancelled. An older waiting Run that
publishes later receives a later batch position. This does not provide an
external provider's exactly-once guarantee or checkpoint continuation.

Web deduplicates events across socket reconnects, clears that cache on account/
conversation changes, rejects identity/payload inconsistencies and re-ACKs new
attempt tokens. The cache is bounded; persisted message ID reconciliation also
prevents duplicate bubbles when history loading overlaps replay or after reload.
An ACK means client delivery, not durable user consumption across devices.
An event acknowledged on one device is obtained on another through message
history. Browser reload is safe because the message exists before the event.
Non-Web client durable ACK adapters must be completed before activating SQL
responses for those entry points; legacy Flutter/WeChat traffic stays on Redis.

All replay/ACK frames bind current SQL scope. Reset/archive/rebinding invalidates
old generations; stale internal events are cancelled without business writes.
Current resource authority is checked before sends, but an already handed-off
network frame cannot be revoked; socket/account identity guards remain required.
Application rollback retains the expanded schema. After durable ingress starts,
R01.07 must prohibit reverting to an unfenced Redis-only consumer.

Verification uses a migrated synthetic local DB and mock senders, including
transaction rollback, competing deliverers, lease takeover, send timeout, late/
stale ACKs, cancellation, scope reset, cross-conversation isolation and replay.
Production checks are read-only; no synthetic chat/kill test runs against prod.
