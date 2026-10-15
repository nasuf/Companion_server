# Location shares and memory

A location card records a point shared in one message. It does not prove the
user's residence, current location on later days, or the Agent's location.
The event time is the server's message acceptance time, not a verified GPS fix.
Existing residence memories and historical messages are not rewritten or
backfilled by this change.

## Admission and recovery

On the existing Redis WebSocket path, location admission uses the existing
execution-scope and quota transaction helpers. A conversation advisory lock
serializes retries; the client ID and canonical request fingerprint are checked
before quota is consumed. Message, attachment binding, quota and the location
outbox record commit or roll back together. Reusing a client ID with changed
content is rejected. This does not activate SQL chat execution or checkpoints;
the existing reply-queue acceptance gap remains a separate roadmap item.

The `messages` INSERT trigger creates `location_memory_events` only for new user
location cards with a valid owner/workspace/Agent scope and no active human
takeover. Redis receives only the message ID, using a stable idempotency key.
The runtime queue scan dispatches at most ten due events after consuming normal
jobs; on the first Redis failure it retains the pending record with a 30-second
backoff. A database dispatch failure cannot prevent other Redis job consumption.

The `location_share_memory` handler prepares two embeddings outside SQL locks,
then rechecks the original content, full card, acceptance time and frozen scope.
An active takeover, deletion or source change cancels the event. Both L3
interaction memories, vectors, audit records, message/input evidence links and
completion receipt commit in one transaction. The immutable `location_share`
input snapshot preserves the accepted card. Shared coordinates do not enter the
permanent memory text. This is deterministic event recording, not LLM extraction.

Completed/cancelled receipts cannot be reopened. A repeated job never recreates
a deleted derived memory. The message foreign key removes its event on message
deletion. Redis dead letters can use the existing administrator retry operation;
missing/expired Redis job records are shown explicitly and are not automatically
reported as success or silently recreated.

## Operations

Web administration exposes a collapsed **位置分享记忆处理记录** panel in the
existing task queue. It reads only when opened, shows the newest 20 events (API
maximum 50), and distinguishes pending dispatch, failed/dead-letter, missing or
unavailable queue state, completed and cancelled. The sample is not a global
success-rate denominator. A recent-order index and due-dispatch partial index
bound these reads. The API requires an administrator JWT and returns IDs,
statuses, attempt counts and timestamps, without coordinates or address text.

Memory source details reuse existing scoped evidence permissions. An input link
means the submitted version is recorded, not that its content or coordinates
were independently verified. Legacy unknown origins remain unknown.

Flutter retries retain the location card, attachments and original client ID.
Service-account H5 displays structured location cards in history and realtime
messages. Small-program code is outside this batch.

## Validation boundaries

Local tests use disposable loopback PostgreSQL/Redis. Failure/replay cases cover
both-side atomicity, lost dispatch responses, dead-letter retry, deletion,
source mutation, takeover, duplicate quota and attachment rollback. A real
HTTP-ticket/WebSocket/history/administrator-API test exercises persisted recovery;
only external providers and unrelated background hooks are fixed by the fixture.
Browser and Flutter widget tests exercise the actual application components.
They do not certify a physical device installation or external model outputs.
