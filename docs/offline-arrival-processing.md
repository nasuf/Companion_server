# Arrival response and durable follow-up

Arrival validation and the HTTP response schema are unchanged. The arrival SQL
transaction explicitly and atomically registers `offline_arrival_followups`; the API no longer
waits for the arrival card, LLM, speech synthesis or delivery. This applies to
existing iOS and Android clients without a new API parameter.

Both the integrated runtime scheduler and the independent background process
dispatch bounded batches through the existing Redis job queue. SQL retains
pending/failed receipts across Redis loss. Each generation
is fenced against the original user, agent, workspace, conversation and arrival;
archived, expired, deleted or moved sources are cancelled. Stable message IDs
make retries idempotent. Voice assets bind in the same transaction as their
message. Notification registration uses the existing message-ID deduplication.
WebSocket delivery may repeat the same ID after recovery; clients already merge
by message ID. Completion means messages were persisted and realtime delivery
was attempted; it does not mean a phone has received or read the message.

Observability decision: reuse the Web runtime-job recovery panel. The arrival
section fetches at most 20 records on expansion/refresh, supports status filters
and retrying failed receipts, and reports sample time, attempt count, generation
and sanitized error class. API access requires an administrator. Responses omit
coordinates, message text, prompts and the stored scope. Completed/cancelled
receipts retain 30 days; cleanup is capped at 100 per scan.

Deploy the Prisma migration before the new server process. Only the new server registers
receipts; no trigger/backfill registers old synchronous or historical arrivals.
This prevents duplicate sends during deployment overlap. On rollback, retain
the additive table and pending receipts; resume/recover them with the new
consumer after roll-forward. Existing reached activities remain idempotent.

Validation covers real PostgreSQL and Redis registration/rollback, HTTP response
before delivery, stale generations/scopes, queue loss, duplicate workers,
partial-delivery retries, voice binding/fallback, validation failures and admin
permissions. Flutter tests preserve the iOS best-accuracy/15-second GPS policy,
10-second reverse geocoding and existing profile-sync behavior. Android alone
uses foreground prewarming, 15-second recent-fix reuse, parallel system provider
requests with one 8-second native deadline, and coordinate-only arrival checks.
Chat can preview approximate positions with the existing accuracy label;
verified arrival still requires at most 100-metre accuracy on the server.
