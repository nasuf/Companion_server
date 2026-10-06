# C01: prompt and configuration consistency

The database is the authority. A prompt write locks its key, validates the
optional revision/timestamp guard, and commits content, revision and audit in one
transaction. Redis is updated only after commit. A failed cache write is reported
as pending; it never reverses the saved database version. Publishing holds a DB
row lock and checks row identity/revision, then uses a Lua fence. This also covers
Redis eviction and deletion/recreation of a row. Old clients may omit guards;
the Web workspace and Trace editor send both guards for every mutation.

No prompt wording changes in this release. Startup checks full history and sticky
Web ownership. Manual saves, restores, reset-default, unknown operations,
unrecorded customization and incomplete bootstrap history are protected. A
bootstrap must cover creation within one minute (old seeders used separate
transactions). Bootstrap/code_sync and enable/disable alone do not constitute a
Web text edit. Existing versions are never rewritten; orphan audit rows are retained
and still protect Web ownership if the same registry key is recreated.

Database triggers advance the model configuration revision for system settings,
agent overrides and model registry writes, including other workers and scripts.
Async entrypoints check the DB; synchronous consumers refresh at most two seconds
later. Delivery does not depend on Pub/Sub. A failed refresh retains the last
complete snapshot and remains retryable.

A logical turn captures models, prices and prompt contents in a repeatable-read
transaction. Nested fragments reuse that view only for the same agent. New turns
load the latest committed configuration. Prompt disable is checked live before
rendering; enabling a previously disabled template takes effect next turn. Already
sent model requests cannot be recalled. Model factories cache by provider/model,
so an old turn cannot poison a newer turn's cache. Embedding selection is unchanged.

Migration `20261006000000_config_consistency` is additive. Historical version
revisions stay NULL; the UI does not invent a historical revision. The old Prisma
client remains able to read/write after the migration. Application rollback leaves
new columns and triggers in place; restoring DB data is unnecessary.

Release gates: isolated PG/Redis concurrency/failure/restart tests, prompt rendering
and existing chat tests, real HTTP/WS/Nginx tests in legacy/allowlist/full modes,
Web build/API/browser checks including Trace single-step replay, migration upgrade
with historical rows and old-client compatibility, and two reviews per repository.
Production acceptance must compare all pre-release prompt content/enabled/history
records, cache state and generated-client fields, and confirm healthy two-worker
LangGraph operation. Persistent checkpoint replay remains disabled.

The existing production deployment still stops the serving container before the
replacement starts. C01 does not claim zero downtime; that remains the separate
recorded deployment workstream.
