# Memory source evidence: first delivery of M02.02

This delivery records where a memory came from. A recorded origin is **not**
proof that the extracted claim is true. Assistant messages remain generated
content; their statements do not become verified business receipts.

## Delivered scope

- Chat extraction pins persisted input messages before calling the model and
  checks their content hashes, role, owner, workspace and Agent again when
  writing. Every split output receives its own links. Missing message IDs are
  explicitly unlinked. Replaying the same origin/version is idempotent.
- A shared adapter supports same-scope parent memories and validated AI template
  copies. Automatic derived inference is unchanged and still waits for M02.04.
- Memory creation and source binding use one transaction. Reconciliation merges
  include the vector in that transaction; name replacement also archives the
  previous name and stores the new vector atomically. Failure preserves the
  prior content/vector/name. Cache versions advance after commit.
- The additive Prisma migration creates `memory_evidence_links`, immutable
  origin snapshots, scope guards, explicit user/AI target FKs and source FKs.
  Source deletion nulls live pointers; target deletion cascades its links.
  It does not rewrite historical text, levels, scores or prompts. The Prisma
  model is client-ignored, preserving the existing Python generated graph.

## Administration

The Agent memory list and memory-repair evidence modal expose **查看来源**.
Requests happen on expansion; each page replaces the previous page. The panel
distinguishes missing historical records, a current content version without an
origin, changed/deleted/unavailable sources, and a failed request. Raw source
text is never returned. References are redacted when a source is inaccessible.

Both routes require an administrator JWT and validate the supplied owner and
workspace against persisted data:

```text
GET /admin-api/memory-repairs/evidence/{user|ai}/{memory_id}
    ?user_id=...&workspace_id=...&limit=20&cursor=...
GET /admin-api/memory-repairs/evidence-audit
    ?side=user&user_id=...&workspace_id=...&limit=100&after_id=...
```

Detail limits are 1–50; audit limits are 1–500. Cursors are keyset-based. Audit
counts refer only to the returned page and the current content/Agent scope:
`linked` means a recorded message/parent/profile/input origin exists, including a subsequently
deleted origin. It does not mean the source is currently available or the claim
is verified. `unknown` is the complement within `checked`. The response includes
the denominator, definition and sampling time. Unknown scopes return 404 rather
than a misleading zero. The audit endpoint never backfills or writes data.

Content/source SHA-256 values identify text versions. `extractor_version` is the
server processing protocol revision; it is **not** a Web prompt publication
version or an exact model/prompt snapshot.

## Legacy read-only compatibility (M02.02 follow-up)

Detail responses add a separate optional `legacy` preview. The original snapshot
`state` and audit counts are unchanged. Explicit `evidence_linked` changelogs are
inspected under the target owner/workspace, newest 20 logs and at most 50 unique
message IDs. Payloads are limited to 16,000 characters; malformed or truncated
records remain incomplete. Counters describe this bounded sample, not all
historical evidence. A full multi-page controller remains M02.02.05.

The preview checks current message owner, workspace, Agent, role, deletion and
creation time. Same-ID memories in both sides of the same scope are ambiguous
because old logs did not store a side. Missing/inaccessible/ambiguous sources
have their message/conversation IDs redacted. A recreated target or source cannot
inherit a pre-creation association. No raw message text is returned.

Even an accessible legacy reference stays **旧版来源待核验**: neither the message
version at extraction nor the linked memory content version was recorded. It
never becomes a new snapshot, verified fact or confidence update. This route
performs no writes, backfill, retrieval changes or prompt changes.

## Remaining original roadmap work

- M02.02.03: durable tool/business receipts and exact model/prompt snapshots.
- M02.02.04: remaining corrections, repair merge/replacement actions, daily
  summaries, compression, location, offerings, offline/game and reminder adapters.
  Their uncollected provenance must remain unknown. Clone, knowledge publication,
  document import and manual edit deliveries are described below.
- M02.02.05: a multi-page audit controller with rate/stop limits. No inferred
  historical backfill is authorized by this first delivery.
- M02.02.06: each remaining writer's independent regression qualification.
- M02.04/M02.06/O01: derived deletion barriers, retention, aggregate diagnostics
  and permission-checked navigation to source messages/Trace.

This first delivery does not complete all of M02.02. Old applications remain
compatible with the additive schema; reverting application code leaves recorded
evidence intact, and subsequent uninstrumented writes remain explicitly unknown.

## Verification boundaries

Real PostgreSQL tests cover source/target scope, side collisions, role/hash
validation, immutability, deletion, idempotence, split outputs and atomic failure.
Migration tests populate 154/155 baselines, upgrade and redeploy through Prisma,
and read unchanged memory data with the existing generated client. Chromium tests
exercise real HTTP/JWT routes and PostgreSQL with synthetic fixtures. Web's
separate Chromium suite exercises the production React components, pagination,
desktop completeness, errors and stale-request isolation using controlled API
responses. Neither is a claim of complete production UI/model E2E coverage.


## Persona initialization follow-up (M02.02 PROFILE01)

Initialization now stores one immutable, scope-bound `memory_profile_origins`
snapshot for the actual profile and career consumed by conversion. Its SHA-256
also covers the invocation inputs when recorded. Generated, imported/postprocessed
and directly provided profiles are distinguished. These are fictional character
settings, not real-world experiences. An imported profile does not certify the
original document. Exact model/prompt receipts and internal random/repair steps
are still uncollected (M02.02.03); a processing revision is not a Web prompt version.

The snapshot is limited to 256 KiB and deduplicated by content version within the
owner/workspace/Agent scope. Every resulting memory gets an independent link.
Raw profile/input text is not returned by the admin API. The existing panel shows
source kind, availability and whether invocation inputs were recorded. Deleted
or inaccessible sources hide their reference and profile metadata. Old memories
are not inferred or backfilled. Direct batch writes without a real profile remain
explicitly unlinked.

Prisma migration 157 is additive and preserves old client compatibility. It
adds immutable payload/scope guards and nullable live profile references. The
legacy dependency cleanup trigger now preserves shared ID-only vectors/audit
while an opposite-side memory survives. Memory, vector, audit, snapshot, source
links and forced replacement commit together under a workspace lock. Embedding
runs before the transaction; entity seeding/cache invalidation follow commit.
Failure preserves the previous memories and sources. This does not claim all
Agent profile fields or the full initialization job share that transaction.

Snapshots cascade on owner/workspace/Agent deletion; regenerations may retain
previous snapshots. A global retention/capacity policy remains M02.16. Schema
rollback is unnecessary for application rollback; do not drop evidence tables.
The initialized snapshot is provenance for the batch input, not a per-field
explanation of deterministic fallbacks or the model's consistency repair.

Qualification adds empty/populated 156→157 Prisma upgrades, repeat deploy,
lock-timeout rollback/retry in a disposable DB, actual PostgreSQL initialization
writes, 250-row transaction timing, same-ID side cleanup, partial failures,
concurrency and scope guards. Real HTTP/JWT/PostgreSQL Chromium tests and separate
actual React controlled-response tests cover the admin display. Production
qualification is read-only; a natural post-release initialization is a separate
positive sample and must not be claimed before one exists.

### Template clone origins

New clones bind each copied AI memory to the exact current source memory version
with `memory/template_copy` and `template-clone-v1`. This records the copy event;
it does not reconstruct the template's original creation history or verify facts.
Existing clones remain unchanged. Source editing, deletion or ownership changes
are shown separately; no template text or owner identity is returned by the API.

Only active, enrolling Agents owned by the reserved template system user with a
matching active workspace can be copied. Up to 1000 nonarchived AI memories are
read, across all existing tiers; over-limit or empty templates fail explicitly.
Memory tiers/importance/provenance are preserved; usage score is not copied and
the new decay clock starts at the copy. Archived, user-side and foreign-owner
rows are excluded. Existing missing template vectors remain missing; actual SQL
copy errors fail the transaction instead of producing a partial clone.

Persona creation, workspace switching, copied memories/vectors, new audit/source
links and conversation creation share a 30-second SQL transaction. A per-user
advisory lock and default-signup recheck prevent duplicate automatic clones
when Redis is unavailable. The shared template-persona lock serializes copies
against force initialization, while allowing concurrent copies for other users.
Voice, entity seeding, old runtime cleanup and first-day scheduling run after
commit and retain their best-effort behavior. Their failure does not undo the
committed clone; durable retry/aggregate diagnostics remain existing O01/M02.06
follow-ups. This is not an atomic migration of all runtime/Redis state.

### Knowledge publication to existing clones

Knowledge sync now records `memory/template_copy` with processing revision
`knowledge-sync-v1`. Web shows **模板知识发布** separately from initial template
cloning. This pins the copied template knowledge text version; it does not certify
the original imported document or factual correctness. Old publications remain
unknown; the following INPUT01 delivery adds document and manual edit inputs.

Each target batch commits memories, available vectors, changelogs and origins
in one transaction. Any write failure rolls back that target and is reported by
the existing sync progress panel; other targets keep their independent outcomes.
A missing template vector remains missing, while SQL copy errors fail explicitly.
Exact duplicate knowledge (including archived target rows) is skipped, and duplicate
source text is copied once. Physical deletion still follows the existing policy:
an explicit later administrator publication may republish it. This is separate
from the future derived-memory deletion barriers in M02.04.

The queued target is bound to user/Agent/workspace generations. Current ownership,
status, template lineage and source text versions are checked again under SQL
locks. A stale scope cannot redirect a queued publish into a replacement workspace.
The template persona lock is shared, acquired before row locks to coordinate with
force initialization. Disabled new-user enrollment does not forbid administrator
publication to existing clones. Each source batch is capped at 1000 rows; an
oversized batch fails before launch. New copies start a new decay clock and keep
existing levels/importance. Sync orchestration still uses the existing in-process
background task and Redis progress; durable job/retry and aggregate capacity work
remain on the original roadmap. No chat frame, prompt or schema changes are made.

A failed full-sync watermark write reports an explicit finalization error while
retaining committed copy counts. It never reports the pending badge as cleared.
Retry skips previously committed knowledge, and the watermark cannot move back.


### Original knowledge documents and manual edits (M02.02 INPUT01)

Prisma migration 158 adds immutable `memory_input_origins`, identified by a
scope-bound SHA-256 receipt and linked through `source_kind=input`. Three kinds
are distinguished: `knowledge_document`, `admin_edit`, and `user_edit`. These
record submitted input; none asserts that its claims have been independently
verified. Existing `import` placeholders remain unverified and are not backfilled.
The raw payload is private and never returned by the administration evidence API.

Knowledge upload freezes the actual decoded parser input, original byte digest,
byte count, parser revision and exact parsed items. Limits remain 2 MiB per file,
200 items per upload and the existing per-memory injection budget; receipts have
an 8 MiB bound. Exact duplicates, including archived knowledge, and oversized
items retain the existing skip/report behavior. No new receipt is stored for an
all-skipped upload. Parsing and embedding happen before the transaction. Under a
dedicated append advisory lock, shared persona lock and active template scope
recheck, all new memories, vectors, changelogs and source links commit together.
The original type/category/importance/provenance and both statement/decay clocks
are preserved. An input prepared for a replaced workspace is rejected, not moved.

The public memory PATCH and repair `edit_memory` share a transactional edit
service. It rechecks the authenticated actor and the observed owner, workspace,
Agent generation, content, timestamp and metadata before writing. Stale edits
return 409; missing/archived targets and same-ID side collisions are rejected.
The new content, vector, audit, input and link commit together. Repair resolution
also commits in that transaction, with an open/unchanged queue-item check. A failed
write leaves the prior version and an open repair item intact. Metadata-only
changes do not manufacture a new text origin. Old content links remain historical;
a source is not relabeled to certify different text. Cache invalidation and the
existing optional quality/achievement hooks run after commit.

The Web template knowledge dialog now offers the existing scoped source panel.
Successful repair edits discard loaded pages and cursors, read the first page
again, and ignore late responses from the prior content version. Failed edits do
not display a successful refresh. A repair without a fixed memory target can
show the actual edited target returned by the API, without rewriting the queue
item's recorded scope. Flutter and the WeChat service entry need no request schema
change; their general non-2xx handling remains applicable. The Mini Program is
outside this batch's scope.

Verification includes synthetic real PostgreSQL/Redis and authenticated HTTP
flows, concurrent imports/edits, write-stage rollback, source/owner isolation,
current-versus-old origins, and the template clone/import/initialization lock
sequence. Separate production React tests use controlled API responses; this is
not a claim that every client or every model response passed a production E2E.
Production acceptance reads bounded existing data and checks the exact release,
migration and health. A natural post-release new-input sample must be reported
separately if none has yet occurred. Application rollback leaves the additive
schema and historical receipts intact; do not delete the evidence tables.

Original-document retention/capacity policy remains M02.16. Other memory writers,
repair merge/replacement actions, exact model/prompt receipts, multipage auditing
and derived-memory deletion barriers remain their existing roadmap tasks.


## Administrator repair receipts

Manual edits and all repair commands share scope/generation, actor, repair-item
and memory-version guards. Merge and replacement commit the memory/vector,
immutable `admin_repair` input, parent links, audit and resolved queue state in
one PostgreSQL transaction. Model reconciliation is intentionally excluded from
an explicit human replacement: it cannot silently overwrite an unrelated row.
The replaced/absorbed originals remain archived, with their historical origins.
A conflicting concurrent command returns 409 (or 404 for an archived/deleted
merge target); refresh before retrying. Repeated completed commands do not write.

The admin source panel displays the repair action and current parent archive
state. The queue retains the last result scope, so reopening a replacement shows
the new memory. Origin linkage records provenance, not independently verified
truth. Historical inputs are not guessed or backfilled. Merge/replacement of
reminder memories is rejected until the scheduled-trigger workflow can be
updated together. Cache and materialized-quality refresh remain postcommit,
best-effort effects; a receipt is not a proof of global cache convergence.

Migration `20261013010000_memory_repair_origins` only expands the input-kind
constraint. A code rollback leaves this additive migration and all snapshots
intact; do not delete receipts or rewrite migration history.

## Confirmed contradictions (CONFIRM01)

A user correction now pins a server-validated conversation/workspace/agent scope,
its execution generations, semantic old-memory version, and persisted user-message
versions before model analysis. Normal retrieval score changes do not expire the
question. Edited/deleted messages, replaced/archived memories and changed scopes
cannot authorize an overwrite. Legacy pending questions lacking these receipts
expire safely; they never write based on model-provided memory IDs alone.

After confirmation, the replacement, vector, old-memory archive (change/error),
audit and evidence links commit in one PostgreSQL transaction. The existing
importance policy remains: change inherits at least .85, error .70, addition .85.
Additions retain the old fact; duplicate/singleton/reminder conflicts fail without
archiving it. No generic model reconciliation can merge an unrelated row.

`contradiction_confirmation` input origins store private immutable command and
message-version snapshots. Their scoped unique pending-ticket index is a durable
receipt: retry never creates another fact, including after deletion of the result.
Redis cleanup compares ticket identity so it cannot delete a newer question.
Success replies require a committed write; no-change, replay and failure are
explicit outcomes. Failure details reuse the scoped administrator repair queue.

User turns reserved for the confirmation flow and its system replies carry
`memory_confirmation` metadata. Generic extraction excludes these messages,
including later watermark backlog scans; evidence binding rejects stale jobs
trying to extract them. Reserved-only bounded pages advance their watermark so
later ordinary messages still receive extraction. The full original chat remains
stored; this special flow records the selected confirmed fact rather than treating
every sentence in a clarification turn as a new independent fact.

The existing Web source panel displays the change kind, pinned user-message links
and parent memory, without returning private text or claiming factual verification.
Read permissions, default 20 / maximum 50 source-page limits and unknown/deleted/
changed states remain unchanged. Both LangGraph and the retained legacy executor
use the same confirmation service; Flutter/service-account message event contracts
are unchanged. Checkpoint replay is not enabled by this batch.

The new Prisma migration adds only an origin kind, ticket validation and a partial
unique index; it does not rewrite existing memory or prompt content. Old code can
ignore the additive origin kind after rollback. Model judgement can still be wrong;
this batch guarantees write consistency and provenance, not semantic infallibility.
No historical evidence is fabricated, no production test memory is inserted, and
all unrelated lifecycle/load, deletion-barrier and source-entry tasks remain open.
