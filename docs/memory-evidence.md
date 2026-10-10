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
`linked` means a recorded message/parent origin exists, including a subsequently
deleted origin. It does not mean the source is currently available or the claim
is verified. `unknown` is the complement within `checked`. The response includes
the denominator, definition and sampling time. Unknown scopes return 404 rather
than a misleading zero. The audit endpoint never backfills or writes data.

Content/source SHA-256 values identify text versions. `extractor_version` is the
server processing protocol revision; it is **not** a Web prompt publication
version or an exact model/prompt snapshot.

## Remaining original roadmap work

- M02.02.03: durable tool/business receipts and exact model/prompt snapshots.
- M02.02.04: profile, clone, knowledge/import publishing, corrections, admin and
  public edits, daily summaries, compression, location, offerings, offline/game
  and reminder adapters. Their uncollected provenance must remain unknown.
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
