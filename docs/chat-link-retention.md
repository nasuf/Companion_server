# Link resource lifecycle

`link_retention_v1` completes the storage side of link preparation. It does not
activate SQL chat admission, consumers or checkpoint replay.

The resource-monitoring page exposes a manual dry-run and a bounded execution.
Each execution scans at most 100 rows per resource kind in primary-key order,
advancing persisted cursors even when rows remain protected. An empty tail wraps
the next scan to the start. Counts describe that scan segment, not whole-table
cardinality. `pg_total_relation_size` includes indexes and may not shrink after
row deletion. Dry-run writes its administrative result only.

A resource must be older than seven days and unreferenced before retirement.
SQL message bindings, legacy message JSON projections and the legacy cache's
message pointer are protected. Cache deletion is allowed only after all its
snapshots have gone; it never cascades an unbounded version history. Referenced
messages and their contents are never modified by maintenance. There is no
historical backfill or claim that an old unversioned projection had a snapshot.

Prisma migration `20261015150000_chat_link_retention` installs narrow equality
indexes and a write guard. Link-related message/cache/snapshot writes acquire the
same transaction advisory lock as maintenance. New references must resolve;
references to retiring covers are rejected. Short SQL maintenance uses a 3s
statement timeout and skips cache/snapshot rows held by independent writers.
There is no provider or file IO under that lock. Regular non-link messages do
not acquire the link lock. SQL ingress maps resource disappearance to its existing
resource-invalid contract; transaction errors still roll back every effect.

New downloaded covers reserve their bytes in SQL before exclusive file creation.
Registered covers have a 1 GiB budget, and caching also leaves 1 GiB disk free.
If either guard prevents caching, the original remote image remains available.
Reservations include incomplete/ambiguous writes, so repeated failures cannot
silently exceed the budget. Existing unregistered covers and user-uploaded media
are outside this budget and are **never** discovered for deletion by a directory
scan. This limitation is displayed in the admin panel.

Cover retirement commits `deleting` before filesystem unlink. A crash at any
later point is safe to retry: missing files count as already unlinked, quota is
released once, and symlinks/non-files are failures. A separate retirement cursor
prevents permanently failing files from starving later rows. One maintenance
run owns a 10-minute lease; expired ownership is recoverable. Reports distinguish
never-run, running/interrupted, failed, committed SQL and unconfirmed SQL commits.
Only aggregate counters and controlled error classes leave the service.

Operations: open Statistics → Resource monitoring → Link storage governance;
refresh, preview, then confirm a bounded execution. Repeat to advance the scan.
This release does not add an automatic destructive cron. A successful API call
with zero candidates proves only the sampled scope, not a full storage audit.

Agent recommendation reshares keep the original source role/version while the
new message is a user message. Ownership and conversation still must match.
Unversioned unusable Weibo previews may refresh from server-owned source URLs;
client titles/bodies cannot replace the trusted extraction. Explicit versions
never silently refresh. Redirects can create a new cache only with a trusted,
same-conversation refresh lineage validated again at admission.

Validation covers real PostgreSQL migration/rollback, legacy and SQL references,
concurrent writes/cleanup, bounded cursors, managed file recovery/capacity,
permissions and error contracts, plus desktop browser state and confirmation.
Production acceptance uses read-only queries/UI; synthetic deletes run only on
isolated fixtures. Actual consumer/delivery/takeover qualification remains in
R01.03.8.3.2.4 and must pass before enabling SQL ingress.
