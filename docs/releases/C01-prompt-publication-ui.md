# C01 follow-up: real Web prompt publication versions

The admin list, editor, Trace single-step editor and history now separate two
concepts: enabled state (`已生效` / `已停用`, mutually exclusive) and content
publication (`默认` / `Web Vn`). Technical revision and database source remain
API compatibility fields and concurrency guards, and are hidden from the UI.
`默认` requires complete history proving no Web content update. Missing or
inconsistent provenance is `版本待核查`, never assumed to be default or a Web
version that contains different content.

Migration `20261006010000_prompt_publication_versions` adds sidecar publication
numbers and per-key counters. Historical manual_save, reset_default and restore:*
receive contiguous numbers ordered by retained timestamp/id; bootstrap/code_sync
and enable/disable do not. Existing audit rows are untouched. Backfill and trigger
installation lock audit inserts in one transaction. The audit insert trigger
numbers subsequent publications atomically, including an old client after rollback.
Deletion/recreation and history pagination cannot reuse or renumber a Web version.
Enable/disable changes the optimistic revision but preserves content version identity.
Restoring content/defaults creates a new Web publication; state audits are not
presented as content restoration targets.

The explicit `publish_version` option uses the normal guarded Web save service to
publish existing content whose history cannot verify a current publication. It
preserves existing surrounding whitespace when republishing unchanged content;
edited drafts retain normal Web trimming. No-op normal saves do not publish.
This release changes no default or active prompt wording and preserves enabled state.

`scripts/align_prompt_publication_versions.py` defaults to read-only preflight.
Its manifest must contain reviewed key, expected_sha256 and allowed_fields.
It reads complete history, checks placeholders and renders sample values, then
--apply calls update_prompt_text with both guards and publish_version=True.
It verifies unchanged text/enabled/default, Redis, old audit rows and hot-path
reads. Reruns skip proven current Web publications. Production reconciliation
must create genuine manual_save records rather than relabel code_sync history.

Release gates: historical migration upgrade/idempotence/old-writer rehearsal,
real PG/Redis concurrency and rollback tests, API auth/guards, Web labels/history/
restore/toggle browser checks, unchanged Trace edit/replay/save, existing chat
regression checks, and two reviews per repo. Deploy server/migration first,
reconcile unverified current content, then deploy Web. Application rollback
retains additive tables/triggers; no destructive reverse migration is needed.
Existing single-container deployment interruption remains a separate workstream.

Production acceptance found the pre-existing Docker image omitted the deterministic
eval files used by prompt-version snapshots. The runtime now packages only the five
required source/data files, and CI runs the snapshot inside the actual image with
network access disabled. These checks validate the evaluation harness and reference
transcript; they do not measure the saved prompt's live model quality. Existing
failed snapshots remain unchanged. The Web history labels execution errors as
`评测不可用`, and completed validate-only results as `基础校验通过/失败`.
