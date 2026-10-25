# Verified production rollback artifacts (F01)

Before Server rsync or rewriting `.env`, the incoming standalone script saves the
**healthy running container's immutable image ID** with `docker image save`.
It captures the old host `.env`/compose and actual container/image inspect in a
private archive. SHA-256 verifies every file, image identity, and every
uncompressed layer. Classic config IDs and containerd OCI index IDs are supported;
OCI descriptor hashes/sizes, referenced configs, compressed blobs and layer DiffIDs
are checked. A partial archive never becomes `latest`. Disk, Docker,
health, config-change or validation failure stops deployment before replacing
production files or stopping the service. A truly empty first deployment may
skip because no previous artifact exists; daemon failure cannot masquerade as
an absent container.

Web independently archives the served `dist` and Universal Links file before
replacing either. Deployment is serialized within each repository. Both use
`/app/companion-release-archives/{server,web}` outside the Docker prune policy.
Directories are 0700, files 0600. These artifacts **contain credentials**; never
upload them to GitHub artifacts or use their configuration in a local test.
CLI diagnostics do not print configuration or Docker error output.

After a successful deployment, retention keeps every archive from the last 14
days, at least three archives, and the latest pointer. A corrupt archive stops
pruning; it never justifies deleting other rollback artifacts. Insufficient
capacity blocks the next deployment without deleting the current rollback.

## Recovery boundary

This is image/config/static rollback, **not database or volume restoration**.
It does not reverse Prisma migrations, restore chats/media, or reproduce the
Docker container writable layer. Prisma compatibility with the previous image
must be evaluated before any schema deployment. Existing daily database backup
is separate; a verified image is not evidence that a database backup restores.
Archives on the same VPS do not protect against losing that machine/disk.

Historical `docker export`/`import` recovery creates a new image ID and can lose
image metadata. Do not label those artifacts as originals. New manifests say
`format=docker-save` and record the image that was actually running; the incoming
commit identifies the deployment requesting the archive, not the archived SHA.

## Manual recovery

Resolve the component's `latest` name to its archive directory. Verify before
use: `python3 scripts/release_archive.py verify --archive ARCHIVE`.

Server: `python3 scripts/release_archive.py load-image --archive ARCHIVE` loads
and confirms the archived image ID without retagging the running deployment.
After confirming DB compatibility and preserving the current version, create a
compose override setting `services.server.image` to that ID. Start only `server`
using the archived compose and `.env`, the original compose project name,
`--no-deps --no-build --pull never`, and the override. Verify the container image
ID, health, PostgreSQL/Redis status and client path. Never run migrations or
replace database volumes as an implicit part of code rollback.

Web: `python3 scripts/release_archive.py restore-web --archive ARCHIVE --source
/app/companion-web/dist --association
/var/www/bansheng-static/.well-known/apple-app-site-association` restores only
verified regular files, preserves the pre-restore site, and rolls both paths
back if installation fails. Check `nginx -t` and both served entrypoints.

Local recovery drills must use synthetic credentials/data, a private Docker
network and disposable volumes. Neither the archive script nor load-image
restarts production, accesses DB, or enables LangGraph/checkpoint execution.

## G03.05 delay-safe rollback floor (2026-10-25)

The routine rollback target must contain commit `410f2ba1` (lease retries and
post-dequeue-only cancellation notifications). Verify Git ancestry of the exact
source commit with `scripts/check_rollback_version.py --target-commit SHA` from a
checkout with the relevant history. Map that commit to the immutable image by
source hashes; a tag or the archive's `incoming_commit` does not prove this.

An emergency target older than `db8f60f9` has serial delayed generation, and still
older versions may dequeue messages without acquiring a conversation lock. Before
using it, turn **reply_delay_enabled OFF in the admin settings**, verify the actual
value, and drain or explicitly resolve every queued/in-flight reply. Turning off
the switch does not drain existing jobs. This also applies conservatively to
`db8f60f9` through pre-`410f2ba1` targets. Run the version gate with `--emergency
--delay-enabled false --queued 0 --inflight 0` only using freshly verified values.
Never reopen delay on these emergency targets. The gate does not certify schema,
receipts, data, or the truth of operator-supplied state; those checks remain mandatory.

For this drill, keep current Prisma migrations and business volumes. In an isolated
network, migrate a synthetic DB using Prisma, persist representative completed
memory-write and turn-followup receipts, switch between qualified images, and
verify receipt readability, non-replay of completed effects, and two-worker health.
Also verify that the version gate rejects pre-floor routine targets and an older
emergency target while delay is enabled or the queue/in-flight state is unknown.

Before production switching, preserve and verify both target and roll-forward
image archives, collect pending queue/in-flight state and an observation snapshot,
and stop only the API service gracefully. Do not stop PostgreSQL, Redis or Ollama,
run implicit migrations, reset volumes, or download private archive configurations.
After each switch verify the immutable image, two workers, health, unchanged
migration history and delay settings, errors/OOM/restarts, and queue/receipt state.
If a gate fails, retain evidence and use the preserved qualified roll-forward
image. Recovery is not a promise of zero downtime or zero lost in-flight work.

## Natural delayed traffic across deployments

Run `scripts/observe_natural_delay.py` on the operator machine, giving a private
state path, a fresh timestamped output path, approved source commit(s), and the SSH
identity. It sends only a stdlib/native read-only probe to production; no app/ORM
imports, messages, notifications, configuration updates, or DDL/DML.

The minimum eligible source commit is `db8f60f9`, checked by Git ancestry and
running-image source hashes. Each start retains its immutable image/commit/window;
repeated polls are deduplicated, actual repeated starts are failures. Previously
selected conversations remain in the query after deployment, from their earliest
selected container start. Every saved user message in that scope must have an exact
`turn_followup_events.user_message_ids` receipt and a persisted assistant message.
Three distinct real conversations are required; empty queues are not success.

State/raw metadata are atomic 0600 local files and contain UUIDs, never chat text
or tokens. Public reports hash IDs and separate due-to-generator-start seconds
from received-to-persisted-reply seconds, with counts and image windows. Logs/SQL
and accumulated messages are bounded. Truncation, retention gaps in a sampled
window, missing sources or overdue unproved replies block acceptance and remain
recorded. Between-container observation gaps remain explicitly unobserved; no
claim is made about unsampled conversations or whole-site reliability.
