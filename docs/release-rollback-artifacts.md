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
