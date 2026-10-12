#!/usr/bin/env bash
# Install an owner-scoped, standalone archive rotation job. No app/DB imports.
set -euo pipefail
ARCHIVE_OWNER="${1:-$(id -un)}"
if [[ ! "$ARCHIVE_OWNER" =~ ^[a-z_][a-z0-9_-]*$ ]] || [[ "$ARCHIVE_OWNER" == root ]]; then
  echo 'Archive maintenance requires an unprivileged owner' >&2
  exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sudo install -d -m 755 /usr/local/lib/companion
sudo install -m 755 "$SCRIPT_DIR/release_archive.py" /usr/local/lib/companion/release_archive.py.new
sudo mv /usr/local/lib/companion/release_archive.py.new /usr/local/lib/companion/release_archive.py
sudo tee /usr/local/bin/companion-release-archive-maintenance >/dev/null <<'ROTATE'
#!/bin/sh
set -u
result=0
for kind in server web; do
    case "$kind" in
        server) count=5; bytes=4294967296 ;;
        web) count=10; bytes=268435456 ;;
    esac
    root="/app/companion-release-archives/$kind"
    [ -d "$root" ] || continue
    if nice -n 19 ionice -c 3 python3 /usr/local/lib/companion/release_archive.py prune \
        --root "$root" --max-archives "$count" --max-bytes "$bytes"; then
        logger -t companion-archive "rotation ok: $kind count<=$count bytes<=$bytes"
    else
        logger -p user.err -t companion-archive "rotation FAILED: $kind; protected rollback copies preserved; inspect .retention-status.json"
        result=1
    fi
done
exit "$result"
ROTATE
sudo chmod 755 /usr/local/bin/companion-release-archive-maintenance
printf '17 * * * * %s /usr/local/bin/companion-release-archive-maintenance >/dev/null 2>&1\n' "$ARCHIVE_OWNER" \
    | sudo tee /etc/cron.d/companion-release-archives >/dev/null
sudo chmod 644 /etc/cron.d/companion-release-archives
