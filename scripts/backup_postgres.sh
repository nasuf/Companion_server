#!/usr/bin/env bash
# One canonical backup engine; no application imports or scheduler dependency.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
BACKUP_DIR="${POSTGRES_BACKUP_DIR:-${COMPANION_DATA_DIR:-/mnt/datadisk0/companion}/backups/postgres}"
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
ACTION="${1:-create}"
set +e
python3 "$SCRIPT_DIR/postgres_backup.py" "$ACTION" --root "$BACKUP_DIR"
RESULT=$?
set -e
if [ "$RESULT" -ne 0 ]; then
  logger -p user.err -t companion-backup 'Database backup failed; inspect the private backup status and systemd journal'
  exit "$RESULT"
fi
