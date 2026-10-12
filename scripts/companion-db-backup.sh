#!/usr/bin/env bash
# Compatibility entry point for existing operator commands, not a second job.
set -euo pipefail
exec /app/companion-server/scripts/backup_postgres.sh "$@"
