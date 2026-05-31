#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-companion-postgres}"
POSTGRES_USER="${POSTGRES_USER:-companion}"
POSTGRES_DB="${POSTGRES_DB:-companion}"
DATA_DIR="${COMPANION_DATA_DIR:-/mnt/datadisk0/companion}"
BACKUP_DIR="${POSTGRES_BACKUP_DIR:-$DATA_DIR/backups/postgres}"
RETENTION_DAYS="${POSTGRES_BACKUP_RETENTION_DAYS:-14}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE_NAME="${POSTGRES_DB}_${TIMESTAMP}"
TMP_DUMP="$BACKUP_DIR/.${BASE_NAME}.dump.tmp"
DUMP_FILE="$BACKUP_DIR/${BASE_NAME}.dump"
GLOBALS_FILE="$BACKUP_DIR/${BASE_NAME}.globals.sql"
MANIFEST_FILE="$BACKUP_DIR/${BASE_NAME}.manifest"
LOG_FILE="$BACKUP_DIR/backup.log"

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

mkdir -p "$BACKUP_DIR"

if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    log "ERROR container not found: $CONTAINER_NAME"
    exit 1
fi

if ! docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null; then
    log "ERROR postgres is not ready: db=$POSTGRES_DB user=$POSTGRES_USER"
    exit 1
fi

cleanup_tmp() {
    rm -f "$TMP_DUMP"
}
trap cleanup_tmp EXIT

log "START db=$POSTGRES_DB container=$CONTAINER_NAME backup=$DUMP_FILE"

docker exec "$CONTAINER_NAME" pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --format=custom \
    --compress=6 \
    --no-owner \
    --no-acl \
    > "$TMP_DUMP"

if [ ! -s "$TMP_DUMP" ]; then
    log "ERROR empty dump produced"
    exit 1
fi

if ! docker exec -i "$CONTAINER_NAME" pg_restore --list >/dev/null < "$TMP_DUMP"; then
    log "ERROR pg_restore could not read dump"
    exit 1
fi

mv "$TMP_DUMP" "$DUMP_FILE"

docker exec "$CONTAINER_NAME" pg_dumpall \
    -U "$POSTGRES_USER" \
    --globals-only \
    > "$GLOBALS_FILE"

sha256sum "$DUMP_FILE" "$GLOBALS_FILE" > "$MANIFEST_FILE"
{
    printf 'created_at=%s\n' "$TIMESTAMP"
    printf 'container=%s\n' "$CONTAINER_NAME"
    printf 'database=%s\n' "$POSTGRES_DB"
    printf 'user=%s\n' "$POSTGRES_USER"
    printf 'retention_days=%s\n' "$RETENTION_DAYS"
    printf 'dump_bytes=%s\n' "$(wc -c < "$DUMP_FILE" | tr -d ' ')"
    printf 'globals_bytes=%s\n' "$(wc -c < "$GLOBALS_FILE" | tr -d ' ')"
} >> "$MANIFEST_FILE"

find "$BACKUP_DIR" -type f \( \
    -name '*.dump' -o \
    -name '*.globals.sql' -o \
    -name '*.manifest' \
\) -mtime +"$RETENTION_DAYS" -delete

log "DONE dump=$DUMP_FILE manifest=$MANIFEST_FILE"
