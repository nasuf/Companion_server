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
BACKUP_SCRIPT="${POSTGRES_BACKUP_SCRIPT:-$PROJECT_DIR/scripts/backup_postgres.sh}"
STOP_SERVER_DURING_RESTORE="${STOP_SERVER_DURING_RESTORE:-1}"

usage() {
    cat <<EOF
Usage:
  $0 --latest
  $0 --list <dump-file>
  $0 --verify <dump-file>
  $0 --test-restore <dump-file>
  CONFIRM_RESTORE=$POSTGRES_DB $0 --restore <dump-file>

Examples:
  $0 --latest
  $0 --test-restore /mnt/datadisk0/companion/backups/postgres/companion_20260531T031858Z.dump
  CONFIRM_RESTORE=companion $0 --restore /mnt/datadisk0/companion/backups/postgres/companion_20260531T031858Z.dump
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_container() {
    docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || die "container not found: $CONTAINER_NAME"
    docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null \
        || die "postgres is not ready: db=$POSTGRES_DB user=$POSTGRES_USER"
}

latest_dump() {
    find "$BACKUP_DIR" -maxdepth 1 -type f -name "${POSTGRES_DB}_*.dump" -print 2>/dev/null \
        | sort \
        | tail -1
}

require_dump() {
    local dump_file="$1"
    [ -n "$dump_file" ] || die "dump file is required"
    [ -f "$dump_file" ] || die "dump file not found: $dump_file"
    [ -s "$dump_file" ] || die "dump file is empty: $dump_file"
}

verify_manifest_if_present() {
    local dump_file="$1"
    local manifest_file="${dump_file%.dump}.manifest"
    if [ -f "$manifest_file" ]; then
        echo "Verifying manifest: $manifest_file"
        (cd / && head -2 "$manifest_file" | sha256sum -c -)
    else
        echo "No manifest next to dump; skipping sha256 verification"
    fi
}

verify_dump_readable() {
    local dump_file="$1"
    echo "Verifying dump can be read by pg_restore..."
    docker exec -i "$CONTAINER_NAME" pg_restore --list >/dev/null < "$dump_file"
}

list_dump() {
    local dump_file="$1"
    require_container
    require_dump "$dump_file"
    verify_manifest_if_present "$dump_file"
    docker exec -i "$CONTAINER_NAME" pg_restore --list < "$dump_file"
}

verify_dump() {
    local dump_file="$1"
    require_container
    require_dump "$dump_file"
    verify_manifest_if_present "$dump_file"
    verify_dump_readable "$dump_file"
    echo "Dump verification OK: $dump_file"
}

test_restore() {
    local dump_file="$1"
    local test_db="${RESTORE_TEST_DB:-restore_verify_$(date -u +%Y%m%d%H%M%S)}"
    require_container
    require_dump "$dump_file"
    verify_manifest_if_present "$dump_file"
    verify_dump_readable "$dump_file"

    echo "Creating temporary restore database: $test_db"
    docker exec "$CONTAINER_NAME" createdb -U "$POSTGRES_USER" "$test_db"
    cleanup_test_db() {
        docker exec "$CONTAINER_NAME" dropdb -U "$POSTGRES_USER" --if-exists "$test_db" >/dev/null 2>&1 || true
    }
    trap cleanup_test_db EXIT

    echo "Restoring dump into temporary database..."
    docker exec -i "$CONTAINER_NAME" pg_restore \
        -U "$POSTGRES_USER" \
        -d "$test_db" \
        --no-owner \
        --no-acl \
        < "$dump_file"

    docker exec "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d "$test_db" -Atc "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
    echo "Test restore OK: $test_db"
    cleanup_test_db
    trap - EXIT
}

server_was_running() {
    docker inspect -f '{{.State.Running}}' companion-server 2>/dev/null | grep -q true
}

restore_prod() {
    local dump_file="$1"
    require_container
    require_dump "$dump_file"

    if [ "${CONFIRM_RESTORE:-}" != "$POSTGRES_DB" ]; then
        die "set CONFIRM_RESTORE=$POSTGRES_DB to restore over the production database"
    fi

    verify_manifest_if_present "$dump_file"
    verify_dump_readable "$dump_file"

    if [ "${SKIP_PRE_RESTORE_BACKUP:-0}" != "1" ]; then
        [ -x "$BACKUP_SCRIPT" ] || die "backup script not executable: $BACKUP_SCRIPT"
        echo "Creating pre-restore backup..."
        "$BACKUP_SCRIPT"
    fi

    local restart_server=0
    if [ "$STOP_SERVER_DURING_RESTORE" = "1" ] && server_was_running; then
        restart_server=1
        echo "Stopping companion-server during restore..."
        docker stop companion-server >/dev/null
    fi

    restart_on_exit() {
        if [ "$restart_server" = "1" ]; then
            docker start companion-server >/dev/null || true
        fi
    }
    trap restart_on_exit EXIT

    echo "Terminating active connections to $POSTGRES_DB..."
    docker exec "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 -v db="$POSTGRES_DB" <<'SQLEOF'
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = :'db'
  AND pid <> pg_backend_pid();
SQLEOF

    echo "Dropping and recreating public schema in $POSTGRES_DB..."
    docker exec "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -v owner="$POSTGRES_USER" <<'SQLEOF'
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO :"owner";
GRANT ALL ON SCHEMA public TO public;
SQLEOF

    echo "Restoring dump into $POSTGRES_DB..."
    docker exec -i "$CONTAINER_NAME" pg_restore \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --no-owner \
        --no-acl \
        < "$dump_file"

    echo "Restore completed: $dump_file -> $POSTGRES_DB"
    restart_on_exit
    trap - EXIT
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

case "$1" in
    --latest)
        latest="$(latest_dump)"
        [ -n "$latest" ] || die "no backups found in $BACKUP_DIR"
        echo "$latest"
        ;;
    --list)
        list_dump "${2:-}"
        ;;
    --verify)
        verify_dump "${2:-}"
        ;;
    --test-restore)
        test_restore "${2:-}"
        ;;
    --restore)
        restore_prod "${2:-}"
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
