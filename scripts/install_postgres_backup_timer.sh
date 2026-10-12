#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${POSTGRES_BACKUP_SERVICE_NAME:-companion-postgres-backup}"
PROJECT_DIR="${POSTGRES_BACKUP_PROJECT_DIR:-/app/companion-server}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
BACKUP_SCRIPT="$PROJECT_DIR/scripts/backup_postgres.sh"
RUN_USER="root" # Same private directory owner as the server container; no world-readable dumps.
ON_CALENDAR="${POSTGRES_BACKUP_ON_CALENDAR:-*-*-* 03:20:00}"
RANDOMIZED_DELAY="${POSTGRES_BACKUP_RANDOMIZED_DELAY:-10m}"

if [ ! -x "$BACKUP_SCRIPT" ] || [ ! -f "$PROJECT_DIR/scripts/postgres_backup.py" ]; then
    echo "Backup script not found or not executable: $BACKUP_SCRIPT"
    exit 1
fi

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<EOF
[Unit]
Description=Backup Companion PostgreSQL database
Documentation=file:$BACKUP_SCRIPT
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=ENV_FILE=$PROJECT_DIR/.env
ExecStart=$BACKUP_SCRIPT
TimeoutStartSec=20min
UMask=0077
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
EOF

sudo tee "/etc/systemd/system/${SERVICE_NAME}.timer" >/dev/null <<EOF
[Unit]
Description=Run Companion PostgreSQL backup daily

[Timer]
OnCalendar=$ON_CALENDAR
Persistent=true
RandomizedDelaySec=$RANDOMIZED_DELAY
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

BACKUP_DIR="${POSTGRES_BACKUP_DIR:-${COMPANION_DATA_DIR:-/mnt/datadisk0/companion}/backups/postgres}"
sudo mkdir -p "$BACKUP_DIR"
sudo tee "/etc/systemd/system/${SERVICE_NAME}-manual.service" >/dev/null <<EOF
[Unit]
Description=Run a queued Companion manual backup with the canonical host tools
Requires=docker.service
After=docker.service
StartLimitIntervalSec=120
StartLimitBurst=20
[Service]
Type=exec
User=root
WorkingDirectory=$PROJECT_DIR
Environment=ENV_FILE=$PROJECT_DIR/.env
ExecStart=$BACKUP_SCRIPT process-request
TimeoutStartSec=20min
UMask=0077
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
RestartForceExitStatus=75
RestartSec=15s
EOF
sudo tee "/etc/systemd/system/${SERVICE_NAME}-manual.path" >/dev/null <<EOF
[Unit]
Description=Watch for admin backup requests
[Path]
PathExists=$BACKUP_DIR/.manual-request.json
Unit=${SERVICE_NAME}-manual.service
[Install]
WantedBy=multi-user.target
EOF
# Validate the rendered units before reloading or activating them. In particular,
# systemd rejects RestartForceExitStatus on oneshot services, even if a path unit
# appears enabled. Type=exec preserves the exit-75 lock-contention retry.
sudo systemd-analyze verify "/etc/systemd/system/${SERVICE_NAME}.service" \
    "/etc/systemd/system/${SERVICE_NAME}.timer" \
    "/etc/systemd/system/${SERVICE_NAME}-manual.service" \
    "/etc/systemd/system/${SERVICE_NAME}-manual.path"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}.timer" "${SERVICE_NAME}-manual.path"
sudo systemctl is-active --quiet "${SERVICE_NAME}.timer"
sudo systemctl is-active --quiet "${SERVICE_NAME}-manual.path"
sudo python3 - "$BACKUP_DIR" <<'PY_MARKER'
import json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); marker=root/".scheduler.json"
if marker.is_symlink(): raise SystemExit("unsafe scheduler marker")
import tempfile
fd,tmp=tempfile.mkstemp(prefix=".scheduler-",dir=root)
with os.fdopen(fd,"w") as out: json.dump({"schema_version":1,"executor":"host_systemd","transport":"docker_pg16"},out)
os.replace(tmp,marker)
PY_MARKER
sudo systemctl list-timers --all "${SERVICE_NAME}.timer" --no-pager

# First prove a recent exact snapshot restored in isolation. A future deploy
# must not remove the old job merely because a new timer file exists.
BACKUP_DIR="${POSTGRES_BACKUP_DIR:-${COMPANION_DATA_DIR:-/mnt/datadisk0/companion}/backups/postgres}"
if sudo test -f /etc/cron.d/companion-db-backup; then
  sudo python3 "$PROJECT_DIR/scripts/postgres_backup.py" cutover-ready --root "$BACKUP_DIR"
fi
sudo chown root:root "$BACKUP_DIR"
sudo chmod 700 "$BACKUP_DIR"
sudo find "$BACKUP_DIR" -maxdepth 1 -type f -exec chmod 600 {} +
# Deployment must never reinstall the duplicate 03:00 job. Keep its existing
# recovery points until an operator verifies the primary set and isolated restore.
sudo rm -f /etc/cron.d/companion-db-backup
sudo install -m 755 "$PROJECT_DIR/scripts/companion-db-backup.sh" /usr/local/bin/companion-db-backup.sh
