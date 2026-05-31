#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${POSTGRES_BACKUP_SERVICE_NAME:-companion-postgres-backup}"
PROJECT_DIR="${POSTGRES_BACKUP_PROJECT_DIR:-/app/companion-server}"
BACKUP_SCRIPT="$PROJECT_DIR/scripts/backup_postgres.sh"
RUN_USER="${POSTGRES_BACKUP_RUN_USER:-ubuntu}"
ON_CALENDAR="${POSTGRES_BACKUP_ON_CALENDAR:-*-*-* 03:20:00}"
RANDOMIZED_DELAY="${POSTGRES_BACKUP_RANDOMIZED_DELAY:-10m}"

if [ ! -x "$BACKUP_SCRIPT" ]; then
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

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}.timer"
sudo systemctl list-timers --all "${SERVICE_NAME}.timer" --no-pager
