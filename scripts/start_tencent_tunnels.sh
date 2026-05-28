#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SSH_HOST="${TENCENT_SSH_HOST:-ubuntu@106.52.115.80}"
SSH_KEY="${TENCENT_SSH_KEY:-$HOME/.ssh/companion_tencent}"
ENV_FILE="${ENV_FILE:-.env}"

ssh_base=(ssh -i "$SSH_KEY" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)

env_value() {
    local key="$1"
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi
    grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "'\""
}

url_part() {
    local raw_url="$1"
    local part="$2"
    python3 - "$raw_url" "$part" <<'PYEOF'
from urllib.parse import urlparse
import sys

url = urlparse(sys.argv[1])
part = sys.argv[2]
if part == "host":
    print(url.hostname or "")
elif part == "port":
    print(url.port or "")
PYEOF
}

is_local_host() {
    [ "$1" = "localhost" ] || [ "$1" = "127.0.0.1" ]
}

tcp_ok() {
    local host="$1"
    local port="$2"
    nc -z "$host" "$port" >/dev/null 2>&1
}

redis_ok() {
    local host="$1"
    local port="$2"
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "$host" -p "$port" ping 2>/dev/null | grep -q PONG
    else
        tcp_ok "$host" "$port"
    fi
}

start_tunnel() {
    local name="$1"
    local local_port="$2"
    local remote_host="$3"
    local remote_port="$4"

    echo "  Starting $name tunnel: 127.0.0.1:${local_port} -> ${remote_host}:${remote_port}"
    "${ssh_base[@]}" -fN -L "127.0.0.1:${local_port}:${remote_host}:${remote_port}" "$SSH_HOST"
    sleep 1
}

echo "Checking Tencent SSH tunnels..."

if [ ! -f "$SSH_KEY" ]; then
    echo "  SSH key not found: $SSH_KEY"
    echo "  Set TENCENT_SSH_KEY or run SKIP_TENCENT_TUNNELS=1 ./start.sh to skip."
    exit 1
fi

DATABASE_URL_RAW="$(env_value DATABASE_URL)"
if [ -z "$DATABASE_URL_RAW" ]; then
    DATABASE_URL_RAW="$(env_value DIRECT_DATABASE_URL)"
fi
if [ -n "$DATABASE_URL_RAW" ]; then
    db_host="$(url_part "$DATABASE_URL_RAW" host)"
    db_port="$(url_part "$DATABASE_URL_RAW" port)"
    db_port="${db_port:-5432}"
    if is_local_host "$db_host"; then
        if tcp_ok "$db_host" "$db_port"; then
            echo "  Postgres: OK (${db_host}:${db_port})"
        else
            start_tunnel "Postgres" "$db_port" "127.0.0.1" "5432"
            tcp_ok "$db_host" "$db_port" || {
                echo "  Postgres tunnel failed (${db_host}:${db_port})"
                exit 1
            }
            echo "  Postgres: OK (${db_host}:${db_port})"
        fi
    fi
fi

REDIS_URL_RAW="$(env_value REDIS_URL)"
if [ -z "$REDIS_URL_RAW" ]; then
    REDIS_URL_RAW="redis://localhost:6379/0"
fi
redis_host="$(url_part "$REDIS_URL_RAW" host)"
redis_port="$(url_part "$REDIS_URL_RAW" port)"
redis_host="${redis_host:-localhost}"
redis_port="${redis_port:-6379}"
if is_local_host "$redis_host"; then
    if redis_ok "$redis_host" "$redis_port"; then
        echo "  Redis: OK (${redis_host}:${redis_port})"
    elif [ "$redis_port" = "6381" ]; then
        redis_container_ip="$("${ssh_base[@]}" "$SSH_HOST" "sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' companion-redis" | tr -d '[:space:]')"
        if [ -z "$redis_container_ip" ]; then
            echo "  Could not resolve remote companion-redis container IP"
            exit 1
        fi
        start_tunnel "Redis" "$redis_port" "$redis_container_ip" "6379"
        redis_ok "$redis_host" "$redis_port" || {
            echo "  Redis tunnel failed (${redis_host}:${redis_port})"
            exit 1
        }
        echo "  Redis: OK (${redis_host}:${redis_port})"
    else
        echo "  Redis: ${redis_host}:${redis_port} is not reachable; not a Tencent tunnel port, leaving local Redis startup to start.sh"
    fi
fi
