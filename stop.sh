#!/usr/bin/env bash

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PID_FILE=".server.pid"

stopped=false

pid_alive() {
    kill -0 "$1" 2>/dev/null
}

pid_pgid() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}

kill_pid_group() {
    local pid="$1"
    local label="$2"
    local pgid
    local self_pgid

    if ! pid_alive "$pid"; then
        return
    fi

    pgid="$(pid_pgid "$pid")"
    self_pgid="$(pid_pgid "$$")"
    if [ -n "$pgid" ] && [ "$pgid" != "$self_pgid" ]; then
        echo "Stopping $label process group (PGID $pgid, from PID $pid)..."
        kill -TERM "-$pgid" 2>/dev/null || true
    else
        echo "Stopping $label process (PID $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
    fi
    stopped=true
}

force_kill_pid_group() {
    local pid="$1"
    local label="$2"
    local pgid
    local self_pgid

    if ! pid_alive "$pid"; then
        return
    fi

    pgid="$(pid_pgid "$pid")"
    self_pgid="$(pid_pgid "$$")"
    if [ -n "$pgid" ] && [ "$pgid" != "$self_pgid" ]; then
        echo "Force killing $label process group (PGID $pgid)..."
        kill -KILL "-$pgid" 2>/dev/null || true
    else
        echo "Force killing $label process (PID $pid)..."
        kill -KILL "$pid" 2>/dev/null || true
    fi
    stopped=true
}

cleanup_orphan_prisma_engines() {
    local pid
    local ppid
    local cwd
    local orphan_pids=""

    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
        [ "$ppid" = "1" ] || continue
        cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)"
        [ "$cwd" = "$PWD" ] || continue
        orphan_pids="$orphan_pids $pid"
    done < <(pgrep -f "prisma/query-engine" 2>/dev/null || true)

    if [ -n "$orphan_pids" ]; then
        echo "Stopping orphan Prisma query-engine process(es):$orphan_pids"
        kill $orphan_pids 2>/dev/null || true
        sleep 1
        local still_alive=""
        for pid in $orphan_pids; do
            if pid_alive "$pid"; then
                still_alive="$still_alive $pid"
            fi
        done
        if [ -n "$still_alive" ]; then
            echo "Force killing orphan Prisma query-engine process(es):$still_alive"
            kill -9 $still_alive 2>/dev/null || true
        fi
        stopped=true
    fi
}

# Stop by PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill_pid_group "$PID" "server"
    rm -f "$PID_FILE"
fi

# Also kill anything on the port
EXISTING=$(lsof -i :"$PORT" -t 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Stopping process(es) listening on port $PORT: $EXISTING"
    for PID in $EXISTING; do
        kill_pid_group "$PID" "port $PORT"
    done
fi

sleep 2

# Force kill if still alive
EXISTING_AGAIN=$(lsof -i :"$PORT" -t 2>/dev/null || true)
if [ -n "$EXISTING_AGAIN" ]; then
    for PID in $EXISTING_AGAIN; do
        force_kill_pid_group "$PID" "port $PORT"
    done
fi

# Uvicorn --reload can leave Prisma query-engine children orphaned. Those
# engines keep DB sessions open even after the API process stops.
cleanup_orphan_prisma_engines

if [ "$stopped" = true ]; then
    echo "Server stopped."
else
    echo "No server running."
fi
