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

process_cwd() {
    lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1
}

process_command() {
    ps -o command= -p "$1" 2>/dev/null || true
}

is_project_process() {
    local pid="$1"
    local cwd
    local command

    cwd="$(process_cwd "$pid")"
    command="$(process_command "$pid")"
    [[ "$cwd" == "$PWD"* || "$command" == *"$PWD"* ]]
}

kill_descendants() {
    local pid="$1"
    local signal="${2:-TERM}"
    local child

    while IFS= read -r child; do
        [ -n "$child" ] || continue
        kill_descendants "$child" "$signal"
        kill "-$signal" "$child" 2>/dev/null || true
    done < <(pgrep -P "$pid" 2>/dev/null || true)
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
        kill_descendants "$pid" TERM
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
        kill_descendants "$pid" KILL
        kill -KILL "$pid" 2>/dev/null || true
    fi
    stopped=true
}

cleanup_project_prisma_engines() {
    local pid
    local command
    local engine_pids=""

    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        [ "$pid" != "$$" ] || continue
        command="$(process_command "$pid")"
        [[ "$command" == *"prisma/query-engine"* ]] || continue
        if is_project_process "$pid"; then
            engine_pids="$engine_pids $pid"
        fi
    done < <(pgrep -f "query-engine" 2>/dev/null || true)

    if [ -n "$engine_pids" ]; then
        echo "Stopping project Prisma query-engine process(es):$engine_pids"
        for pid in $engine_pids; do
            kill_descendants "$pid" TERM
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 1
        local still_alive=""
        for pid in $engine_pids; do
            if pid_alive "$pid"; then
                still_alive="$still_alive $pid"
            fi
        done
        if [ -n "$still_alive" ]; then
            echo "Force killing project Prisma query-engine process(es):$still_alive"
            for pid in $still_alive; do
                kill_descendants "$pid" KILL
                kill -KILL "$pid" 2>/dev/null || true
            done
        fi
        stopped=true
    fi
}

cleanup_project_uvicorn_processes() {
    local pid
    local command
    local server_pids=""

    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        [ "$pid" != "$$" ] || continue
        command="$(process_command "$pid")"
        [[ "$command" == *"uvicorn"* ]] || continue
        [[ "$command" == *"app.main:app"* ]] || continue
        [[ "$command" != *"pgrep -f app.main:app"* ]] || continue
        if is_project_process "$pid"; then
            server_pids="$server_pids $pid"
        fi
    done < <(pgrep -f "app.main:app" 2>/dev/null || true)

    if [ -n "$server_pids" ]; then
        echo "Stopping stale project uvicorn process(es):$server_pids"
        for pid in $server_pids; do
            kill_descendants "$pid" TERM
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 1
        local still_alive=""
        for pid in $server_pids; do
            if pid_alive "$pid"; then
                still_alive="$still_alive $pid"
            fi
        done
        if [ -n "$still_alive" ]; then
            echo "Force killing stale project uvicorn process(es):$still_alive"
            for pid in $still_alive; do
                kill_descendants "$pid" KILL
                kill -KILL "$pid" 2>/dev/null || true
            done
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
EXISTING=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Stopping process(es) listening on port $PORT: $EXISTING"
    for PID in $EXISTING; do
        kill_pid_group "$PID" "port $PORT"
    done
fi

sleep 2

# Force kill if still alive
EXISTING_AGAIN=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$EXISTING_AGAIN" ]; then
    for PID in $EXISTING_AGAIN; do
        force_kill_pid_group "$PID" "port $PORT"
    done
fi

# Interrupted reloads can leave uvicorn/query-engine children alive even after
# the API port is free. Kill any remaining project-local processes that could
# keep database sessions open.
cleanup_project_uvicorn_processes
cleanup_project_prisma_engines

if [ "$stopped" = true ]; then
    echo "Server stopped."
else
    echo "No server running."
fi
