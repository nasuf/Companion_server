#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PID_FILE=".server.pid"

pid_alive() {
    kill -0 "$1" 2>/dev/null
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

kill_process_tree() {
    local pid="$1"
    local signal="${2:-TERM}"
    local child

    while IFS= read -r child; do
        [ -n "$child" ] || continue
        kill_process_tree "$child" "$signal"
    done < <(pgrep -P "$pid" 2>/dev/null || true)

    kill "-$signal" "$pid" 2>/dev/null || true
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
            kill_process_tree "$pid" TERM
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
                kill_process_tree "$pid" KILL
            done
        fi
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
            kill_process_tree "$pid" TERM
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
                kill_process_tree "$pid" KILL
            done
        fi
    fi
}

# ── Check if already running ──
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server already running (PID $OLD_PID). Run ./stop.sh first."
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# ── Kill anything on the port ──
EXISTING=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Killing existing process on port $PORT (PID $EXISTING)..."
    for pid in $EXISTING; do
        kill_process_tree "$pid" TERM
    done
    sleep 2
    # Force kill if still alive
    EXISTING_AGAIN=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$EXISTING_AGAIN" ]; then
        for pid in $EXISTING_AGAIN; do
            kill_process_tree "$pid" KILL
        done
    fi
    sleep 1
fi

# Uvicorn reloads and interrupted shell sessions can leave Python/Prisma child
# processes alive without listening on the API port. Those children keep DB
# sessions open, so clean project-local leftovers before starting a fresh server.
cleanup_project_uvicorn_processes
cleanup_project_prisma_engines

# ── Check Docker ──
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Please install Docker."
    exit 1
fi

# ── Check & start Redis ──
echo "Checking Redis..."
REDIS_URL_RAW="redis://localhost:6379/0"
if [ -f ".env" ]; then
    env_redis_url=$(grep -E "^REDIS_URL=" .env | head -1 | cut -d= -f2- | tr -d "'\"")
    if [ -n "$env_redis_url" ]; then
        REDIS_URL_RAW="$env_redis_url"
    fi
fi
REDIS_HOST=$(.venv/bin/python - <<PYEOF
from urllib.parse import urlparse
url = urlparse("$REDIS_URL_RAW")
print(url.hostname or "localhost")
PYEOF
)
REDIS_PORT=$(.venv/bin/python - <<PYEOF
from urllib.parse import urlparse
url = urlparse("$REDIS_URL_RAW")
print(url.port or 6379)
PYEOF
)

if [ "$REDIS_HOST" = "localhost" ] || [ "$REDIS_HOST" = "127.0.0.1" ]; then
    REDIS_RUNNING=$(docker ps --filter "publish=${REDIS_PORT}" --format "{{.Names}}" 2>/dev/null || true)
    if [ -z "$REDIS_RUNNING" ] && [ "$REDIS_PORT" = "6379" ]; then
        # Try starting existing stopped container only for the default local Redis.
        REDIS_STOPPED=$(docker ps -a --filter "publish=6379" --format "{{.Names}}" 2>/dev/null | head -1)
        if [ -n "$REDIS_STOPPED" ]; then
            echo "Starting stopped Redis container ($REDIS_STOPPED)..."
            docker start "$REDIS_STOPPED"
        else
            echo "Starting new Redis container..."
            docker run -d --name companion-redis -p 6379:6379 redis:7-alpine
        fi
        sleep 1
    fi
fi

if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
    echo "  Redis: OK ($REDIS_HOST:$REDIS_PORT)"
else
    echo "  Redis: $REDIS_HOST:$REDIS_PORT unreachable"
    if [ "$REDIS_PORT" != "6379" ]; then
        echo "  If this is the Tencent tunnel, run:"
        echo "    ssh -i ~/.ssh/companion_tencent -fN -o ExitOnForwardFailure=yes -L 127.0.0.1:${REDIS_PORT}:172.18.0.2:6379 ubuntu@106.52.115.80"
    fi
    exit 1
fi

# ── Ensure Prisma client is generated ──
echo "Generating Prisma client..."
export PATH="$(pwd)/.venv/bin:$PATH"
.venv/bin/prisma generate 2>/dev/null || true

# ── 绕过代理 ──
# 本地开发时如果开启了代理 (HTTP_PROXY/HTTPS_PROXY)，
# Prisma engine / Ollama / Postgres 连接会被代理拦截。
# 把 localhost + .env 里所有 database host 加到 NO_PROXY 白名单绕过代理直连。
# 仅影响通过 start.sh 启动的本地开发环境，生产部署不走此脚本。
# httpx 在看到 socks:// 代理变量时会在 Client 初始化阶段就要求 socksio，
# 即使请求目标命中 NO_PROXY 也可能先失败。因此本地脚本直接清理代理变量，
# 避免 Ollama/LangChain import 阶段被宿主 shell 的代理设置绊倒。
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
NO_PROXY_ENTRIES="localhost 127.0.0.1 0.0.0.0 ::1"
if [ -f ".env" ]; then
    # 同时抓 DATABASE_URL 和 DIRECT_DATABASE_URL 两个 host,
    # 运行时和迁移 URL 可能不同, 漏掉一个会让某一路径被代理拦截.
    for var in DATABASE_URL DIRECT_DATABASE_URL; do
        url=$(grep -E "^${var}=" .env | head -1 | cut -d= -f2- | tr -d "'\"")
        host=$(echo "$url" | sed -nE 's|^.*@([^:/]+).*$|\1|p')
        if [ -n "$host" ]; then
            NO_PROXY_ENTRIES="$NO_PROXY_ENTRIES $host"
        fi
    done
fi
# 去重 + 逗号拼接
LOCAL_NO_PROXY=$(echo "$NO_PROXY_ENTRIES" | tr ' ' '\n' | awk '!seen[$0]++' | paste -sd, -)
if [ -n "$NO_PROXY" ]; then
    export NO_PROXY="$LOCAL_NO_PROXY,$NO_PROXY"
else
    export NO_PROXY="$LOCAL_NO_PROXY"
fi
export no_proxy="$NO_PROXY"

# ── 打印代理状态, 让用户一眼看到 ──
echo "Proxy env:"
echo "  HTTP_PROXY=${HTTP_PROXY:-<unset>}"
echo "  HTTPS_PROXY=${HTTPS_PROXY:-<unset>}"
echo "  ALL_PROXY=${ALL_PROXY:-<unset>}"
echo "  NO_PROXY=$NO_PROXY"
echo "  http_proxy=${http_proxy:-<unset>}"
echo "  https_proxy=${https_proxy:-<unset>}"
echo "  all_proxy=${all_proxy:-<unset>}"
echo "  no_proxy=$no_proxy"

# ── TCP preflight: 在 uvicorn 启动前直接用 python socket 握手 DB 端口 ──
# socket 不受 HTTP_PROXY 影响, 所以这个预检能精确告诉用户: 是 TCP 不通 (VPN/
# 防火墙/数据库不可达), 还是 Prisma 特有的代理问题.
if [ -f ".env" ]; then
    echo "Testing DB TCP connectivity..."
    for var in DATABASE_URL DIRECT_DATABASE_URL; do
        url=$(grep -E "^${var}=" .env | head -1 | cut -d= -f2- | tr -d "'\"")
        host=$(echo "$url" | sed -nE 's|^.*@([^:/]+).*$|\1|p')
        port=$(echo "$url" | sed -nE 's|^.*@[^:]+:([0-9]+).*$|\1|p')
        if [ -n "$host" ] && [ -n "$port" ]; then
            .venv/bin/python - <<PYEOF || echo "  (TCP 预检失败, 继续启动让 Prisma 自己重试)"
import socket, sys
s = socket.socket()
s.settimeout(5)
try:
    s.connect(("$host", int("$port")))
    print(f"  ✓ {'$var':22s} $host:$port reachable")
except Exception as e:
    print(f"  ✗ {'$var':22s} $host:$port unreachable: {e}")
    sys.exit(1)
finally:
    s.close()
PYEOF
        fi
    done
fi

# ── Start server ──
echo ""
echo "Starting server on port $PORT..."
export DB_CONNECTION_LIMIT="${DB_CONNECTION_LIMIT:-3}"
export DB_CONNECTION_LIMIT_MAX="${DB_CONNECTION_LIMIT_MAX:-5}"
export DB_MAX_CONCURRENT_QUERIES="${DB_MAX_CONCURRENT_QUERIES:-$DB_CONNECTION_LIMIT}"
export DB_QUERY_MAX_RETRIES="${DB_QUERY_MAX_RETRIES:-4}"

UVICORN_ARGS=(app.main:app --host 0.0.0.0 --port "$PORT")
if [ "${RELOAD:-0}" = "1" ] || [ "${RELOAD:-0}" = "true" ]; then
    UVICORN_ARGS=(app.main:app --reload --host 0.0.0.0 --port "$PORT")
    echo "  Reload: enabled"
else
    echo "  Reload: disabled (use RELOAD=1 ./start.sh to enable)"
fi
echo "  DB connection_limit=$DB_CONNECTION_LIMIT max_concurrent_queries=$DB_MAX_CONCURRENT_QUERIES"

.venv/bin/uvicorn "${UVICORN_ARGS[@]}" &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Server started (PID $SERVER_PID)"
echo "  API:  http://localhost:$PORT"
echo "  Docs: http://localhost:$PORT/docs"
echo ""
echo "Run ./stop.sh to stop."
