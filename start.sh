#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PID_FILE=".server.pid"

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
EXISTING=$(lsof -i :"$PORT" -t 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Killing existing process on port $PORT (PID $EXISTING)..."
    kill $EXISTING 2>/dev/null || true
    sleep 2
    # Force kill if still alive
    EXISTING_AGAIN=$(lsof -i :"$PORT" -t 2>/dev/null || true)
    if [ -n "$EXISTING_AGAIN" ]; then
        kill -9 $EXISTING_AGAIN 2>/dev/null || true
    fi
    sleep 1
fi

# ── Check Docker ──
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Please install Docker."
    exit 1
fi

# ── Check & start Redis ──
echo "Checking Redis..."
REDIS_RUNNING=$(docker ps --filter "publish=6379" --format "{{.Names}}" 2>/dev/null || true)
if [ -z "$REDIS_RUNNING" ]; then
    # Try starting existing stopped container
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
# Verify Redis is reachable
if redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "  Redis: OK"
else
    echo "  Redis: started (waiting for ready...)"
    sleep 2
fi

# ── Ensure Prisma client is generated ──
echo "Generating Prisma client..."
export PATH="$(pwd)/.venv/bin:$PATH"
.venv/bin/prisma generate 2>/dev/null || true

# ── 绕过代理 ──
# 本地开发时如果开启了代理 (HTTP_PROXY/HTTPS_PROXY)，
# Prisma engine / Ollama / Supabase 连接会被代理拦截。
# 把 localhost + Supabase host 加到 NO_PROXY 白名单绕过代理直连。
# 仅影响通过 start.sh 启动的本地开发环境，生产部署不走此脚本，不受影响。
LOCAL_NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
if [ -f ".env" ]; then
    DB_HOST=$(grep -E '^DATABASE_URL=' .env | sed -E 's/.*@([^:\/]+).*/\1/' || true)
    [ -n "$DB_HOST" ] && LOCAL_NO_PROXY="$LOCAL_NO_PROXY,$DB_HOST"
fi
if [ -n "$NO_PROXY" ]; then
    export NO_PROXY="$LOCAL_NO_PROXY,$NO_PROXY"
else
    export NO_PROXY="$LOCAL_NO_PROXY"
fi
export no_proxy="$NO_PROXY"

# ── Start server ──
echo ""
echo "Starting server on port $PORT..."
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Server started (PID $SERVER_PID)"
echo "  API:  http://localhost:$PORT"
echo "  Docs: http://localhost:$PORT/docs"
echo ""
echo "Run ./stop.sh to stop."
