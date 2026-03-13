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
    kill "$EXISTING" 2>/dev/null || true
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

# ── Check & start Neo4j ──
echo "Checking Neo4j..."
NEO4J_RUNNING=$(docker ps --filter "publish=7687" --format "{{.Names}}" 2>/dev/null || true)
if [ -z "$NEO4J_RUNNING" ]; then
    # Try docker-compose
    COMPOSE_FILE=""
    [ -f "docker-compose.yml" ] && COMPOSE_FILE="docker-compose.yml"
    [ -f "../docker-compose.yml" ] && COMPOSE_FILE="../docker-compose.yml"

    if [ -n "$COMPOSE_FILE" ]; then
        echo "Starting Neo4j via docker compose..."
        docker compose -f "$COMPOSE_FILE" up -d neo4j 2>/dev/null || \
        docker-compose -f "$COMPOSE_FILE" up -d neo4j 2>/dev/null || true
    else
        NEO4J_STOPPED=$(docker ps -a --filter "publish=7687" --format "{{.Names}}" 2>/dev/null | head -1)
        if [ -n "$NEO4J_STOPPED" ]; then
            echo "Starting stopped Neo4j container ($NEO4J_STOPPED)..."
            docker start "$NEO4J_STOPPED"
        else
            echo "WARNING: No Neo4j container found. Start it manually."
        fi
    fi
    sleep 2
fi
# Verify Neo4j is reachable
if curl -s http://localhost:7474 >/dev/null 2>&1; then
    echo "  Neo4j: OK"
else
    echo "  Neo4j: started (may take a few seconds to be ready)"
fi

# ── Ensure Prisma client is generated ──
echo "Generating Prisma client..."
export PATH="$(pwd)/.venv/bin:$PATH"
.venv/bin/prisma generate 2>/dev/null || true

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
