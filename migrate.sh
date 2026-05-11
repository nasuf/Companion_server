#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SCHEMA_PATH="prisma/schema.prisma"
PRISMA_BIN=".venv/bin/prisma"

if [ ! -f "$SCHEMA_PATH" ]; then
    echo "ERROR: Prisma schema not found at $SCHEMA_PATH"
    exit 1
fi

if [ ! -x "$PRISMA_BIN" ]; then
    echo "ERROR: Prisma CLI not found at $PRISMA_BIN"
    echo "Run your environment setup first so .venv is available."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "WARNING: .env not found in $(pwd)"
    echo "Prisma will rely on the current shell environment."
fi

export PATH="$(pwd)/.venv/bin:$PATH"

read_env_value() {
    local key="$1"
    if [ ! -f ".env" ]; then
        return 0
    fi
    grep -E "^${key}=" .env 2>/dev/null \
        | head -1 \
        | cut -d= -f2- \
        | sed -E 's/^["'\'']?//; s/["'\'']?$//'
}

# ── 临时卸载代理 ──
# Prisma CLI 的 Rust 引擎不尊重 NO_PROXY，代理会拦截 TCP 连接导致连不上 Supabase。
# 在运行 prisma 命令期间完全卸载代理环境变量，跑完后恢复。
_saved_http="${HTTP_PROXY:-}"
_saved_https="${HTTPS_PROXY:-}"
_saved_all="${ALL_PROXY:-}"
_saved_http_lc="${http_proxy:-}"
_saved_https_lc="${https_proxy:-}"
_saved_all_lc="${all_proxy:-}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true

# Migration 必须使用专用 URL。运行时 DATABASE_URL 通常走 Supabase session
# pooler(5432), 长跑服务占用连接后容易让 prisma migrate deploy 撞
# EMAXCONNSESSION。MIGRATION_DATABASE_URL 使用 transaction pooler(6543) +
# connection_limit=1, 并显式覆盖 DIRECT_DATABASE_URL, 避免 Prisma migrate
# 因 schema.prisma 的 directUrl 回退到运行时连接。
MIGRATION_URL="${MIGRATION_DATABASE_URL:-$(read_env_value MIGRATION_DATABASE_URL)}"
if [ -n "$MIGRATION_URL" ]; then
    export DATABASE_URL="$MIGRATION_URL"
    export DIRECT_DATABASE_URL="$MIGRATION_URL"
    echo "Using MIGRATION_DATABASE_URL for Prisma migrations."
else
    echo "WARNING: MIGRATION_DATABASE_URL is not set; falling back to DATABASE_URL."
    echo "         Supabase deployments should set MIGRATION_DATABASE_URL to the"
    echo "         transaction pooler URL (:6543?pgbouncer=true&connection_limit=1)."
fi

# 诊断：检查 migration 数据库端口是否可达
ACTIVE_DATABASE_URL="${DATABASE_URL:-$(read_env_value DATABASE_URL)}"
DB_HOST=$(echo "$ACTIVE_DATABASE_URL" | sed -nE 's|^.*@([^:/]+).*$|\1|p')
DB_PORT=$(echo "$ACTIVE_DATABASE_URL" | sed -nE 's|^.*@[^:]+:([0-9]+).*$|\1|p')
DB_PORT="${DB_PORT:-5432}"
if [ -n "$DB_HOST" ]; then
    if ! nc -z -w 5 "$DB_HOST" "$DB_PORT" 2>/dev/null; then
        echo ""
        echo "⚠️  无法连接 $DB_HOST:$DB_PORT"
        echo "   如果你开启了系统级代理 (Clash TUN/Surge 增强模式等)，"
        echo "   请将 $DB_HOST 加入代理的「直连规则」或临时切换为「仅代理模式」。"
        echo ""
        echo "   快速验证: nc -z -w 5 $DB_HOST $DB_PORT"
        echo ""
        exit 1
    fi
fi

echo "Applying pending Prisma migrations..."
"$PRISMA_BIN" migrate deploy --schema "$SCHEMA_PATH"

echo ""
echo "Generating Prisma client..."
"$PRISMA_BIN" generate --schema "$SCHEMA_PATH"

# 恢复代理
[ -n "$_saved_http" ] && export HTTP_PROXY="$_saved_http"
[ -n "$_saved_https" ] && export HTTPS_PROXY="$_saved_https"
[ -n "$_saved_all" ] && export ALL_PROXY="$_saved_all"
[ -n "$_saved_http_lc" ] && export http_proxy="$_saved_http_lc"
[ -n "$_saved_https_lc" ] && export https_proxy="$_saved_https_lc"
[ -n "$_saved_all_lc" ] && export all_proxy="$_saved_all_lc"

echo ""
echo "Migration complete."
echo "Prisma applied any pending migration SQL tracked by _prisma_migrations."
