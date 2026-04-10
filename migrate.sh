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

# 诊断：检查 Supabase 是否可达
DB_HOST=$(grep -E '^DATABASE_URL=' .env 2>/dev/null | sed -E 's/.*@([^:\/]+).*/\1/' || true)
if [ -n "$DB_HOST" ]; then
    if ! nc -z -w 5 "$DB_HOST" 5432 2>/dev/null; then
        echo ""
        echo "⚠️  无法连接 $DB_HOST:5432"
        echo "   如果你开启了系统级代理 (Clash TUN/Surge 增强模式等)，"
        echo "   请将 $DB_HOST 加入代理的「直连规则」或临时切换为「仅代理模式」。"
        echo ""
        echo "   快速验证: nc -z -w 5 $DB_HOST 5432"
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
