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

echo "Applying pending Prisma migrations..."
"$PRISMA_BIN" migrate deploy --schema "$SCHEMA_PATH"

echo ""
echo "Generating Prisma client..."
"$PRISMA_BIN" generate --schema "$SCHEMA_PATH"

echo ""
echo "Migration complete."
echo "Prisma applied any pending migration SQL tracked by _prisma_migrations."
