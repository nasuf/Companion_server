"""Upgrade the populated pre-validity baseline without inventing historical bounds."""

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4

from prisma import Prisma
import pytest


@pytest.mark.asyncio
async def test_validity_upgrade_preserves_rows_and_does_not_backfill(tmp_path):
    # The proactive test service has pgvector, as required by the historical
    # migration chain. G03 deliberately uses plain PostgreSQL in Linux CI.
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Memory evidence migration requires disposable PostgreSQL")
    parsed = urlsplit(url)
    assert (
        parsed.hostname in {"localhost", "127.0.0.1"}
        and parsed.path == "/companion_proactive_e2e"
    )
    url = urlunsplit(parsed._replace(path="/postgres"))
    database_name = "validity_migration_" + uuid4().hex
    restore_name = "validity_restore_" + uuid4().hex
    target_url = urlunsplit(parsed._replace(path="/" + database_name))
    admin = Prisma(datasource={"url": url}, http={"trust_env": False})
    await admin.connect()
    client = None
    try:
        await admin.execute_raw(f'CREATE DATABASE "{database_name}"')
        client = Prisma(datasource={"url": target_url}, http={"trust_env": False})
        await client.connect()
        await client.execute_raw("CREATE SCHEMA extensions")
        await client.execute_raw("CREATE EXTENSION vector WITH SCHEMA extensions")
        schema = tmp_path / "prisma"
        schema.mkdir()
        shutil.copy2("prisma/schema.prisma", schema / "schema.prisma")
        migrations = schema / "migrations"
        migrations.mkdir()
        shutil.copy2(
            "prisma/migrations/migration_lock.toml", migrations / "migration_lock.toml"
        )
        all_migrations = sorted(Path("prisma/migrations").glob("*/migration.sql"))
        # The parent schema is the exact baseline before this additive migration.
        candidate = next(
            p
            for p in all_migrations
            if p.parent.name == "20261101010000_memory_temporal_assertions"
        )
        release_chain = [
            p for p in all_migrations if p.parent.name <= candidate.parent.name
        ]
        old_count = len(release_chain) - 1
        for path in release_chain[:old_count]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        env = {
            **os.environ,
            "DATABASE_URL": target_url,
            "DIRECT_DATABASE_URL": target_url,
        }

        async def deploy():
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "prisma",
                    "migrate",
                    "deploy",
                    "--schema",
                    str(schema / "schema.prisma"),
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr

        await deploy()
        # Explicit synthetic rows let the upgrade prove data preservation.
        user = SimpleNamespace(id=str(uuid4()))
        await client.execute_raw(
            "INSERT INTO users (id, username, updated_at) VALUES ($1, $2, CURRENT_TIMESTAMP)",
            user.id,
            "evidence-migration-" + uuid4().hex,
        )
        agent_id = str(uuid4())
        await client.execute_raw(
            "INSERT INTO ai_agents (id, name, user_id, updated_at) VALUES ($1, 'Synthetic', $2, CURRENT_TIMESTAMP)",
            agent_id,
            user.id,
        )
        agent = SimpleNamespace(id=agent_id)
        space = await client.chatworkspace.create(
            data={"userId": user.id, "agentId": agent.id}
        )
        mid = uuid4().hex
        for model in (client.usermemory, client.aimemory):
            await model.create(
                data={
                    "id": mid,
                    "userId": user.id,
                    "workspaceId": space.id,
                    "content": "合成原文保持不变",
                    "level": 2,
                    "importance": 0.61,
                }
            )
        for path in release_chain[old_count:]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        await deploy()
        await deploy()
        for model in (client.usermemory, client.aimemory):
            row = await model.find_unique(where={"id": mid})
            assert (
                row.content == "合成原文保持不变"
                and row.level == 2
                and row.importance == 0.61
            )
        assert await client.query_raw("SELECT id FROM memory_temporal_assertions") == []
        result = await client.query_raw(
            "SELECT count(*)::int AS n FROM _prisma_migrations WHERE finished_at IS NOT NULL"
        )
        assert result[0]["n"] == len(release_chain)
        # Restore into another owned database. A rollback keeps the additive
        # schema; it must not require dropping the new provenance table.
        pg_env = {
            **os.environ,
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGUSER": unquote(parsed.username or ""),
            "PGPASSWORD": unquote(parsed.password or ""),
        }
        dump = tmp_path / "validity-upgraded.dump"

        async def pg_command(args):
            result = await asyncio.to_thread(
                subprocess.run, args, env=pg_env, capture_output=True, timeout=60
            )
            assert result.returncode == 0, "isolated PostgreSQL backup/restore failed"

        await pg_command(
            [
                "pg_dump",
                "-Fc",
                "--no-owner",
                "--no-acl",
                "-d",
                database_name,
                "-f",
                str(dump),
            ]
        )
        await admin.execute_raw(f'CREATE DATABASE "{restore_name}"')
        await pg_command(
            [
                "pg_restore",
                "--exit-on-error",
                "--no-owner",
                "--no-acl",
                "-d",
                restore_name,
                str(dump),
            ]
        )
        restored = Prisma(
            datasource={"url": urlunsplit(parsed._replace(path="/" + restore_name))},
            http={"trust_env": False},
        )
        await restored.connect()
        try:
            assert (
                await restored.usermemory.find_unique(where={"id": mid})
            ).content == "合成原文保持不变"
            assert (
                await restored.aimemory.find_unique(where={"id": mid})
            ).importance == 0.61
            assert (
                await restored.query_raw("SELECT id FROM memory_temporal_assertions")
                == []
            )
            triggers = await restored.query_raw(
                "SELECT count(*)::int AS n FROM pg_trigger WHERE tgname='memory_temporal_assertion_validate'"
            )
            assert triggers[0]["n"] == 1
        finally:
            await restored.disconnect()
    finally:
        if client is not None:
            await client.disconnect()
        await admin.execute_raw(
            f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'
        )
        await admin.execute_raw(
            f'DROP DATABASE IF EXISTS "{restore_name}" WITH (FORCE)'
        )
        await admin.disconnect()
