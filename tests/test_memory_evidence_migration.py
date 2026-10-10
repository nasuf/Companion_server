"""Prisma upgrades populated 154/155 baselines; old generated client still reads."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from prisma import Prisma
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("old_count", [154, 155])
async def test_prisma_evidence_upgrade_is_additive_and_redeployable(tmp_path, old_count):
    url = os.environ.get("G03_TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("Memory evidence migration requires disposable PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == "/postgres"
    database_name = "evidence_migration_" + uuid4().hex
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
        shutil.copy2("prisma/migrations/migration_lock.toml", migrations / "migration_lock.toml")
        all_migrations = sorted(Path("prisma/migrations").glob("*/migration.sql"))
        # This release's explicit historical upgrade cases stay reproducible as
        # later migrations are added to the repository.
        candidate = next(p for p in all_migrations if p.parent.name == "20261010100000_memory_evidence_links")
        release_chain = [p for p in all_migrations if p.parent.name <= candidate.parent.name]
        assert len(release_chain) == 156
        for path in release_chain[:old_count]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        env = {**os.environ, "DATABASE_URL": target_url, "DIRECT_DATABASE_URL": target_url}

        async def deploy():
            result = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(schema / "schema.prisma")],
                cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr

        await deploy()
        user = await client.user.create(data={"username": "evidence-migration-" + uuid4().hex})
        agent = await client.aiagent.create(data={"userId": user.id, "name": "Synthetic"})
        space = await client.chatworkspace.create(data={"userId": user.id, "agentId": agent.id})
        mid = uuid4().hex
        for model in (client.usermemory, client.aimemory):
            await model.create(data={"id": mid, "userId": user.id, "workspaceId": space.id,
                                     "content": "合成原文保持不变", "level": 2, "importance": .61})
        for path in release_chain[old_count:]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        await deploy()
        await deploy()
        for model in (client.usermemory, client.aimemory):
            row = await model.find_unique(where={"id": mid})
            assert row.content == "合成原文保持不变" and row.level == 2 and row.importance == .61
        assert await client.query_raw("SELECT id FROM memory_evidence_links") == []
        result = await client.query_raw("SELECT count(*)::int AS n FROM _prisma_migrations WHERE finished_at IS NOT NULL")
        assert result[0]["n"] == 156
    finally:
        if client is not None:
            await client.disconnect()
        await admin.execute_raw(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
        await admin.disconnect()
