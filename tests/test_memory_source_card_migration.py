"""Prisma upgrades populated source evidence without rewriting messages or memories."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from prisma import Prisma
import pytest


@pytest.mark.asyncio
async def test_prisma_card_version_upgrade_preserves_legacy_rows(tmp_path):
    # The proactive test service has pgvector, as required by the historical
    # migration chain. G03 deliberately uses plain PostgreSQL in Linux CI.
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Memory evidence migration requires disposable PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == "/companion_proactive_e2e"
    url = urlunsplit(parsed._replace(path="/postgres"))
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
        candidate = next(p for p in all_migrations if p.parent.name == "20261018234000_memory_source_card_version")
        release_chain = [p for p in all_migrations if p.parent.name <= candidate.parent.name]
        assert len(release_chain) == 173
        old_count = 172
        for path in release_chain[:old_count]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        env = {**os.environ, "DATABASE_URL": target_url, "DIRECT_DATABASE_URL": target_url}

        async def deploy():
            result = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(schema / "schema.prisma")],
                cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr

        await deploy()
        # 历史 schema 没有之后新增的 users 列 (如 is_internal); 只写当时存在的列。
        user = SimpleNamespace(id=str(uuid4()))
        await client.execute_raw(
            "INSERT INTO users (id, username, updated_at) VALUES ($1, $2, CURRENT_TIMESTAMP)",
            user.id, "evidence-migration-" + uuid4().hex,
        )
        # 历史版本的库只有当时的列; 用当前生成客户端的 aiagent.create 会去读写之后
        # 新增的列 (如 nickname) 而失败。基线数据只写历史 schema 里存在的列。
        agent_id = str(uuid4())
        await client.execute_raw(
            "INSERT INTO ai_agents (id, name, user_id, updated_at) VALUES ($1, 'Synthetic', $2, CURRENT_TIMESTAMP)",
            agent_id, user.id)
        agent = SimpleNamespace(id=agent_id)
        space = await client.chatworkspace.create(data={"userId": user.id, "agentId": agent.id})
        mid = uuid4().hex
        for model in (client.usermemory, client.aimemory):
            await model.create(data={"id": mid, "userId": user.id, "workspaceId": space.id,
                                     "content": "合成原文保持不变", "level": 2, "importance": .61})
        conv=await client.conversation.create(data={"userId":user.id,"agentId":agent.id,"workspaceId":space.id})
        msg=await client.message.create(data={"conversationId":conv.id,"role":"user","content":"原始音乐描述"})
        async def legacy_bind():
            await client.execute_raw("""INSERT INTO memory_evidence_links
                (id,memory_id,memory_source,user_memory_id,user_id,workspace_id,agent_id,content_version,
                 source_kind,source_ref,source_version,source_user_id,source_workspace_id,source_role,
                 source_status,source_message_id,relation,extractor_version)
                VALUES ($1,$2,'user',$2,$3,$4,$5,encode(sha256(convert_to('合成原文保持不变','UTF8')),'hex'),
                  'message',$6,encode(sha256(convert_to('原始音乐描述','UTF8')),'hex'),$3,$4,'user','recorded',$6,'extracted_from','legacy-test')""",
                uuid4().hex,mid,user.id,space.id,agent.id,msg.id)
        await legacy_bind()
        for path in release_chain[old_count:]:
            shutil.copytree(path.parent, migrations / path.parent.name)
        await deploy()
        await deploy()
        for model in (client.usermemory, client.aimemory):
            row = await model.find_unique(where={"id": mid})
            assert row.content == "合成原文保持不变" and row.level == 2 and row.importance == .61
        rows=await client.query_raw("SELECT source_card_version FROM memory_evidence_links")
        assert rows==[{"source_card_version":None}]
        await legacy_bind()  # old SQL contract remains valid after migration
        assert (await client.message.find_unique(where={"id":msg.id})).content=="原始音乐描述"
        assert len(await client.query_raw("SELECT id FROM memory_evidence_links"))==2
        result = await client.query_raw("SELECT count(*)::int AS n FROM _prisma_migrations WHERE finished_at IS NOT NULL")
        assert result[0]["n"] == 173
    finally:
        if client is not None:
            await client.disconnect()
        await admin.execute_raw(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
        await admin.disconnect()
