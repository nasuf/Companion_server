"""Prisma-only empty/populated rollout, old reader compatibility and DB bounds."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from prisma import Prisma


@pytest.mark.parametrize('populated', [False, True])
async def test_filler_prisma_upgrade_preserves_settings_and_redeploys(tmp_path, populated):
    configured = os.getenv('PROACTIVE_E2E_DATABASE_URL', '')
    if not configured:
        pytest.skip('Filler migration requires disposable PostgreSQL')
    url = urlsplit(configured)
    assert url.hostname in {'localhost', '127.0.0.1'} and url.path == '/companion_proactive_e2e'
    name = 'filler_migration_' + uuid4().hex
    target = urlunsplit(url._replace(path='/' + name))
    admin = Prisma(datasource={'url':urlunsplit(url._replace(path='/postgres'))}, http={'trust_env':False})
    client = None
    await admin.connect()
    try:
        await admin.execute_raw(f'CREATE DATABASE "{name}"')
        client = Prisma(datasource={'url':target}, http={'trust_env':False}); await client.connect()
        await client.execute_raw('CREATE SCHEMA extensions')
        await client.execute_raw('CREATE EXTENSION vector WITH SCHEMA extensions')
        schema = tmp_path/'prisma'; schema.mkdir(); migrations = schema/'migrations'; migrations.mkdir()
        shutil.copy2('prisma/schema.prisma', schema/'schema.prisma')
        shutil.copy2('prisma/migrations/migration_lock.toml', migrations/'migration_lock.toml')
        candidate = '20261025100000_filler_emoji_guard'
        chain = [p for p in sorted(Path('prisma/migrations').glob('*/migration.sql')) if p.parent.name <= candidate]
        for path in chain[:-1] if populated else chain:
            shutil.copytree(path.parent, migrations/path.parent.name)
        env = {**os.environ, 'DATABASE_URL':target, 'DIRECT_DATABASE_URL':target}
        async def deploy():
            p = await asyncio.to_thread(subprocess.run, [sys.executable,'-m','prisma','migrate','deploy','--schema',str(schema/'schema.prisma')],
                                         env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
            assert p.returncode == 0, p.stdout+p.stderr
        await deploy()
        if populated:
            await client.execute_raw('UPDATE system_config SET reply_delay_enabled=true, reply_delay_max_seconds=30 WHERE id=1')
            before = await client.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds FROM system_config WHERE id=1')
            shutil.copytree(chain[-1].parent, migrations/candidate)
        await deploy(); await deploy()
        if populated:
            assert await client.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds FROM system_config WHERE id=1') == before
        assert (await client.query_raw('SELECT filler_emoji_probability FROM system_config WHERE id=1'))[0]['filler_emoji_probability'] == 0
        assert await client.query_raw('SELECT id FROM runtime_config_changes') == []
        for value in [-1, 1.1, float('nan'), float('inf')]:
            with pytest.raises(Exception):
                await client.execute_raw('UPDATE system_config SET filler_emoji_probability=$1 WHERE id=1', value)
        assert (await client.query_raw('SELECT filler_emoji_probability FROM system_config WHERE id=1'))[0]['filler_emoji_probability'] == 0
        assert (await client.query_raw('SELECT count(*)::int n FROM _prisma_migrations WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL'))[0]['n'] == len(chain)
    finally:
        if client: await client.disconnect()
        await admin.execute_raw(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.disconnect()
