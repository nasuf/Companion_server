"""Upgrade the complete historical Prisma chain in a disposable database."""
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


@pytest.mark.asyncio
async def test_payment_gate_upgrade_defaults_closed_and_preserves_wallet(tmp_path, monkeypatch):
    url = os.environ.get('PROACTIVE_E2E_DATABASE_URL', '')
    if not url:
        pytest.skip('Disposable local pgvector PostgreSQL required')
    parsed = urlsplit(url)
    assert parsed.hostname in {'localhost', '127.0.0.1'} and parsed.path == '/companion_proactive_e2e'
    name = 'payment_gate_migration_' + uuid4().hex
    admin = Prisma(datasource={'url': urlunsplit(parsed._replace(path='/postgres'))}, http={'trust_env': False})
    target = urlunsplit(parsed._replace(path='/' + name))
    await admin.connect()
    client = None
    try:
        await admin.execute_raw(f'CREATE DATABASE "{name}"')
        client = Prisma(datasource={'url': target}, http={'trust_env': False})
        await client.connect()
        await client.execute_raw('CREATE SCHEMA extensions')
        await client.execute_raw('CREATE EXTENSION vector WITH SCHEMA extensions')
        schema = tmp_path / 'prisma'; schema.mkdir()
        shutil.copy2('prisma/schema.prisma', schema / 'schema.prisma')
        migrations = schema / 'migrations'; migrations.mkdir()
        shutil.copy2('prisma/migrations/migration_lock.toml', migrations / 'migration_lock.toml')
        candidate = Path('prisma/migrations/20261019090000_real_money_payment_gate')
        for source in sorted(Path('prisma/migrations').glob('*/migration.sql')):
            if source.parent.name < candidate.name:
                shutil.copytree(source.parent, migrations / source.parent.name)
        env = {**os.environ, 'DATABASE_URL': target, 'DIRECT_DATABASE_URL': target}
        async def deploy():
            result = await asyncio.to_thread(subprocess.run,
                [sys.executable, '-m', 'prisma', 'migrate', 'deploy', '--schema', str(schema / 'schema.prisma')],
                env=env, cwd=tmp_path, text=True, capture_output=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
        await deploy()
        await client.execute_raw('INSERT INTO system_config (id,offline_activity_enabled,offline_gift_enabled,updated_at) VALUES (1,TRUE,TRUE,CURRENT_TIMESTAMP) ON CONFLICT (id) DO UPDATE SET offline_activity_enabled=TRUE,offline_gift_enabled=TRUE')
        user = await client.user.create(data={'username': 'synthetic-payment-' + uuid4().hex})
        await client.execute_raw("INSERT INTO user_wallets (id,user_id,ticket_balance,point_balance,vip_until,updated_at) VALUES ($1,$2,123,456,'2030-01-01',CURRENT_TIMESTAMP)", str(uuid4()), user.id)
        before = await client.query_raw('SELECT * FROM user_wallets WHERE user_id=$1', user.id)
        shutil.copytree(candidate, migrations / candidate.name)
        await deploy(); await deploy()  # second deployment is a no-op, not a reset
        from app.services.payments import availability
        monkeypatch.setattr(availability, 'db', client)
        assert await availability.get_payment_settings() == {'real_money_payments_enabled': False}
        for enabled in (True, False):
            assert (await availability.set_payment_settings(enabled=enabled, actor_id='synthetic-admin'))['real_money_payments_enabled'] is enabled
            assert (await client.systemconfig.find_unique(where={'id': 1})).offlineGiftEnabled is True
            assert (await client.systemconfig.find_unique(where={'id': 1})).offlineActivityEnabled is True
        assert await client.query_raw('SELECT * FROM user_wallets WHERE user_id=$1', user.id) == before
        assert (await client.systemconfig.find_unique(where={'id': 1})).realMoneyPaymentsEnabled is False
    finally:
        if client is not None: await client.disconnect()
        await admin.execute_raw(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.disconnect()
