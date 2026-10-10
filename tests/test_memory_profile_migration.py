"""157 migration on empty/populated DBs, safe lock failure and old-client reads."""
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit,urlunsplit
from uuid import uuid4

import pytest
from prisma import Prisma
from app.services.memory.evidence import content_version


@pytest.mark.parametrize('old_count',[0,156])
async def test_profile_prisma_upgrade_preserves_old_data_and_recovers_lock_failure(tmp_path,old_count):
    configured=os.getenv('PROACTIVE_E2E_DATABASE_URL','')
    if not configured:pytest.skip('Profile migration requires disposable PostgreSQL')
    url=urlsplit(configured)
    assert url.hostname in {'localhost','127.0.0.1'} and url.path=='/companion_proactive_e2e'
    name='profile_migration_'+uuid4().hex
    admin=Prisma(datasource={'url':urlunsplit(url._replace(path='/postgres'))},http={'trust_env':False})
    target_url=urlunsplit(url._replace(path='/'+name))
    await admin.connect();client=None
    try:
        await admin.execute_raw(f'CREATE DATABASE "{name}"')
        client=Prisma(datasource={'url':target_url},http={'trust_env':False});await client.connect()
        await client.execute_raw('CREATE SCHEMA extensions')
        await client.execute_raw('CREATE EXTENSION vector WITH SCHEMA extensions')
        schema=tmp_path/'prisma';schema.mkdir();migrations=schema/'migrations';migrations.mkdir()
        shutil.copy2('prisma/schema.prisma',schema/'schema.prisma')
        shutil.copy2('prisma/migrations/migration_lock.toml',migrations/'migration_lock.toml')
        chain=sorted(Path('prisma/migrations').glob('*/migration.sql'))
        candidate='20261010200000_memory_profile_origins'
        chain=[p for p in chain if p.parent.name<=candidate]
        assert len(chain)==157
        for path in chain[:old_count or 157]:shutil.copytree(path.parent,migrations/path.parent.name)
        env={**os.environ,'DATABASE_URL':target_url,'DIRECT_DATABASE_URL':target_url}
        async def command(*args,success=True):
            result=await asyncio.to_thread(subprocess.run,[sys.executable,'-m','prisma','migrate',*args,
                '--schema',str(schema/'schema.prisma')],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=90)
            if success:assert result.returncode==0,result.stdout+result.stderr
            else:assert result.returncode!=0 and 'current transaction is aborted' in result.stderr+result.stdout
            return result
        await command('deploy')
        user=await client.user.create(data={'username':'profile-migration-'+uuid4().hex})
        agent=await client.aiagent.create(data={'userId':user.id,'name':'Synthetic'})
        space=await client.chatworkspace.create(data={'userId':user.id,'agentId':agent.id})
        conv=await client.conversation.create(data={'userId':user.id,'agentId':agent.id,'workspaceId':space.id})
        msg=await client.message.create(data={'conversationId':conv.id,'role':'user','content':'Synthetic source'})
        mid=uuid4().hex
        for model in (client.usermemory,client.aimemory):
            await model.create(data={'id':mid,'userId':user.id,'workspaceId':space.id,'content':'Synthetic unchanged',
                'level':2,'importance':.61,'provenance':'profile_seed'})
        if old_count:
            # Bind using the old schema through SQL, exactly as the old writer did.
            await client.execute_raw("""INSERT INTO memory_evidence_links
                (id,memory_id,memory_source,user_memory_id,user_id,workspace_id,agent_id,
                 content_version,source_kind,source_ref,source_version,source_user_id,source_workspace_id,
                 source_role,source_status,source_message_id,relation,extractor_version)
                VALUES ($1,$2,'user',$2,$3,$4,$5,$6,'message',$7,$8,$3,$4,'user','recorded',$7,'extracted_from','old-client-v1')""",
                uuid4().hex,mid,user.id,space.id,agent.id,content_version('Synthetic unchanged'),msg.id,content_version(msg.content))
            before=await client.query_raw('SELECT id,content_version,source_version FROM memory_evidence_links')
            shutil.copytree(chain[-1].parent,migrations/candidate)
            # A long reader blocks ALTER TABLE. Timeout rolls back the whole SQL
            # migration: no partial origin table survives and no manual prod fix.
            async with client.tx(timeout=timedelta(seconds=30)) as tx:
                await tx.query_raw('SELECT id FROM memory_evidence_links')
                started=time.monotonic()
                await command('deploy',success=False)
                assert time.monotonic()-started>=4.5
                assert (await tx.query_raw("SELECT to_regclass('memory_profile_origins')::text AS name"))[0]['name'] is None
            await command('resolve','--rolled-back',candidate)
            await command('deploy')
            assert await client.query_raw('SELECT id,content_version,source_version FROM memory_evidence_links')==before
        await command('deploy')
        for model in (client.usermemory,client.aimemory):
            row=await model.find_unique(where={'id':mid})
            assert row.content=='Synthetic unchanged' and row.level==2 and row.importance==.61
        assert await client.query_raw('SELECT id FROM memory_profile_origins')==[]
        count=await client.query_raw('SELECT count(*)::int n FROM _prisma_migrations WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL')
        assert count[0]['n']==157
    finally:
        if client:await client.disconnect()
        await admin.execute_raw(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.disconnect()
