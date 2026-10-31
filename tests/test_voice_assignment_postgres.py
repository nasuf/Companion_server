"""Gender pools and persistent assignment through the real PostgreSQL queries."""
from datetime import timedelta
import os
from types import SimpleNamespace
from urllib.parse import urlsplit

from prisma import Prisma
import pytest

from app.services.speech_output import voices


async def test_gender_pool_excludes_disabled_voices_and_keeps_existing_binding(monkeypatch):
    url = os.getenv("TTS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TTS_TEST_DATABASE_URL must point to disposable local PostgreSQL")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    try:
        async with database.tx(timeout=timedelta(seconds=30)) as tx:
            await tx.execute_raw("""CREATE TEMP TABLE ai_agents (
                id TEXT PRIMARY KEY, gender TEXT, tts_voice_id TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            ) ON COMMIT DROP""")
            await tx.execute_raw("""CREATE TEMP TABLE tts_voice_profiles (
                voice_id TEXT PRIMARY KEY, gender TEXT, enabled BOOLEAN,
                provider TEXT DEFAULT 'dashscope',
                model TEXT DEFAULT 'qwen-audio-3.0-tts-plus'
            ) ON COMMIT DROP""")
            await tx.execute_raw("""INSERT INTO tts_voice_profiles
                (voice_id,gender,enabled) VALUES
                ('male-1','male',true), ('female-1','female',true),
                ('disabled','male',false)""")
            await tx.execute_raw("""INSERT INTO tts_voice_profiles
                (voice_id,gender,enabled,model) VALUES
                ('wrong-model','male',true,'different-model')""")
            await tx.execute_raw("""INSERT INTO tts_voice_profiles
                (voice_id,gender,enabled,provider) VALUES
                ('wrong-provider','male',true,'different-provider')""")
            monkeypatch.setattr(voices, "db", tx)
            for gender in ('male', 'female'):
                await tx.execute_raw("INSERT INTO ai_agents(id,gender) VALUES ($1,$2)", gender, gender)
                agent = SimpleNamespace(id=gender, gender=gender)
                assert await voices.ensure_agent_voice(agent) == f'{gender}-1'
                assert agent.ttsVoiceId == f'{gender}-1'

            # Growing the library affects new agents only. Even disabling an
            # old voice excludes it from draws without changing its owners.
            await tx.execute_raw("INSERT INTO tts_voice_profiles(voice_id,gender,enabled) VALUES ('male-2','male',true)")
            await tx.execute_raw("UPDATE tts_voice_profiles SET enabled=false WHERE voice_id='male-1'")
            assert await voices.ensure_agent_voice(SimpleNamespace(id='male', gender='male')) == 'male-1'
            assert await voices.assign_random_voice(agent_id='male', gender='male') == 'male-1'
            await tx.execute_raw("INSERT INTO ai_agents(id,gender) VALUES ('new-male','male')")
            assert await voices.assign_random_voice(agent_id='new-male', gender='male') == 'male-2'

            # Two enabled choices: repeated creations must remain inside the
            # appropriate pool; retries must keep the originally stored value.
            await tx.execute_raw("UPDATE tts_voice_profiles SET enabled=true WHERE voice_id='male-1'")
            for index in range(12):
                agent_id = f'new-{index}'
                await tx.execute_raw("INSERT INTO ai_agents(id,gender) VALUES ($1,'male')", agent_id)
                chosen = await voices.assign_random_voice(agent_id=agent_id, gender='male')
                assert chosen in {'male-1', 'male-2'}
                assert await voices.assign_random_voice(agent_id=agent_id, gender='male') == chosen
    finally:
        await database.disconnect()
