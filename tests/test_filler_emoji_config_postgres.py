"""Real authenticated API/SQL: global probability, atomic audit and worker refresh."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.admin import runtime_config as api
from app.services import runtime_config
from app.services.chat.filler_reply import build_filler_emoji_reply
from tests.test_admin_proactive_tuning_postgres import env, _auth


@pytest.fixture
async def filler_env(env, monkeypatch):
    db, _, admin = env
    monkeypatch.setattr(api, 'db', db)
    original = (await db.query_raw('SELECT filler_emoji_probability FROM system_config WHERE id=1'))[0]['filler_emoji_probability']
    await db.execute_raw('UPDATE system_config SET filler_emoji_probability=0 WHERE id=1')
    await runtime_config.load_caches()
    app = FastAPI(); app.include_router(api.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        try:
            yield SimpleNamespace(db=db, client=client, admin=admin, headers=_auth(admin.id))
        finally:
            await db.execute_raw('DELETE FROM runtime_config_changes WHERE admin_id=$1', admin.id)
            await db.execute_raw('UPDATE system_config SET filler_emoji_probability=$1 WHERE id=1', original)


async def put(io, value):
    return await io.client.put('/admin-api/runtime-config', headers=io.headers, json={'filler_emoji_probability': value})


async def history(io):
    response = await io.client.get('/admin-api/runtime-config/filler-emoji-history', headers=io.headers)
    assert response.status_code == 200, response.text
    return response.json()


async def test_real_default_write_history_and_restore(filler_env):
    io = filler_env
    assert build_filler_emoji_reply('好') is None
    before = (await io.db.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds FROM system_config WHERE id=1'))[0]
    response = await put(io, 1)
    assert response.status_code == 200, response.text
    assert response.json()['resolved']['filler_emoji_probability'] == 1
    assert build_filler_emoji_reply('好') is not None
    entries = await history(io)
    latest = entries['changes'][0]
    assert latest['before'] == 0 and latest['after'] == 1
    assert latest['admin_id'] == io.admin.id
    assert latest['created_at'].endswith('+00:00')
    assert entries['limit'] == 30
    await put(io, 1)
    assert len((await history(io))['changes']) == len(entries['changes'])
    assert (await put(io, None)).status_code == 200
    assert build_filler_emoji_reply('好') is None
    assert (await history(io))['changes'][0]['after'] == 0
    assert (await io.db.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds FROM system_config WHERE id=1'))[0] == before


async def test_concurrent_updates_serialize_exact_audit_chain(filler_env):
    io = filler_env
    results = await asyncio.gather(put(io, .25), put(io, .75))
    assert all(r.status_code == 200 for r in results), [r.text for r in results]
    entries = (await history(io))['changes'][:2]
    assert entries[1]['before'] == 0
    assert entries[0]['before'] == entries[1]['after']
    assert entries[0]['after'] == (await io.db.systemconfig.find_unique(where={'id':1})).fillerEmojiProbability


async def test_audit_failure_rolls_back_probability(filler_env):
    io = filler_env
    await io.db.execute_raw("CREATE FUNCTION filler_audit_failure() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'synthetic audit failure'; END $$")
    await io.db.execute_raw('CREATE TRIGGER filler_audit_failure BEFORE INSERT ON runtime_config_changes FOR EACH ROW EXECUTE FUNCTION filler_audit_failure()')
    try:
        with pytest.raises(Exception, match='synthetic audit failure'):
            await put(io, .4)
        assert (await io.db.systemconfig.find_unique(where={'id':1})).fillerEmojiProbability == 0
        assert build_filler_emoji_reply('好') is None
    finally:
        await io.db.execute_raw('DROP TRIGGER filler_audit_failure ON runtime_config_changes')
        await io.db.execute_raw('DROP FUNCTION filler_audit_failure()')


@pytest.mark.parametrize('value', [-.1, 1.01, 'NaN', 'Infinity'])
async def test_invalid_probability_never_writes(filler_env, value):
    assert (await put(filler_env, value)).status_code == 422
    assert (await filler_env.db.systemconfig.find_unique(where={'id':1})).fillerEmojiProbability == 0


async def test_permission_and_bounded_history(filler_env):
    io = filler_env
    response = await io.client.get('/admin-api/runtime-config/filler-emoji-history')
    assert response.status_code in (401, 403)
    for n in range(35):
        assert (await put(io, .25 if n%2 else .75)).status_code == 200
    assert len((await history(io))['changes']) == 30
    await io.db.user.update(where={'id':io.admin.id}, data={'role':'user'})
    assert (await put(io, 0)).status_code == 403
    assert (await io.client.get('/admin-api/runtime-config/filler-emoji-history', headers=io.headers)).status_code == 403


async def test_other_worker_refreshes_and_bound_snapshot_stays_stable(filler_env):
    io = filler_env
    snapshot = runtime_config.ConfigurationSnapshot({'fillerEmojiProbability':0}, {}, {}, {})
    assert (await put(io, 1)).status_code == 200
    token = runtime_config._current_snapshot.set(snapshot)
    try:
        assert build_filler_emoji_reply('好') is None
    finally:
        runtime_config._current_snapshot.reset(token)
    # Emulate a worker's old snapshot/revision; exercise the real revision query,
    # same refresh function the 2-second scheduler invokes.
    runtime_config._GLOBAL_CACHE = {'fillerEmojiProbability':0}
    runtime_config._CACHE_REVISION = -1
    await runtime_config.ensure_loaded()
    assert runtime_config.resolve_config_sync().filler_emoji_probability == 1
