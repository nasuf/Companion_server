"""Reply repetition policy API: real migration, concurrency, audit and worker refresh."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.admin import reply_repetition_policy as api
from app.services import runtime_config
from app.services.chat.reply_repetition import breaker, policy
from tests.test_admin_proactive_tuning_postgres import env, _auth

URL = '/admin-api/reply-repetition-policy'


@pytest.fixture
async def policy_env(env, monkeypatch):
    db, _, admin = env
    monkeypatch.setattr(api, 'db', db)

    async def closed(effective, now=None):
        return breaker.judge({'turns': 12, 'rewrites': 0, 'window_seconds': 3600}, effective)

    monkeypatch.setattr(breaker, 'state', closed)
    original = (await db.query_raw('SELECT reply_repetition_policy FROM system_config WHERE id=1'))[0]['reply_repetition_policy']
    await db.execute_raw('UPDATE system_config SET reply_repetition_policy=NULL WHERE id=1')
    await runtime_config.load_caches()
    app = FastAPI(); app.include_router(api.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        try:
            yield SimpleNamespace(db=db, client=client, admin=admin, headers=_auth(admin.id))
        finally:
            await db.execute_raw('DELETE FROM reply_repetition_policy_changes WHERE admin_id=$1', admin.id)
            await db.execute_raw('UPDATE system_config SET reply_repetition_policy=$1::jsonb WHERE id=1',
                                 None if original is None else json.dumps(original))
            await runtime_config.load_caches()


async def get(io):
    r = await io.client.get(URL, headers=io.headers)
    assert r.status_code == 200, r.text
    return r.json()


async def put(io, values, revision=None):
    if revision is None:
        revision = (await get(io))['revision']
    return await io.client.put(URL, headers=io.headers, json={'values': values, 'expected_revision': revision})


async def test_defaults_edit_audit_restore_and_hot_path_refresh(policy_env):
    io = policy_env
    body = await get(io)
    assert body['rewrite_enabled'] is True and body['backstop_verbatim_overlap'] == 0.8
    assert body['stored'] == {} and body['breaker']['state'] == 'closed' and body['changes'] == []
    assert [p['key'] for p in body['params']][0] == 'rewrite_enabled'
    r = await put(io, {'rewrite_enabled': False, 'breaker_rate': 0.1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['rewrite_enabled'] is False and body['breaker_rate'] == 0.1
    assert policy.current().rewrite_enabled is False  # workers read the refreshed cache
    change = body['changes'][0]
    assert change['admin_name'] == io.admin.username and change['created_at'].endswith('+00:00')
    assert {c['key']: (c['before'], c['after']) for c in change['changes']} == {
        'rewrite_enabled': (True, False), 'breaker_rate': (0.05, 0.1)}
    # Restoring the default removes the override and is audited as such
    body = (await put(io, {'rewrite_enabled': None})).json()
    assert body['stored'] == {'breaker_rate': 0.1} and body['changes'][0]['changes'][0]['restored_default']
    # A no-op leaves no audit row
    assert len((await put(io, {'breaker_rate': 0.1})).json()['changes']) == 2


async def test_stale_revision_and_invalid_values_write_nothing(policy_env):
    io = policy_env
    stale = (await get(io))['revision']
    assert (await put(io, {'backstop_overlap': 0.7})).status_code == 200
    r = await put(io, {'backstop_overlap': 0.8}, revision=stale)
    assert r.status_code == 409
    for bad in ({'backstop_overlap': 2}, {'rewrite_enabled': 'no'}, {'unknown': 1}):
        r = await put(io, bad)
        assert r.status_code == 400, bad
    body = await get(io)
    assert body['stored'] == {'backstop_overlap': 0.7} and len(body['changes']) == 1


async def test_unreadable_stored_policy_is_reported_not_used(policy_env):
    io = policy_env
    await io.db.execute_raw('UPDATE system_config SET reply_repetition_policy=$1::jsonb WHERE id=1',
                            json.dumps({'rewrite_overlap': 7}))
    await runtime_config.load_caches()
    body = await get(io)
    assert body['rewrite_overlap'] == 0.6 and body['stored_policy_valid'] is False
    assert policy.current().rewrite_overlap == 0.6
