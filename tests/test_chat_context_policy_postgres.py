"""Authenticated policy API: real migration, concurrency, audit and worker refresh."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.admin import chat_context_policy as api
from app.services import runtime_config
from app.services.chat import context_policy
from tests.test_admin_proactive_tuning_postgres import env, _auth


@pytest.fixture
async def policy_env(env, monkeypatch):
    db, _, admin = env
    monkeypatch.setattr(api, 'db', db)
    original = (await db.query_raw('SELECT chat_context_policy FROM system_config WHERE id=1'))[0]['chat_context_policy']
    await db.execute_raw('UPDATE system_config SET chat_context_policy=NULL WHERE id=1')
    await runtime_config.load_caches()
    app = FastAPI(); app.include_router(api.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        try:
            yield SimpleNamespace(db=db, client=client, admin=admin, headers=_auth(admin.id))
        finally:
            await db.execute_raw('DELETE FROM chat_context_policy_changes WHERE admin_id=$1', admin.id)
            await db.execute_raw('UPDATE system_config SET chat_context_policy=$1::jsonb WHERE id=1', None if original is None else json.dumps(original))


async def get(io):
    r = await io.client.get('/admin-api/chat-context-policy', headers=io.headers)
    assert r.status_code == 200, r.text
    return r.json()


async def put(io, rounds, revision=None, values=None):
    if revision is None:
        revision = (await get(io))['revision']
    return await io.client.put('/admin-api/chat-context-policy', headers=io.headers,
                              json={'values': values or {'rounds': rounds}, 'expected_revision': revision})


async def test_defaults_persistence_audit_noop_restore_and_cache(policy_env):
    io = policy_env
    original = await io.db.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds, filler_emoji_probability FROM system_config WHERE id=1')
    body = await get(io)
    assert body['effective_rounds'] == 5 and body['main_history_mode'] == 'budget'
    assert context_policy.current().rounds == 5
    r = await put(io, 8)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['effective_rounds'] == 8 and context_policy.current().rounds == 8
    assert body['changes'][0]['admin_name'] == io.admin.username
    assert body['changes'][0]['admin_id'] == io.admin.id  # fallback when the username is gone
    caps = body['token_estimate']['window_caps']
    assert caps['judgements'] == {'line_chars': 150, 'max_chars': 1500, 'grows_with_rounds': False}
    assert body['changes'][0]['changes'] == [{'key': 'rounds', 'before': 5, 'after': 8, 'stored_before': None, 'stored_after': 8, 'restored_default': False}]
    assert body['changes'][0]['created_at'].endswith('+00:00')
    assert (await put(io, 8)).json()['changes'] == body['changes']
    assert (await put(io, None)).json()['effective_rounds'] == 5
    assert (await io.db.query_raw('SELECT chat_context_policy FROM system_config WHERE id=1'))[0]['chat_context_policy'] is None
    assert await io.db.query_raw('SELECT reply_delay_enabled, reply_delay_max_seconds, filler_emoji_probability FROM system_config WHERE id=1') == original


async def test_main_history_length_saves_audits_and_restores(policy_env):
    io = policy_env
    assert (await get(io))['main_history_token_budget'] == 4000
    r = await put(io, None, values={'rounds': 8, 'main_history_tokens': 6000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['effective_rounds'], body['main_history_token_budget']) == (8, 6000)
    assert body['stored'] == {'rounds': 8, 'main_history_tokens': 6000}
    assert context_policy.current().main_history_tokens == 6000
    assert body['changes'][0]['changes'] == [
        {'key': 'rounds', 'before': 5, 'after': 8, 'stored_before': None, 'stored_after': 8, 'restored_default': False},
        {'key': 'main_history_tokens', 'before': 4000, 'after': 6000, 'stored_before': None, 'stored_after': 6000, 'restored_default': False}]
    assert [p['key'] for p in body['params']] == ['rounds', 'main_history_tokens']
    assert 'main' not in {x['key'] for x in body['exclusions']}
    body = (await put(io, None, values={'main_history_tokens': None})).json()
    assert (body['effective_rounds'], body['main_history_token_budget']) == (8, 4000)
    assert body['changes'][0]['changes'] == [
        {'key': 'main_history_tokens', 'before': 6000, 'after': 4000, 'stored_before': 6000, 'stored_after': None, 'restored_default': True}]
    assert context_policy.current().main_history_tokens == 4000
    # Resetting an invalid stored value stays visible in the audit.
    await io.db.execute_raw('UPDATE system_config SET chat_context_policy=$1::jsonb WHERE id=1',
                            json.dumps({'rounds': 8, 'main_history_tokens': 99999}))
    body = (await put(io, None, values={'main_history_tokens': None})).json()
    assert body['stored_policy_valid'] is True and body['stored'] == {'rounds': 8}
    assert body['changes'][0]['changes'] == [
        {'key': 'main_history_tokens', 'before': 4000, 'after': 4000, 'stored_before': 99999, 'stored_after': None, 'restored_default': True}]


@pytest.mark.parametrize('value', [499, 8001, True, '4000', 4000.5])
async def test_invalid_main_history_length_never_writes(policy_env, value):
    r = await put(policy_env, None, values={'main_history_tokens': value})
    assert r.status_code == 400 and '主回复历史长度' in r.json()['detail']
    assert (await get(policy_env))['changes'] == []


async def test_concurrent_writers_conflict_without_lost_update(policy_env):
    io = policy_env
    revision = (await get(io))['revision']
    results = await asyncio.gather(put(io, 8, revision), put(io, 12, revision))
    assert sorted(r.status_code for r in results) == [200, 409]
    body = await get(io)
    assert len(body['changes']) == 1
    assert body['changes'][0]['changes'][0]['after'] == body['effective_rounds']


async def test_audit_failure_rolls_back_policy(policy_env):
    io = policy_env
    await io.db.execute_raw("CREATE FUNCTION context_audit_failure() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'synthetic context audit failure'; END $$")
    await io.db.execute_raw('CREATE TRIGGER context_audit_failure BEFORE INSERT ON chat_context_policy_changes FOR EACH ROW EXECUTE FUNCTION context_audit_failure()')
    try:
        with pytest.raises(Exception, match='synthetic context audit failure'):
            await put(io, 8)
        assert (await get(io))['effective_rounds'] == 5
        assert context_policy.current().rounds == 5
    finally:
        await io.db.execute_raw('DROP TRIGGER context_audit_failure ON chat_context_policy_changes')
        await io.db.execute_raw('DROP FUNCTION context_audit_failure()')


@pytest.mark.parametrize('value', [0, 21, True, '5', 3.5])
async def test_invalid_never_writes(policy_env, value):
    assert (await put(policy_env, value)).status_code == 400
    assert (await get(policy_env))['changes'] == []


async def test_permissions_readonly_main_and_history_bound(policy_env):
    io = policy_env
    assert (await io.client.get('/admin-api/chat-context-policy')).status_code in (401, 403)
    r = await io.client.put('/admin-api/chat-context-policy', headers=io.headers,
                           json={'values': {'main_history_mode': 'rounds'}, 'expected_revision': (await get(io))['revision']})
    assert r.status_code == 400
    for n in range(32):
        assert (await put(io, 8 if n % 2 else 12)).status_code == 200
    assert len((await get(io))['changes']) == 30
    await io.db.user.update(where={'id': io.admin.id}, data={'role': 'user'})
    for path in ('', '/usage'):
        assert (await io.client.get('/admin-api/chat-context-policy' + path, headers=io.headers)).status_code == 403
    assert (await put(io, 5, 'synthetic')).status_code == 403


async def test_other_worker_reload_and_inflight_snapshot(policy_env):
    io = policy_env
    snapshot = runtime_config.ConfigurationSnapshot({'chatContextPolicy': {'rounds': 3}}, {}, {}, {})
    assert (await put(io, 8)).status_code == 200
    token = runtime_config._current_snapshot.set(snapshot)
    try:
        assert context_policy.current().rounds == 3
    finally:
        runtime_config._current_snapshot.reset(token)
    runtime_config._GLOBAL_CACHE = {'chatContextPolicy': {'rounds': 2}}
    runtime_config._CACHE_REVISION = -1
    await runtime_config.ensure_loaded()
    assert context_policy.current().rounds == 8


async def test_usage_empty_real_query(policy_env):
    r = await policy_env.client.get('/admin-api/chat-context-policy/usage?days=1', headers=policy_env.headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['rows'] == 0 and body['status'] == 'empty'
    assert body['groups'] == body['costs'] == []
    assert (await policy_env.client.get('/admin-api/chat-context-policy/usage?days=31', headers=policy_env.headers)).status_code == 422


async def test_usage_groups_actual_trace_rounds_unknown_cache_and_scope(policy_env):
    from uuid import uuid4
    io = policy_env
    run = str(uuid4())
    ids = [str(uuid4()) for _ in range(3)]
    await io.db.execute_raw("INSERT INTO trace_runs(run_id,trace_id,name,run_type,started_at,extra_json) VALUES ($1,$1,'synthetic','chain',CURRENT_TIMESTAMP,$2::jsonb)", run,
                            json.dumps({'metadata': {'chat_context_policy': {'effective_rounds': 8, 'main_history_token_budget': 6000}}}))
    try:
        ark, qwen = 'ark/doubao-seed-character-260628', 'dashscope/qwen3.5-flash'
        for ident, scope, trace, tokens in [(ids[0], 'chat', run, {ark: {'input': 1000, 'output': 20, 'cached_input': 500},
                                                                     qwen: {'input': 100, 'output': 2, 'cached_input': 0}}),
                                           (ids[1], 'chat', None, {ark: {'input': 200, 'output': 5, 'cached_input': 0}}),
                                           (ids[2], 'background', run, {ark: {'input': 99999, 'output': 1}})]:
            await io.db.execute_raw('INSERT INTO llm_usage(id,scope,trace_id,input_tokens,output_tokens,tokens_by_model,cost_cny,call_count,web_search_calls) VALUES($1,$2,$3,1,1,$4::jsonb,0.2,1,2)', ident, scope, trace, json.dumps(tokens))
        r = await io.client.get('/admin-api/chat-context-policy/usage?days=1', headers=io.headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['rows'] == 2
        groups = {(x['rounds'], x['main_history_tokens'], x['model']): x for x in body['groups']}
        assert groups[(8, 6000, ark)]['cache_hit_rate'] == .5
        # The ledger writes 0 for every provider; only Ark's 0 is a measured miss.
        assert groups[(8, 6000, qwen)]['cache_hit_rate'] is None
        assert groups[(8, 6000, qwen)]['cache_unknown_turns'] == 1
        assert groups[(None, None, ark)]['cache_hit_rate'] == 0
        assert len(body['costs']) == 2  # Multi-model turn cost is never counted twice.
        assert {(x['rounds'], x['main_history_tokens']) for x in body['costs']} == {(8, 6000), (None, None)}
        assert sum(x['recorded_token_cost_cny'] for x in body['costs']) == .4
        assert sum(x['normalized_search_cost_cny'] for x in body['costs']) == .016
    finally:
        for ident in ids:
            await io.db.execute_raw('DELETE FROM llm_usage WHERE id=$1', ident)
        await io.db.execute_raw('DELETE FROM trace_runs WHERE run_id=$1', run)
