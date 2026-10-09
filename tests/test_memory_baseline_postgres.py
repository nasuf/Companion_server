"""Fresh-process synthetic fixtures against migrated local PG and Redis."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    not os.environ.get("MEMORY_EVAL_TEST_DATABASE_URL"),
    reason="explicit disposable memory eval database required",
)
def test_real_scope_cache_and_cleanup_in_fresh_process():
    script = r"""
import asyncio, os
from unittest.mock import patch
from evals.memory_baseline.safety import configure_isolation, loopback_network_fence
configure_isolation(os.environ['MEMORY_EVAL_TEST_DATABASE_URL'],os.environ['MEMORY_EVAL_TEST_REDIS_URL'])
from evals.memory_baseline.dataset import load_cases
from evals.memory_baseline.replay import fixture,cleanup
from app.db import db
from app.redis_client import get_redis,close_redis
from app.services.memory.retrieval import vector_search,hybrid
from app.services.runtime import cache
from uuid import uuid4
async def run():
    case=next(c for c in load_cases() if c['group']=='multi_agent')
    vectors={m['text']:[1.]+[0.]*1023 for m in case['memories']}
    async def embed(_):return [1.]+[0.]*1023
    await db.connect();redis=await get_redis();prefix='memory-eval-test:'+uuid4().hex+':'
    own=None
    try:
        own=await fixture(db,case,vectors)
        owners,agents,spaces,keys=own
        assert spaces['target'][0]==spaces['other_agent'][0]
        assert spaces['target'][1]!=spaces['other_agent'][1]
        with patch.object(vector_search,'generate_embedding',embed),patch.object(cache,'CACHE_PREFIX',prefix+'cache:'),patch.object(cache,'VERSION_PREFIX',prefix+'version:'):
            for scope in ('target','other_agent','other_user'):
                uid,wid=spaces[scope]
                expected={m['key'] for m in case['memories'] if m['scope']==scope and not m.get('archived')}
                a=await hybrid.hybrid_retrieve(case['query'],uid,workspace_id=wid)
                b=await hybrid.hybrid_retrieve(case['query'],uid,workspace_id=wid)
                assert {keys[m.id] for m in a['memories']}<=expected
                assert [m.id for m in a['memories']]==[m.id for m in b['memories']]
                assert await cache.cache_retrieval(case['query'],uid,workspace_id=wid)
                await cache.bump_cache_version(uid,wid)
                assert await cache.cache_retrieval(case['query'],uid,workspace_id=wid) is None
        await cleanup(db,*own);own=None
        assert await db.user.count(where={'id':{'in':owners}})==0
        assert await db.chatworkspace.count(where={'id':{'in':[v[1] for v in spaces.values()]}})==0
    finally:
        if own:await cleanup(db,*own)
        async for key in redis.scan_iter(match=prefix+'*'):await redis.delete(key)
        await close_redis();await db.disconnect()
with loopback_network_fence() as violations:asyncio.run(run())
assert not violations
print('scope/cache/cleanup passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scope/cache/cleanup passed" in result.stdout
