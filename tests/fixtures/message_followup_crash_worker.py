"""Owned fixture: exit after a SQL effect or before Redis acknowledges it."""
import asyncio
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def local_url(name,scheme,path):
    parts=urlsplit(os.environ[name])
    assert parts.scheme==scheme and parts.hostname in {'127.0.0.1','localhost'}
    assert parts.port is not None and parts.path==path
    assert not parts.query and not parts.fragment
    # CI calls its loopback services localhost; the network fence requires a
    # literal address. Normalize only this explicitly validated local alias.
    credentials=parts.netloc.rsplit('@',1)[0]+'@' if '@' in parts.netloc else ''
    return parts._replace(netloc=f'{credentials}127.0.0.1:{parts.port}').geturl()

url=local_url('PROACTIVE_E2E_DATABASE_URL','postgresql','/companion_proactive_e2e')
redis_url=local_url('RUNTIME_JOB_TEST_REDIS_URL','redis','/15')
assert not (ROOT/'.env').exists()
from evals.memory_baseline.safety import configure_isolation, loopback_network_fence
configure_isolation(url.rsplit('/',1)[0]+'/companion_memory_eval_ci',redis_url.rsplit('/',1)[0]+'/13')
os.environ.update(DATABASE_URL=url,DIRECT_DATABASE_URL=url)
from app.db import db
from app.services.achievements import engine,repository,mode,source
from app.services.runtime import message_followups as f,job_queue as q,distributed_lock
from redis.asyncio import Redis

jid,phase,keys=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
assert phase in {'after_effect','before_ack'}
for name,value in keys.items():
    assert value.startswith('test:runtime-terminal:')
    setattr(q,name,value)
distributed_lock._KEY_PREFIX=keys['_READY_KEY'].removesuffix('runtime:jobs:ready')+'lock'
redis=Redis.from_url(redis_url,decode_responses=True)
async def get_redis():return redis
q.get_redis=get_redis;distributed_lock.get_redis=get_redis
async def enabled():return True
engine.achievement_evaluation_enabled=enabled
mode.achievement_evaluation_enabled=enabled
async def effect(event):
    s=source.current_source.get()
    await repository.record_event(user_id=s.user_id,agent_id=s.agent_id,workspace_id=s.workspace_id,
        conversation_id=s.conversation_id,event_type=s.role+'_message',source_id=s.message_id,value_int=2)
    if phase=='after_effect':os._exit(73)
engine._evaluate_event=effect
async def handler(payload):
    await f.run_message_followup(payload)
    os._exit(73)
q.register_job_handler(f.JOB_TYPE,handler)
async def main():
    await db.connect()
    await q._run_job(jid)
with loopback_network_fence():asyncio.run(main())
raise RuntimeError('crash boundary not reached')
