"""Synthetic subprocess killed at actual Redis commit/response boundaries.

Never imported or copied into the production image. Guards run before app
imports or Redis connections. The only business effect is an isolated counter.
"""
import argparse
import asyncio
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['before_enqueue', 'after_enqueue', 'after_claim', 'after_promote', 'before_finish', 'after_finish', 'heartbeat'])
parser.add_argument('prefix')
parser.add_argument('--job-id', default='')
args = parser.parse_args()
url = os.environ.get('RUNTIME_JOB_TEST_REDIS_URL', '')
if (
    os.environ.get('APP_ENV') != 'test'
    or os.environ.get('N05_FAULT_TESTS') != 'synthetic-only'
    or urlsplit(url).hostname not in {'127.0.0.1', 'localhost', 'test-redis'}
    or not re.fullmatch(r'test:runtime-terminal:[0-9a-f]{32}:', args.prefix)
):
    raise RuntimeError('Fault worker refuses production or non-isolated execution')
os.environ.update(PYTHON_DOTENV_DISABLED='1', ONLINE_MODEL='false', TRACE_BACKEND='off')
for name in tuple(os.environ):
    if name.endswith('API_KEY') or name.startswith(('DASHSCOPE_', 'ARK_', 'ANTHROPIC_', 'LANGSMITH_', 'AXIOM_', 'MINIMAX_', 'DEEPSEEK_')):
        os.environ.pop(name, None)


def guarded(original):
    def connect(self, address):
        if isinstance(address, tuple):
            try:
                safe = address[0] == 'localhost' or ipaddress.ip_address(address[0]).is_loopback
            except ValueError:
                safe = False
            if not safe:
                raise RuntimeError('Synthetic worker attempted an external connection')
        return original(self, address)
    return connect


socket.socket.connect = guarded(socket.socket.connect)
socket.socket.connect_ex = guarded(socket.socket.connect_ex)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from redis.asyncio import Redis
from app.services.runtime import job_queue as q, distributed_lock


async def main():
    r = Redis.from_url(url, decode_responses=True)
    for name in ['_READY_KEY', '_DELAYED_KEY', '_RUNNING_KEY', '_DLQ_KEY', '_SUCCEEDED_KEY', '_JOB_KEY_PREFIX', '_IDEMP_KEY_PREFIX']:
        setattr(q, name, args.prefix + getattr(q, name))
    distributed_lock._KEY_PREFIX = args.prefix + 'lock'
    q._DEFAULT_LEASE_S = 1
    q._HEARTBEAT_INTERVAL_S = 0.1
    targets = {
        'before_enqueue': q._ENQUEUE_JOB_LUA, 'after_enqueue': q._ENQUEUE_JOB_LUA,
        'after_claim': q._CLAIM_JOB_LUA, 'after_promote': q._PROMOTE_JOB_LUA,
        'before_finish': q._FINISH_JOB_LUA, 'after_finish': q._FINISH_JOB_LUA,
    }
    async def barrier(job_id):
        await r.set(args.prefix + 'barrier', json.dumps({'job_id': job_id, 'mode': args.mode}), ex=60)
        await asyncio.Event().wait()
    class Backend:
        def __getattr__(self, name):
            return getattr(r, name)
        async def eval(self, script, numkeys, *values):
            target = targets.get(args.mode)
            jid = values[numkeys + 2] if script == q._ENQUEUE_JOB_LUA else values[numkeys]
            if script == target and args.mode.startswith('before_'):
                await barrier(jid)
            result = await r.eval(script, numkeys, *values)
            if script == target:
                await barrier(jid)
            return result
    backend = Backend()
    async def redis():
        return backend
    q.get_redis = redis
    async def handler(payload):
        await r.incr(args.prefix + 'invocations')
        await r.eval("if redis.call('SET', KEYS[1], '1', 'NX') then return redis.call('INCR', KEYS[2]) end return 0", 2, args.prefix + 'business-effect-identity', args.prefix + 'effects')
        if args.mode == 'heartbeat':
            await barrier(args.job_id)
    q.register_job_handler('test.job', handler)
    try:
        if 'enqueue' in args.mode:
            await q.enqueue_runtime_job('test.job', {'synthetic': True}, idempotency_key='same')
        elif args.mode == 'after_promote':
            await q._promote_due_jobs()
        else:
            await q._run_job_with_lock(args.job_id, require_ready=True)
    finally:
        await r.aclose()


asyncio.run(main())
