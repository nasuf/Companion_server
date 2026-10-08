"""Real packaged role processes with owned PostgreSQL/Redis and synthetic data.

Never loads .env, publishes host ports, or connects to a production service.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import time
import uuid


def run(*args, timeout=25, check=True):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('Isolated docker operation failed: '+result.stderr[-800:])
    return result


def health(name):
    probe = """import urllib.request,urllib.error,json
try: r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)
except urllib.error.HTTPError as error: r=error
print(json.dumps({'code':r.status,'body':json.loads(r.read())}))"""
    result = run('exec', name, 'python', '-c', probe, check=False, timeout=10)
    return json.loads(result.stdout) if result.returncode==0 else None


def wait_ready(name, role=None, timeout=120):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        result = health(name)
        if result and result['code']==200:
            body = result['body']
            if role:
                assert body['role']==role and body['ready']
            else:
                assert body['postgres'] and body['redis']
            return
        time.sleep(1)
    raise AssertionError('Isolated role did not become ready: '+name)


def worker_probe(name, code):
    result = run('exec', name, 'python', '-c', code, timeout=45)
    return json.loads(result.stdout.strip().splitlines()[-1])


def wait_postgres_database(name, database, timeout=40):
    # The official image's temporary bootstrap server accepts Unix-socket
    # connections before POSTGRES_DB exists. Require a query against the target
    # database over TCP, which is opened only by the final server.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = run('exec', '-e', 'PGPASSWORD=synthetic', name, 'psql',
                    '-h', '127.0.0.1', '-U', 'postgres', '-d', database,
                    '-Atqc', 'SELECT 1', check=False, timeout=5)
        if probe.returncode == 0 and probe.stdout.strip() == '1':
            return
        time.sleep(1)
    raise AssertionError('Owned PostgreSQL target database failed to start')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = 'companion-role-e2e-'+uuid.uuid4().hex[:12]
    network = prefix+'-net'
    pg, redis = prefix+'-pg', prefix+'-redis'
    names = [pg, redis, prefix+'-api', prefix+'-background', prefix+'-scheduler']
    api, background, scheduler = names[2:]
    result = {'passed':False, 'image':args.image, 'checks':[], 'production_connections':0}
    url = f'postgresql://postgres:synthetic@{pg}:5432/companion_role_e2e'
    redis_url = f'redis://{redis}:6379/13'
    environment = [
        '-e','APP_ENV=test','-e','PYTHON_DOTENV_DISABLED=1',
        '-e',f'DATABASE_URL={url}','-e',f'DIRECT_DATABASE_URL={url}',
        '-e',f'REDIS_URL={redis_url}','-e','JWT_SECRET=isolated-runtime-roles-test-secret-at-least-32-characters',
        '-e','ONLINE_MODEL=false','-e','TRACE_BACKEND=off','-e','LANGSMITH_TRACING=false',
        '-e','WEB_CONCURRENCY=2','-e','LLM_PROCESS_COUNT=4',
        '-e','DB_CONNECTION_LIMIT=3','-e','DB_CONNECTION_LIMIT_MAX=3',
    ]
    try:
        run('network','create','--internal',network)
        run('run','-d','--name',pg,'--network',network,'--memory','512m',
            '-e','POSTGRES_PASSWORD=synthetic','-e','POSTGRES_DB=companion_role_e2e',
            'pgvector/pgvector:0.8.0-pg16')
        run('run','-d','--name',redis,'--network',network,'--memory','128m','redis:7-alpine')
        wait_postgres_database(pg, 'companion_role_e2e')
        run('exec',pg,'psql','-U','postgres','-d','companion_role_e2e','-c',
            'CREATE SCHEMA extensions; CREATE EXTENSION vector WITH SCHEMA extensions;')
        migration=run('run','--rm','--network',network,*environment,'--entrypoint','prisma',
            args.image,'migrate','deploy',timeout=180)
        (args.output/'migration.log').write_text(migration.stdout+ migration.stderr)
        result['checks'].append('full Prisma migrations on owned synthetic PostgreSQL')

        for role,name in [('background',background),('scheduler',scheduler),('api',api)]:
            command=['run','-d','--name',name,'--network',network,'--memory','4g',
                     '--init','--restart','on-failure:1',*environment,'-e',f'APP_RUNTIME_ROLE={role}',
                     '--entrypoint','python',args.image,'-m','uvicorn',
                     'app.main:app' if role=='api' else 'jobs.runtime:app',
                     '--host','0.0.0.0','--port','8000','--workers','2' if role=='api' else '1',
                     '--timeout-worker-healthcheck','60']
            run(*command)
            wait_ready(name, None if role=='api' else role)
        logs=run('logs',api)
        assert 'Job scheduler started' not in logs.stdout+logs.stderr
        from verify_server_workers import snapshot
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            gate=snapshot(['docker'], api, 2)
            if gate['child_deaths'] or gate['startup_failures']:
                raise AssertionError(gate)
            if gate['ready']:
                break
            time.sleep(1)
        assert gate['ready'], gate
        result['checks'].append('two API workers start without business scheduler')
        result['checks'].append('scheduler and background expose independent dependency readiness')

        # A real, centrally registered handler reads a nonexistent synthetic agent.
        # No model call, user record, or production identifier is needed.
        job_id=worker_probe(background,"""import asyncio,json
from app.services.runtime.job_queue import enqueue_runtime_job
async def main():
 print(json.dumps(await enqueue_runtime_job('agent_initialization', {'agent_id':'00000000-0000-0000-0000-000000000001','user_id':'00000000-0000-0000-0000-000000000002'},idempotency_key='isolated-role-initialization')))
asyncio.run(main())""")
        def status():
            return run('exec',redis,'redis-cli','-n','13','HGET','runtime:job:'+job_id,'status').stdout.strip()
        deadline=time.monotonic()+35
        while time.monotonic()<deadline and status()!='succeeded':
            time.sleep(1)
        assert status()=='succeeded'
        result['checks'].append('independent worker consumes centrally registered initialization handler')
        raw=run('exec',redis,'redis-cli','-n','13','HGETALL','scheduler:health').stdout
        assert 'aggregation_scan:ok_at' in raw
        result['checks'].append('independent scheduler retains delayed-turn scan and distributed health')

        # Never pause a shared service; both dependencies were created by this test.
        run('pause',redis)
        deadline=time.monotonic()+35
        observed=None
        while time.monotonic()<deadline:
            observed=health(background)
            if observed and observed['code']==503:
                break
            time.sleep(1)
        assert observed and observed['code']==503 and not observed['body']['redis'], 'Background falsely ready while Redis is paused'
        run('unpause',redis)
        wait_ready(background,'background',timeout=35)
        result['checks'].append('Redis loss removes readiness; recovery restores it')

        # Kill the owned runtime PID inside its container. `docker kill` is a
        # manual container stop and suppresses Docker's automatic restart policy.
        before=json.loads(run('inspect',background).stdout)[0]
        logs=run('logs',background)
        pids=re.findall(r'Started server process \[(\d+)\]',logs.stdout+logs.stderr)
        assert pids and int(pids[-1])>1
        kill=run('exec',background,'python','-c',
                 f'import os,signal;os.kill({int(pids[-1])},signal.SIGKILL)',check=False)
        assert kill.returncode in (0,137), kill.stderr
        wait_ready(background,'background')
        inspection=json.loads(run('inspect',background).stdout)[0]
        result['restart_probe']={'before':before['RestartCount'],'after':inspection['RestartCount'],'oom':inspection['State']['OOMKilled']}
        assert inspection['RestartCount']==before['RestartCount']+1 and not inspection['State']['OOMKilled']
        result['checks'].append('own worker process is replaced after SIGKILL')
        started=time.monotonic()
        run('stop','--time','15',background,timeout=20)
        inspection=json.loads(run('inspect',background).stdout)[0]
        shutdown_logs=run('logs','--since',inspection['State']['StartedAt'],background)
        output=shutdown_logs.stdout+shutdown_logs.stderr
        result['shutdown_probe']={'exit_code':inspection['State']['ExitCode'],'seconds':round(time.monotonic()-started,2),'application_shutdown_complete':'Application shutdown complete.' in output}
        # Uvicorn restores handlers and re-raises captured SIGTERM after a
        # graceful shutdown. 143 is expected; forced SIGKILL/137 is rejected.
        assert inspection['State']['ExitCode'] in (0,143) and time.monotonic()-started<18
        assert 'Application shutdown complete.' in output and 'deadline exceeded' not in output
        result['checks'].append('cooperative worker drains and exits on SIGTERM')
        rows=run('exec',pg,'psql','-U','postgres','-d','companion_role_e2e','-At','-c',
            'SELECT (SELECT count(*) FROM agent_runs)+(SELECT count(*) FROM runtime_jobs)+(SELECT count(*) FROM runtime_outbox);').stdout.strip()
        assert rows=='0'
        result['checks'].append('staged SQL consumers remain disabled')
        result['passed']=True
    finally:
        run('unpause',redis,check=False)
        for name in reversed(names):
            logs=run('logs',name,check=False)
            (args.output/(name+'.log')).write_text(logs.stdout+logs.stderr)
            run('rm','-fv',name,check=False)
        run('network','rm',network,check=False)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
