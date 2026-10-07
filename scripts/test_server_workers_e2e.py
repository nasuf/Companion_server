"""Real image/supervisor regression: slow cold imports and worker replacement.

The disposable container has no network, DB, Redis or production credentials.
It imports the packaged app but replaces its ASGI target with a pure PID probe,
so no production startup jobs or business operations execute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

FIXTURE = '''import os,sys,time
# A deterministic GIL stall represents expensive cold model construction.
sys.setswitchinterval(15)
until=time.monotonic()+7
while time.monotonic()<until: pass
sys.setswitchinterval(0.005)
import app.main
from fastapi import FastAPI
app=FastAPI()
@app.get('/worker-probe')
async def worker_probe(): return {'pid':os.getpid()}
'''
PROBE = '''import urllib.request,json
pids=set()
for _ in range(40):
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/worker-probe',headers={'Connection':'close'}),timeout=2) as r:
  pids.add(json.loads(r.read())['pid'])
print(json.dumps(sorted(pids)))'''


def docker(*args, timeout=20):
    return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)


def serving_workers(name):
    result=docker('exec',name,'python','-c',PROBE,timeout=12)
    return set(json.loads(result.stdout)) if result.returncode==0 else set()


def wait_pair(name, previous=None, timeout=100):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        pids=serving_workers(name)
        if len(pids)==2 and (previous is None or pids!=previous):
            return pids
        time.sleep(1)
    raise AssertionError('Two initialized workers did not serve within deadline')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--healthcheck-timeout',type=int)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    name='companion-worker-test-'+uuid.uuid4().hex[:12]
    cmd_result=docker('image','inspect',args.image,'--format','{{json .Config.Cmd}}')
    assert cmd_result.returncode==0,cmd_result.stderr
    cmd=json.loads(cmd_result.stdout)
    assert cmd[:2]==['sh','-c'] and len(cmd)==3 and 'uvicorn app.main:app' in cmd[2]
    cmd[2]=cmd[2].replace('uvicorn app.main:app','uvicorn worker_fixture:app --app-dir /worker-test')
    started=time.monotonic()
    result={'passed':False,'image':args.image,'container':name}
    try:
        with tempfile.TemporaryDirectory(prefix='companion-worker-fixture-') as temp:
            Path(temp,'worker_fixture.py').write_text(FIXTURE)
            options=['run','-d','--name',name,'--network','none','--cpus','1','--memory','5g',
                     '-e','APP_ENV=test','-e','PYTHON_DOTENV_DISABLED=1',
                     '-e','JWT_SECRET=isolated-two-worker-test-secret-at-least-32-characters',
                     '-e','TRACE_BACKEND=off','-e','LANGSMITH_TRACING=false','-e','ONLINE_MODEL=false',
                     '-e','WEB_CONCURRENCY=2','-e','DATABASE_URL=postgresql://unused:unused@127.0.0.1:5432/unused',
                     '-e','DIRECT_DATABASE_URL=postgresql://unused:unused@127.0.0.1:5432/unused',
                     '-e','REDIS_URL=redis://127.0.0.1:6379/0','-v',temp+':/worker-test:ro']
            if args.healthcheck_timeout is not None:
                assert args.healthcheck_timeout>0
                options+=['-e',f'UVICORN_WORKER_HEALTHCHECK_TIMEOUT={args.healthcheck_timeout}']
            launch=docker(*options,'--entrypoint',cmd[0],args.image,*cmd[1:])
            assert launch.returncode==0,launch.stderr
            original=wait_pair(name,timeout=100)
            gate=[sys.executable,str(Path(__file__).with_name('verify_server_workers.py')),
                  '--container',name,'--workers','2','--timeout','20','--stable-seconds','3']
            ready=subprocess.run(gate,capture_output=True,text=True,timeout=30)
            assert ready.returncode==0,ready.stdout+ready.stderr
            # Only an initialized PID in our uniquely owned disposable container.
            victim=min(original)
            kill=docker('exec',name,'python','-c',f'import os,signal;os.kill({victim},signal.SIGKILL)')
            assert kill.returncode==0,kill.stderr
            unready=subprocess.run(gate,capture_output=True,text=True,timeout=30)
            assert unready.returncode!=0,'Deployment gate accepted a crashed startup worker'
            recovered=wait_pair(name,original,timeout=100)
            assert victim not in recovered and original-{victim}<=recovered
            time.sleep(12)
            assert serving_workers(name)==recovered,'Recovered workers were not stable'
            logs=docker('logs',name)
            lines=(logs.stdout+logs.stderr).splitlines()
            deaths=sum('Child process' in line and 'died' in line for line in lines)
            assert deaths==1,f'Unexpected supervisor deaths: {deaths}'
            result.update(passed=True,initial_pids=sorted(original),recovered_pids=sorted(recovered),
                          child_deaths=deaths,seconds=round(time.monotonic()-started,2),
                          checks=['two workers serve after GIL-blocking cold app import',
                                  'deployment gate verifies initialized stable workers',
                                  'deployment gate rejects a crashed startup worker',
                                  'one crashed worker is replaced while the survivor remains',
                                  'replacement remains stable without repeated restarts'])
    finally:
        logs=docker('logs',name)
        (args.output/'container.log').write_text(logs.stdout+logs.stderr)
        state=docker('inspect',name)
        (args.output/'container-state.json').write_text(state.stdout)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        docker('rm','-f',name)
    print(json.dumps(result))


if __name__=='__main__':
    main()
