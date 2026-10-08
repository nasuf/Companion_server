"""Faults only in owned, network-isolated containers using the image's real CMD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import uuid

from test_server_workers_e2e import docker, wait_pair

FIXTURE = '''import json,os,re,faulthandler,signal
from pathlib import Path
def hold_gil():
 synthetic_private_value='DIAGNOSTIC_PRIVATE_LOCAL_CANARY_7951'
 re.fullmatch(r'(a+)+', 'a'*100+'!')
async def app(scope, receive, send):
 if scope['type']=='lifespan':
  while True:
   event=await receive()
   if event['type']=='lifespan.startup':
    if os.getenv('WORKER_FIXTURE_FAIL_STARTUP')=='1':
     await send({'type':'lifespan.startup.failed','message':'isolated startup failure'})
     return
    await send({'type':'lifespan.startup.complete'})
   elif event['type']=='lifespan.shutdown':
    await send({'type':'lifespan.shutdown.complete'})
    return
 elif scope['type']=='http':
  if scope['path'] in {'/stall-unregistered','/stall-handler-removed'}:
   faulthandler.unregister(signal.SIGUSR2)
   if scope['path']=='/stall-unregistered':
    (Path(os.environ['COMPANION_WORKER_DIAGNOSTIC_SESSION'])/f'{os.getpid()}.ready').unlink()
  await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
  await send({'type':'http.response.body','body':json.dumps({'pid':os.getpid()}).encode()})
  if scope['path'].startswith('/stall'):hold_gil()
'''

REGISTRATIONS = '''from pathlib import Path
import json,os
folders=list(Path('/tmp').glob('companion-worker-diagnostic-*'))
assert len(folders)==1
folder=folders[0];session=json.loads((folder/'session.json').read_text());rows=[]
for marker in folder.glob('*.ready'):
 data=json.loads(marker.read_text());trace=folder/f"{data['pid']}.trace"
 rows.append({'pid':data['pid'],'marker_mode':oct(marker.stat().st_mode&0o777),'trace_mode':oct(trace.stat().st_mode&0o777),'trace_bytes':trace.stat().st_size})
print(json.dumps({'parent_pid':session['parent_pid'],'directory_mode':oct(folder.stat().st_mode&0o777),'registrations':rows,'trace_files':len(list(folder.glob('*.trace')))}))'''


def registrations(name):
    r=docker('exec',name,'python','-c',REGISTRATIONS)
    assert r.returncode==0,r.stderr
    result=json.loads(r.stdout)
    assert result['directory_mode']=='0o700'
    assert all(row['marker_mode']==row['trace_mode']=='0o600' for row in result['registrations'])
    return result


def trigger_gil_stall(name, *, scenario):
    path={'unregistered':'/stall-unregistered','handler-removed':'/stall-handler-removed'}.get(scenario,'/stall')
    r=docker('exec',name,'python','-c',
             'import json,urllib.request;print(urllib.request.urlopen('+repr('http://127.0.0.1:8000'+path)+',timeout=5).read().decode())')
    assert r.returncode==0,r.stderr
    return json.loads(r.stdout)['pid']


def launch(image, name, folder, *, fail_startup=False):
    command = docker('image', 'inspect', image, '--format', '{{json .Config.Cmd}}')
    assert command.returncode == 0
    cmd = json.loads(command.stdout)
    assert cmd[:2] == ['sh', '-c'] and len(cmd) == 3 and 'app.api_server app.main:app' in cmd[2]
    cmd[2] = cmd[2].replace('app.api_server app.main:app',
                          'app.api_server worker_fixture:app --app-dir /worker-test')
    options = ['run', '-d', '--name', name, '--network', 'none', '--cpus', '1', '--memory', '512m',
               '-e', 'WEB_CONCURRENCY=2', '-e', 'UVICORN_WORKER_HEALTHCHECK_TIMEOUT=5',
               '-e', 'WORKER_FIXTURE_FAIL_STARTUP=' + ('1' if fail_startup else '0'),
               '-v', str(folder) + ':/worker-test:ro', '--entrypoint', cmd[0], image, *cmd[1:]]
    result = docker(*options)
    assert result.returncode == 0, result.stderr


def capture(name, folder):
    logs = docker('logs', name)
    text = logs.stdout + '\n' + logs.stderr
    (folder / 'container.log').write_text(text)
    records = [json.loads(line.split('worker_diagnostic ', 1)[1])
               for line in text.splitlines() if 'worker_diagnostic ' in line]
    state = docker('inspect', name, '--format', '{{json .State}}')
    if state.returncode == 0:
        (folder / 'container-state.json').write_text(state.stdout)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'passed': False, 'image': args.image, 'checks': []}
    with tempfile.TemporaryDirectory(prefix='companion-worker-diagnostic-fixture-') as temp:
        folder = Path(temp)
        (folder / 'worker_fixture.py').write_text(FIXTURE)
        for scenario in ('heartbeat', 'gil-held', 'unregistered', 'handler-removed', 'hup', 'startup-failure'):
            name = 'companion-worker-diagnostic-' + uuid.uuid4().hex[:12]
            output = args.output / scenario
            output.mkdir(exist_ok=True)
            try:
                launch(args.image, name, folder, fail_startup=scenario == 'startup-failure')
                if scenario != 'startup-failure':
                    original = wait_pair(name, timeout=30)
                    initial=registrations(name)
                    assert {row['pid'] for row in initial['registrations']}==original
                    assert initial['trace_files']==2 and all(row['trace_bytes']==0 for row in initial['registrations'])
                    if scenario=='hup':
                        restart=docker('exec',name,'python','-c',f'import os,signal;os.kill({initial["parent_pid"]},signal.SIGHUP)')
                        assert restart.returncode==0
                        deadline=time.monotonic()+30
                        while time.monotonic()<deadline:
                            recovered=wait_pair(name,timeout=5)
                            if not original.intersection(recovered):break
                        else:raise AssertionError('HUP did not replace both workers')
                        final=registrations(name)
                        assert {row['pid'] for row in final['registrations']}==recovered and final['trace_files']==2
                        assert capture(name,output)==[]
                        result['checks'].append('real HUP preserves ready-before-retire policy and cleans normal worker sinks')
                        continue
                    victim = trigger_gil_stall(name,scenario=scenario) if scenario!='heartbeat' else min(original)
                    # No production address/credentials; the victim is a served
                    # PID discovered in our uniquely named disposable container.
                    if scenario=='heartbeat':
                        stopped = docker('exec', name, 'python', '-c',
                                         f'import os,signal;os.kill({victim},signal.SIGSTOP)')
                        assert stopped.returncode == 0
                    recovered = wait_pair(name, original, timeout=30)
                    assert victim not in recovered and original - {victim} <= recovered
                    records = capture(name, output)
                    unhealthy = [r for r in records if r['event'] == 'api_worker_unhealthy']
                    joined = [r for r in records if r['event'] == 'api_worker_failure_joined']
                    assert len(unhealthy) == len(joined) == 1
                    assert unhealthy[0]['worker_pid'] == victim
                    assert unhealthy[0]['reason'] == 'unresponsive_before_replacement'
                    assert unhealthy[0]['exitcode_before_replacement'] is None
                    assert unhealthy[0]['healthcheck_timeout_seconds'] == 5
                    assert joined[0]['exitcode_after_join'] == -9
                    snapshot=unhealthy[0]['failure_snapshot']
                    assert snapshot['capture_ms']<500 and len(snapshot['frames'])<=32
                    assert 'DIAGNOSTIC_PRIVATE_LOCAL_CANARY_7951'not in json.dumps(records)
                    if scenario=='gil-held':
                        assert snapshot['stack_status']=='captured'
                        assert snapshot['diagnostic_signal_sent']
                        assert any(frame['function']=='fullmatch' for frame in snapshot['frames'])
                        assert any(frame['function']=='hold_gil' for frame in snapshot['frames'])
                        assert any(frame['function']in {'pong','always_pong','_healthcheck'} for frame in snapshot['frames'])
                        result['checks'].append('registered real worker yields bounded Python frame locations while native regex retains GIL')
                    elif scenario in {'unregistered','handler-removed'}:
                        expected='unavailable' if scenario=='unregistered' else 'handler_unregistered'
                        assert snapshot['stack_status']==expected and snapshot['frames']==[]
                        assert not snapshot['diagnostic_signal_sent']
                        result['checks'].append('unregistered real worker receives no diagnostic signal and retains upstream replacement')
                    else:
                        assert snapshot['stack_status']=='no_frames_before_deadline'
                        assert any(thread['state']=='T' for thread in snapshot['native']['threads'])
                        result['checks'].append('SIGSTOP records native stopped state and bounded unavailable-stack deadline')
                    final=registrations(name)
                    assert {row['pid'] for row in final['registrations']}==recovered and final['trace_files']==2
                    assert all(row['trace_bytes']==0 for row in final['registrations'])
                    result['checks'].append(scenario+': old raw sink is deleted after join and healthy/replacement sinks stay empty')
                    result['checks'].extend(['heartbeat failure distinguished from a prior process exit',
                                             'existing watchdog kills unresponsive worker and keeps survivor'])
                    shutdown = docker('stop', '--time', '10', name, timeout=15)
                    assert shutdown.returncode == 0
                    assert capture(name, output) == records, 'Normal shutdown misreported as worker failure'
                    result['checks'].append('graceful shutdown does not produce false worker-failure diagnostics')
                else:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        state = docker('inspect', name, '--format', '{{.State.Status}}')
                        if state.stdout.strip() == 'exited':
                            break
                        time.sleep(0.5)
                    else:
                        raise AssertionError('Startup failure left supervisor in an endless restart loop')
                    records = capture(name, output)
                    joined = [r for r in records if r['event'] == 'api_worker_failure_joined']
                    assert len(joined) == 1 and joined[0]['exitcode_before_replacement'] == 3
                    assert joined[0]['reason'] == 'exited_before_replacement'
                    assert joined[0]['exitcode_after_join'] == 3
                    assert joined[0]['failure_snapshot']['stack_status']=='already_exited'
                    result['checks'].append('startup failure records exit code and retains upstream parent-stop policy')
            finally:
                capture(name, output)
                docker('rm', '-f', name)
    result['passed'] = True
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
