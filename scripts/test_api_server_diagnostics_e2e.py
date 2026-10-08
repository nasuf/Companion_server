"""Faults only in owned, network-isolated containers using the image's real CMD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import uuid

from test_server_workers_e2e import docker, wait_pair

FIXTURE = '''import json,os
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
  await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
  await send({'type':'http.response.body','body':json.dumps({'pid':os.getpid()}).encode()})
'''


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
        for scenario in ('heartbeat', 'startup-failure'):
            name = 'companion-worker-diagnostic-' + uuid.uuid4().hex[:12]
            output = args.output / scenario
            output.mkdir(exist_ok=True)
            try:
                launch(args.image, name, folder, fail_startup=scenario == 'startup-failure')
                if scenario == 'heartbeat':
                    original = wait_pair(name, timeout=30)
                    victim = min(original)
                    # No production address/credentials; the victim is a served
                    # PID discovered in our uniquely named disposable container.
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
                    result['checks'].append('startup failure records exit code and retains upstream parent-stop policy')
            finally:
                capture(name, output)
                docker('rm', '-f', name)
    result['passed'] = True
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
