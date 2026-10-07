"""Deployment gate: initialized workers must stay alive, with no startup restarts.

Read-only host probe. Never prints configuration, logs or environment secrets.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time

PROCESS_PROBE = '''from pathlib import Path
import json
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try: args=(p/'cmdline').read_bytes().split(b'\\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if b'-c' in args and args[args.index(b'-c')+1].startswith(b'from multiprocessing.spawn import spawn_main') and b'--multiprocessing-fork' in args:
  pids.append(int(p.name))
print(json.dumps(sorted(pids)))'''


def snapshot(command, container, expected):
    def run(*args):
        r=subprocess.run([*command,*args],capture_output=True,text=True,timeout=15)
        if r.returncode:
            raise RuntimeError('container probe unavailable')
        return r
    inspection=json.loads(run('inspect',container).stdout)[0]
    state=inspection['State']
    active=set(json.loads(run('exec',container,'python','-c',PROCESS_PROBE).stdout))
    logs=run('logs','--since',state['StartedAt'],container)
    lines=(logs.stdout+'\n'+logs.stderr).splitlines()
    initialized={int(m.group(1)) for line in lines if (m:=re.search(r'Started server process \[(\d+)\]',line))}
    completed=sum('Application startup complete' in line for line in lines)
    deaths=sum('Child process' in line and 'died' in line for line in lines)
    failure=sum('Application startup failed' in line for line in lines)
    ready=(state['Status']=='running' and state.get('Health',{}).get('Status','healthy')=='healthy'
           and not state['OOMKilled'] and len(active)==expected
           and active<=initialized and completed>=expected and deaths==0 and failure==0)
    return {'ready':ready,'image':inspection['Image'],'started_at':state['StartedAt'],
            'active_pids':sorted(active),'initialized_pids':sorted(initialized),
            'startup_completions':completed,'child_deaths':deaths,'startup_failures':failure}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--docker',default='docker')
    parser.add_argument('--container',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--stable-seconds',type=float,default=15)
    args=parser.parse_args()
    if args.workers<2 or args.timeout<=0 or args.stable_seconds<0:
        parser.error('multiworker gate requires workers >= 2 and valid deadlines')
    deadline=time.monotonic()+args.timeout
    stable=None
    identity=None
    last={'ready':False,'reason':'not observed'}
    while time.monotonic()<deadline:
        try:
            last=snapshot(shlex.split(args.docker),args.container,args.workers)
        except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError):
            last={'ready':False,'reason':'probe unavailable'}
        if last.get('child_deaths',0) or last.get('startup_failures',0):
            break
        current=(last.get('image'),last.get('started_at'),tuple(last.get('active_pids',[])))
        if last['ready']:
            if current!=identity:
                stable=time.monotonic()
                identity=current
            if time.monotonic()-stable>=args.stable_seconds:
                print(json.dumps({'ok':True,**last,'stable_seconds':args.stable_seconds}))
                return
        else:
            stable=None
            identity=None
        time.sleep(2)
    print(json.dumps({'ok':False,**last}))
    raise SystemExit(1)


if __name__=='__main__':
    main()
