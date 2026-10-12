"""Render the real installer safely; Linux validates units with native systemd."""
import json
import os
from pathlib import Path
import subprocess

import pytest


FAKE_SUDO = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, subprocess, sys
root=pathlib.Path(os.environ['UNIT_TEST_ROOT']);a=sys.argv[1:]
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps(a)+'\n')
def target(s):
    if s.startswith(('/etc/','/usr/local/')):return str(root/s.lstrip('/'))
    return s
if a[0]=='tee':
    p=pathlib.Path(target(a[1]));p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(sys.stdin.buffer.read())
elif a[0]=='systemd-analyze':
    if os.getenv('UNIT_TEST_FAIL')=='validator':sys.exit(42)
    paths=[target(s) for s in a[2:]]
    if os.getenv('UNIT_TEST_FAIL')=='invalid_manual':
        p=next(pathlib.Path(s) for s in paths if s.endswith('-manual.service'))
        p.write_text(p.read_text().replace('Type=exec','Type=oneshot'))
    if sys.platform.startswith('linux'):
        validator=shutil.which('systemd-analyze');assert validator, 'Linux CI must provide native systemd unit validation'
        units=root/'etc/systemd/system'
        (units/'docker.service').write_text('[Service]\nType=oneshot\nExecStart=/bin/true\n')
        env={**os.environ,'SYSTEMD_UNIT_PATH':str(units)+':/usr/lib/systemd/system:/lib/systemd/system'}
        sys.exit(subprocess.run([validator,'verify',*paths],env=env).returncode)
    if os.getenv('UNIT_TEST_FAIL')=='invalid_manual':sys.exit(42)
elif a[0]=='systemctl':
    if a[1]=='is-active' and a[-1].endswith('-manual.path') and os.getenv('UNIT_TEST_FAIL')=='inactive_path':sys.exit(3)
elif a[0]=='python3' and 'cutover-ready' in a:
    pass  # Backup engine's data guards are independently covered with real PG.
elif a[0] in {'python3','mkdir','chmod','find','rm','install','test'}:
    b=[target(s) for s in a]
    if a[0]=='install':pathlib.Path(b[-1]).parent.mkdir(parents=True,exist_ok=True)
    sys.exit(subprocess.run(b).returncode)
elif a[0]=='chown':pass
else:raise AssertionError('Unexpected privileged command: '+a[0])
'''


def install(tmp_path, failure=''):
    project=Path(__file__).resolve().parents[1]
    binaries=tmp_path/'bin';binaries.mkdir()
    sudo=binaries/'sudo';sudo.write_text(FAKE_SUDO);sudo.chmod(0o755)
    legacy=tmp_path/'etc/cron.d/companion-db-backup'
    legacy.parent.mkdir(parents=True);legacy.write_text('existing protected backup schedule\n')
    env={**os.environ,'PATH':str(binaries)+os.pathsep+os.environ['PATH'],
         'UNIT_TEST_ROOT':str(tmp_path),'UNIT_TEST_FAIL':failure,
         'ENV_FILE':str(tmp_path/'nonexistent.env'),
         'POSTGRES_BACKUP_PROJECT_DIR':str(project),
         'POSTGRES_BACKUP_DIR':str(tmp_path/'backups'),
         'POSTGRES_BACKUP_SERVICE_NAME':'companion-postgres-backup'}
    result=subprocess.run(['bash',str(project/'scripts/install_postgres_backup_timer.sh')],env=env,text=True,capture_output=True,timeout=30)
    calls=[json.loads(l) for l in (tmp_path/'calls.jsonl').read_text().splitlines()]
    return result,calls,legacy


def test_real_rendered_backup_units_and_activation_guards(tmp_path):
    result,calls,legacy=install(tmp_path)
    assert result.returncode==0,result.stderr
    units=tmp_path/'etc/systemd/system'
    manual=(units/'companion-postgres-backup-manual.service').read_text()
    assert 'Type=exec\n' in manual and 'RestartForceExitStatus=75\n' in manual
    assert 'UMask=0077\n' in manual and 'User=root\n' in manual
    assert 'Type=oneshot\n' in (units/'companion-postgres-backup.service').read_text()
    verify=next(i for i,c in enumerate(calls) if c[:2]==['systemd-analyze','verify'])
    reload=next(i for i,c in enumerate(calls) if c==['systemctl','daemon-reload'])
    enable=next(i for i,c in enumerate(calls) if c[:3]==['systemctl','enable','--now'])
    active=next(i for i,c in enumerate(calls) if c==['systemctl','is-active','--quiet','companion-postgres-backup-manual.path'])
    retired=next(i for i,c in enumerate(calls) if c[:3]==['rm','-f','/etc/cron.d/companion-db-backup'])
    assert verify<reload<enable<active<retired
    assert not legacy.exists()
    assert json.loads((tmp_path/'backups/.scheduler.json').read_text())['executor']=='host_systemd'


@pytest.mark.parametrize('failure',['validator','invalid_manual','inactive_path'])
def test_bad_unit_or_inactive_watcher_keeps_legacy_schedule(tmp_path,failure):
    result,calls,legacy=install(tmp_path,failure)
    assert result.returncode!=0
    assert legacy.read_text()=='existing protected backup schedule\n'
    assert not (tmp_path/'backups/.scheduler.json').exists()
    assert not any(c[:2]==['rm','-f'] for c in calls)
    if failure!='inactive_path':
        assert ['systemctl','daemon-reload'] not in calls
        assert not any(c[:2]==['systemctl','enable'] for c in calls)
