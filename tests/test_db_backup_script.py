"""Backup commit/retention and subprocess failure tests, no production access."""
from datetime import timedelta
import os
from pathlib import Path
import subprocess

import pytest
from scripts import postgres_backup as engine
from tests.test_db_backup_api import complete


@pytest.fixture
def fake_commands(monkeypatch):
    calls=[]
    def run(command,*,env,output=None,stdin=None):
        calls.append(command)
        if output:
            output.write(b'full database' if 'pg_dump' in command else b'role data')
            return b''
        return b'TABLE DATA\n'*30
    monkeypatch.setattr(engine,'run_command',run)
    return calls


def test_commit_private_files_and_expire_only_old_complete_sets(tmp_path,fake_commands):
    old=complete(tmp_path,1,days=20);recent=complete(tmp_path,2,days=1)
    partial=tmp_path/'.unknown.tmp';partial.write_bytes(b'unknown')
    name=engine.create(tmp_path,'tcp',{},'manual')
    assert len(fake_commands)==3
    rows,_,_=engine.inventory(tmp_path)
    assert {r['name'] for r in rows}=={name,recent}
    assert not (tmp_path/old).exists() and partial.exists()
    for p in engine.paths(tmp_path,name):assert p.stat().st_mode&0o777==0o600
    assert engine.read_json(tmp_path/'.backup-status.json')['status']=='success'
    assert engine.snapshot(tmp_path)['backups'][0]['restore']['status']=='not_run'


@pytest.mark.parametrize('stage',[1,2,3])
def test_failure_cleans_only_own_partials_and_preserves_recovery_points(tmp_path,monkeypatch,stage):
    kept=complete(tmp_path,1,days=20);counter=0
    def fail(command,*,env,output=None,stdin=None):
        nonlocal counter
        counter+=1
        if output:output.write(b'partial')
        if counter==stage:raise engine.BackupError('受控错误')
        return b'TABLE DATA\n'*30
    monkeypatch.setattr(engine,'run_command',fail)
    with pytest.raises(engine.BackupError):engine.create(tmp_path,'docker',{},'scheduled')
    assert (tmp_path/kept).exists()
    assert not list(tmp_path.glob('*.tmp'))
    assert len(engine.inventory(tmp_path)[0])==1
    assert engine.read_json(tmp_path/'.backup-status.json')['status']=='failed'


def test_reserve_failure_is_visible_and_never_prunes(tmp_path,monkeypatch):
    name=complete(tmp_path,1,days=30)
    monkeypatch.setattr(engine.shutil,'disk_usage',lambda _:type('Usage',(),{'free':1})())
    with pytest.raises(engine.BackupError,match='空间'):engine.create(tmp_path,'tcp',{},'manual')
    assert (tmp_path/name).exists() and engine.read_json(tmp_path/'.backup-status.json')['status']=='failed'


def test_toc_failure_no_commit(tmp_path,monkeypatch,fake_commands):
    monkeypatch.setattr(engine,'run_command',lambda command,**kw:kw['output'].write(b'data') if kw.get('output') else b'bad toc')
    with pytest.raises(engine.BackupError,match='少于20'):engine.create(tmp_path,'tcp',{},'manual')
    assert not engine.inventory(tmp_path)[0]


def test_verify_previous_manifest_exact_hashes_and_no_history_rewrite(tmp_path,fake_commands):
    name='companion_20261012T000000Z.dump';dump,glob,meta=engine.paths(tmp_path,name)
    dump.write_bytes(b'data');glob.write_bytes(b'roles')
    manifest=tmp_path/(name[:-5]+'.manifest')
    manifest.write_text(f'{engine.digest(dump)}  /old/root/{dump.name}\n{engine.digest(glob)}  /old/root/{glob.name}\ncreated_at=old\n')
    before=manifest.read_bytes()
    data=engine.verify(tmp_path,name,'docker',{})
    assert data['source']=='legacy_systemd' and manifest.read_bytes()==before
    dump.write_bytes(b'changed')
    with pytest.raises(engine.BackupError,match='不一致'):engine.verify(tmp_path,name,'docker',{})


def test_verify_refuses_in_root_symlink(tmp_path,fake_commands):
    name='companion_20261012T000000Z.dump';dump,glob,meta=engine.paths(tmp_path,name)
    secret=tmp_path/'secret';secret.write_bytes(b'secret');dump.symlink_to(secret);glob.write_bytes(b'role')
    with pytest.raises(engine.BackupError):engine.verify(tmp_path,name,'docker',{})


def test_timeout_and_secret_stderr_never_surface(monkeypatch):
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('cmd',300,stderr='SECRET')
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(engine.BackupError,match='超时') as e:engine.run_command(['pg_dump'],env={})
    assert 'SECRET' not in str(e.value)
    def failed(*args,**kwargs):raise subprocess.CalledProcessError(1,'cmd',stderr='SECRET')
    monkeypatch.setattr(subprocess,'run',failed)
    with pytest.raises(engine.BackupError) as e:engine.run_command(['pg_dump'],env={})
    assert 'SECRET' not in str(e.value)


def test_installer_one_timer_and_retirement_guard():
    root=Path(__file__).resolve().parents[1]
    installer=(root/'scripts/install_postgres_backup_timer.sh').read_text()
    maintenance=(root/'scripts/deploy_host_maintenance.sh').read_text()
    assert 'cutover-ready' in installer and installer.index('cutover-ready')<installer.index('sudo rm -f /etc/cron.d/companion-db-backup')
    assert 'UMask=0077' in installer and 'RUN_USER="root"' in installer
    assert 'install_postgres_backup_timer.sh' in maintenance and "<<'BACKUP'" not in maintenance
    compose=(root/'docker-compose.deploy.yml').read_text()
    assert '/backups/postgres:/var/backups/companion' in compose
    assert '0 3 * * * root /usr/local/bin/companion-db-backup.sh' not in maintenance


def test_catalog_entry_bound_and_incomplete_marker(tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'MAX_ENTRIES',3)
    for i in range(4):(tmp_path/str(i)).touch()
    with pytest.raises(engine.BackupError,match='超过1000'):engine.snapshot(tmp_path)


def test_manual_request_consumed_once_and_same_host_format(tmp_path,fake_commands):
    engine.atomic_json(tmp_path/'.scheduler.json',{'schema_version':1})
    requested=engine.request_backup(tmp_path)
    result=engine.process_request(tmp_path,'docker',{})
    assert result['name'].endswith('.dump')
    assert engine.process_request(tmp_path,'docker',{})=={'status':'no_request'}
    assert engine.snapshot(tmp_path)['backups'][0]['source']=='manual'
    assert fake_commands[0][:5]==['docker','exec','-i','companion-postgres','pg_dump']


def test_manual_waits_under_the_same_backup_lock(tmp_path):
    engine.atomic_json(tmp_path/'.scheduler.json',{'schema_version':1})
    engine.request_backup(tmp_path)
    with engine.locked(tmp_path):
        with pytest.raises(engine.BackupError):engine.process_request(tmp_path,'docker',{})
    assert (tmp_path/'.manual-request.json').exists()


def test_queue_expiry_is_warning_not_success(tmp_path,monkeypatch):
    engine.atomic_json(tmp_path/'.backup-status.json',{'status':'queued','started_at':(engine.now()-timedelta(minutes=6)).isoformat()})
    assert any('超过5分钟' in w for w in engine.snapshot(tmp_path)['summary']['warnings'])


def test_same_size_corruption_of_retained_copy_blocks_delete(tmp_path):
    names=[complete(tmp_path,i,days=20) for i in range(1,4)]
    report=engine.snapshot(tmp_path)
    removable=next(row['name'] for row in report['backups'] if row['can_delete'])
    kept=next(row['name'] for row in report['backups'] if not row['can_delete'])
    dump=tmp_path/kept;dump.write_bytes(b'X'*dump.stat().st_size)
    with pytest.raises(engine.BackupError,match='校验和'):engine.remove(tmp_path,removable)
    assert (tmp_path/removable).exists()


def test_invalid_request_does_not_spin_forever(tmp_path):
    engine.atomic_json(tmp_path/'.manual-request.json',{'schema_version':1,'request_id':'invalid'})
    with pytest.raises(engine.BackupError):engine.process_request(tmp_path,'docker',{})
    assert not (tmp_path/'.manual-request.json').exists()
    assert engine.read_json(tmp_path/'.backup-status.json')['status']=='failed'
