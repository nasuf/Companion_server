"""Real files and HTTP security, failure states and bounded admin contract."""
import asyncio
from datetime import timedelta
import os
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.ops import db_backup
from scripts import postgres_backup as engine


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(db_backup, "BACKUP_DIR", tmp_path)
    engine.atomic_json(tmp_path/".scheduler.json", {"schema_version": 1})
    return tmp_path


def complete(root, i=1, days=0, source="scheduled"):
    name = f"companion_20261012T0000{i:02d}Z_{i:08x}.dump"
    dump, glob, meta = engine.paths(root, name)
    dump.write_bytes(b"snapshot" + bytes([i % 255]))
    glob.write_bytes(b"role snapshot")
    engine.atomic_json(meta, {"schema_version": 1, "name": name, "source": source, "checked_at": engine.stamp(),
        "validation": "checksum_and_toc", "table_count": 30, "dump_bytes": dump.stat().st_size,
        "globals_bytes": glob.stat().st_size, "dump_sha256": engine.digest(dump), "globals_sha256": engine.digest(glob)})
    timestamp = (engine.now()-timedelta(days=days)).timestamp()
    os.utime(dump, (timestamp, timestamp))
    return name


@pytest.mark.parametrize("name", ["../foo", "/etc/passwd", "..%2fetc", "", "companion_.dump", "companion_20261012T000000Z.dump/../foo"])
def test_path_rejection(root, name):
    with pytest.raises(ValueError): db_backup.delete_backup(name)


def test_symlinks_never_followed(root):
    name = complete(root)
    dump = root/name
    dump.unlink(); dump.symlink_to(root / "secret")
    (root/"secret").write_text("private")
    with pytest.raises(ValueError): db_backup.delete_backup(name)
    report = db_backup.list_report()
    assert report["available"] is False and report["total"] is None
    assert (root/"secret").read_text() == "private"


def test_recent_and_last_two_protected(root):
    for i in range(1,4): complete(root,i,days=20)
    recent = complete(root,4)
    rows = engine.snapshot(root)["backups"]
    newest_old = next(r for r in rows if r["name"] != recent)
    for name in (recent, newest_old["name"]):
        with pytest.raises(ValueError): db_backup.delete_backup(name)
    removable = next(r for r in rows if r["can_delete"])
    db_backup.delete_backup(removable["name"])
    assert not any(p.name.startswith(removable["name"][:-5]) for p in root.iterdir())


def test_failed_candidate_does_not_count_as_valid_copy(root):
    name=complete(root,1,days=20); other=complete(root,2,days=20)
    (root/other).write_bytes(b"wrong size")
    with pytest.raises(ValueError): db_backup.delete_backup(name)
    rows=engine.snapshot(root)["backups"]
    assert next(r for r in rows if r["name"]==other)["integrity"]=="invalid"


def test_missing_partial_legacy_and_pagination_are_distinct(root):
    for i in range(1,26): complete(root,i)
    (root/".unfinished.tmp").write_bytes(b"x")
    legacy = root / "companion_20260901T000000Z.dump"
    legacy.write_bytes(b"old")
    # Filename age is not the sort key. Pin mtime so the legacy dump stays last
    # even when writing the validated copies takes more than one second.
    old = (engine.now() - timedelta(days=40)).timestamp()
    os.utime(legacy, (old, old))
    (root/"companion_20261012T230000Z_aabbccdd.dump").write_bytes(b"incomplete")
    report=db_backup.list_report(limit=20)
    assert report["total"]==26 and len(report["backups"])==20 and report["has_more"]
    page2=db_backup.list_report(offset=20)
    assert len(page2["backups"])==6 and not page2["has_more"]
    assert page2["backups"][-1]["name"] == legacy.name
    assert page2["backups"][-1]["integrity"]=="not_checked"  # legacy is not silently qualified
    assert len(report["summary"]["warnings"])>=2
    assert "not_configured"==report["summary"]["offsite"]


def test_missing_directory_and_binary(root,monkeypatch):
    monkeypatch.setattr(db_backup,"BACKUP_DIR",root/"missing")
    r=db_backup.list_report();assert not r["available"] and r["summary"] is None
    assert any("执行器" in reason for reason in r["reasons"])


def test_metadata_bound_and_corruption(root):
    complete(root)
    (root/".backup-status.json").write_text("x"*65537)
    assert db_backup.list_report()["summary"] is None
    (root/".backup-status.json").write_text("[]")
    assert db_backup.list_report()["total"] is None


def test_real_cross_process_lock_and_crash_state(root):
    with engine.locked(root):
        code='from pathlib import Path;from scripts.postgres_backup import locked;\nwith locked(Path(__import__("sys").argv[1])):pass'
        result=subprocess.run([sys.executable,"-c",code,str(root)],capture_output=True,text=True)
        assert result.returncode!=0 and "已有备份" in result.stderr
        assert engine.snapshot(root)["summary"]["last_run"]["status"]=="running"
    engine.atomic_json(root/".backup-status.json",{"status":"running"})
    assert engine.snapshot(root)["summary"]["last_run"]["status"]=="interrupted"


def test_budget_warns_without_deleting_and_missing_days(root, monkeypatch):
    complete(root)
    monkeypatch.setattr(engine,"BUDGET_BYTES",1)
    r=engine.snapshot(root)
    assert len(r["backups"])==1 and any("预算" in w for w in r["summary"]["warnings"])
    assert r["summary"]["covered_days"]==0 and r["summary"]["coverage_days"]==14


def test_restore_proof_bound_to_exact_hash_and_cutover(root):
    name=complete(root);complete(root,2)
    with pytest.raises(engine.BackupError):engine.cutover_ready(root)
    path=root/(name[:-5]+".restore.json")
    engine.atomic_json(path,{"status":"passed","checked_at":engine.stamp(),"dump_sha256":"0"*64})
    row=next(r for r in engine.snapshot(root)["backups"] if r["name"]==name)
    assert row["restore"]["status"]=="invalid"
    engine.atomic_json(path,{"status":"passed","checked_at":engine.stamp(),"dump_sha256":engine.digest(root/name)})
    assert engine.cutover_ready(root)["ready"]


@pytest.fixture
def client(root):
    from app.api.admin.stats import router
    from app.api.jwt_auth import require_admin_jwt
    app=FastAPI();app.include_router(router)
    # JWT boundary still exercised in unauthenticated cases by removing override.
    app.dependency_overrides[require_admin_jwt]=lambda:{"role":"admin"}
    return TestClient(app)


def test_http_pagination_delete_and_permission(root,client):
    for i in range(1,24):complete(root,i)
    assert len(client.get('/admin-api/stats/db-backups?limit=10').json()['backups'])==10
    for query in ('limit=51','offset=-1','offset=1001'):
        assert client.get('/admin-api/stats/db-backups?'+query).status_code==422
    assert client.delete('/admin-api/stats/db-backups/'+complete(root,24)).status_code==400
    assert client.delete('/admin-api/stats/db-backups/companion_20261012T111111Z.dump').status_code==404
    client.app.dependency_overrides.clear()
    for method in ('get','post','delete'):
        url='/admin-api/stats/db-backups'+('/anything' if method=='delete' else '')
        assert getattr(client,method)(url).status_code in (401,403)


def test_manual_http_success_and_safe_failures(root,client,monkeypatch):
    fake=AsyncMock(return_value={'accepted':True,'request_id':'synthetic'})
    monkeypatch.setattr(db_backup,'create_backup',fake)
    assert client.post('/admin-api/stats/db-backups').json()=={'accepted':True,'request_id':'synthetic'}
    fake.side_effect=engine.BackupError('已有备份操作正在进行')
    assert client.post('/admin-api/stats/db-backups').status_code==409
    fake.side_effect=asyncio.TimeoutError()
    assert client.post('/admin-api/stats/db-backups').status_code==504


@pytest.mark.asyncio
async def test_manual_adapter_queues_without_database_credentials(root,monkeypatch):
    result=await db_backup.create_backup()
    assert result['accepted'] and len(result['request_id'])==32
    assert engine.read_json(root/'.manual-request.json')['request_id']==result['request_id']
    assert engine.snapshot(root)['summary']['last_run']['status']=='queued'
    with pytest.raises(engine.BackupError,match='等待执行'):await db_backup.create_backup()
    monkeypatch.setattr(db_backup,'BACKUP_DIR',root/'missing')
    with pytest.raises(RuntimeError):await db_backup.create_backup()


def test_unreadable_directory_and_unsupported_executor_are_unavailable(root, monkeypatch):
    complete(root)
    monkeypatch.setattr(db_backup.os, 'access', lambda *_: False)
    report = db_backup.list_report()
    assert not report['available'] and '备份目录无读写权限' in report['reasons']
    engine.atomic_json(root / '.scheduler.json', {'schema_version': 2})
    report = db_backup.list_report()
    assert not report['available'] and '宿主机备份执行器未配置' in report['reasons']
    assert len(report['backups']) == 1  # Availability failure never deletes recovery points.


def test_directory_stat_error_is_explicit(root, monkeypatch):
    def denied(_):
        raise PermissionError('private diagnostic must not escape')
    monkeypatch.setattr(type(root), 'is_dir', denied)
    state = db_backup.availability()
    assert not state['available'] and '备份目录不可访问' in state['reasons']
    assert 'private diagnostic' not in str(state)


def test_delete_io_failure_is_controlled_and_preserves_files(root, client, monkeypatch):
    name = complete(root)
    def denied(*_):
        raise PermissionError('private diagnostic must not escape')
    monkeypatch.setattr(engine, 'remove', denied)
    response = client.delete('/admin-api/stats/db-backups/' + name)
    assert response.status_code == 400 and 'private diagnostic' not in response.text
    assert (root / name).is_file()


@pytest.mark.asyncio
async def test_manual_request_write_failure_is_controlled(root, monkeypatch):
    name = complete(root)
    def denied(*_):
        raise PermissionError('private diagnostic must not escape')
    monkeypatch.setattr(engine, 'atomic_json', denied)
    with pytest.raises(RuntimeError, match='备份请求写入失败'):
        await db_backup.create_backup()
    assert not (root / '.manual-request.json').exists()
    assert (root / name).is_file()
