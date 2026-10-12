#!/usr/bin/env python3
"""One local backup format for host scheduling and admin manual operations.

Stdlib only: the host scheduler must work while the application is unavailable.
A JSON manifest is the commit marker; no credentials or database content enter it.
Reading the catalog checks metadata, not the entire dump. Integrity verification
and an actual isolated restore are separate, explicitly dated operations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import uuid

NAME = re.compile(r"^companion_(\d{8}T\d{6}Z)(?:_[0-9a-f]{8})?\.dump$")
MAX_ENTRIES = 1000
MAX_METADATA = 65536
RETENTION_DAYS = 14
BUDGET_BYTES = 8 * 1024**3
RESERVE_BYTES = 1024**3
MIN_COPIES = 2


class BackupError(RuntimeError):
    """Safe operational error; subprocess output must never reach callers."""


def now() -> datetime:
    return datetime.now(timezone.utc)


def stamp() -> str:
    return now().isoformat(timespec="seconds")


def regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise BackupError("备份路径包含非普通文件")
    return info


def read_json(path: Path) -> dict:
    if regular(path).st_size > MAX_METADATA:
        raise BackupError("备份元数据超过读取上限")
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, UnicodeError) as exc:
        raise BackupError("备份元数据损坏") from exc


def atomic_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        regular(path)
    fd, temp = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, ensure_ascii=False, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally:
        Path(temp).unlink(missing_ok=True)


def sync_dir(root: Path) -> None:
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(root: Path):
    # No O_TRUNC: concurrent workers and the host must lock the same inode.
    fd = os.open(root / ".backup.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BackupError("备份锁不是普通文件")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError("已有备份操作正在进行，请稍候") from exc
        yield
    finally:
        os.close(fd)


def digest(path: Path) -> str:
    regular(path)
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(root: Path, name: str) -> tuple[Path, Path, Path]:
    if not NAME.fullmatch(name):
        raise ValueError("非法的备份文件名")
    base = name[:-5]
    return root / name, root / (base + ".globals.sql"), root / (base + ".backup.json")


def inventory(root: Path) -> tuple[list[dict], int, list[str]]:
    """Bounded metadata inventory; unknown/partial files consume real capacity."""
    entries = []
    with os.scandir(root) as it:
        for entry in it:
            entries.append(entry)
            if len(entries) > MAX_ENTRIES:
                raise BackupError("备份目录超过1000项，需运维检查")
    total = 0
    warnings: list[str] = []
    rows = []
    for entry in entries:
        try:
            info = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue  # A concurrently committed/pruned temporary file is no longer in this sample.
        if not stat.S_ISREG(info.st_mode):
            raise BackupError("备份目录包含非普通文件，需运维检查")
        total += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
        if not NAME.fullmatch(entry.name):
            if entry.name.endswith(".tmp"):
                warnings.append("发现未完成备份文件；未计为可用恢复点")
            continue
        dump, globals_file, manifest = paths(root, entry.name)
        data = None
        error = None
        try:
            if manifest.exists():
                data = read_json(manifest)
                if data.get("schema_version") != 1 or data.get("name") != entry.name:
                    raise BackupError("清单版本或备份名不一致")
                for p, field in ((dump, "dump_bytes"), (globals_file, "globals_bytes")):
                    if regular(p).st_size != data[field] or data[field] <= 0:
                        raise BackupError("备份文件大小与清单不一致")
                for field in ("dump_sha256", "globals_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", data[field]):
                        raise BackupError("校验和格式无效")
                if data.get("validation") != "checksum_and_toc" or data.get("table_count", 0) < 20:
                    raise BackupError("缺少完整性校验记录")
            elif "_" in entry.name[:-5] and len(entry.name[:-5].split("_")) > 2:
                # New unique format MUST have its completion marker.
                warnings.append("发现缺少完成清单的新备份；未计为可用恢复点")
                continue
        except (OSError, BackupError, KeyError, TypeError) as exc:
            error = "备份清单或文件不完整，需重新校验"
            warnings.append(error)
        created = datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(timespec="seconds")
        restore = {"status": "not_run", "checked_at": None}
        if data and not error:
            proof = root / (entry.name[:-5] + ".restore.json")
            if proof.exists():
                try:
                    candidate = read_json(proof)
                    if candidate.get("dump_sha256") == data["dump_sha256"] and candidate.get("status") in {"passed", "failed"}:
                        restore = {"status": candidate["status"], "checked_at": candidate.get("checked_at")}
                    else:
                        restore = {"status": "invalid", "checked_at": None}
                except (OSError, BackupError):
                    restore = {"status": "invalid", "checked_at": None}
        rows.append({"name": entry.name, "size_bytes": info.st_size, "created_at": created,
                     "source": data.get("source", "legacy") if data else "legacy",
                     "integrity": "invalid" if error else "recorded" if data else "not_checked",
                     "integrity_checked_at": data.get("checked_at") if data and not error else None,
                     "restore": restore, "error": error})
    rows.sort(key=lambda row: (row["created_at"], row["name"]), reverse=True)
    return rows, total, sorted(set(warnings))


def protected(rows: list[dict], at: datetime) -> set[str]:
    valid = [row for row in rows if row["integrity"] == "recorded"]
    cutoff = (at - timedelta(days=RETENTION_DAYS)).isoformat()
    return {r["name"] for r in valid[:MIN_COPIES]} | {r["name"] for r in rows if r["created_at"] >= cutoff}


def snapshot(root: Path, offset: int = 0, limit: int = 20) -> dict:
    rows, total, warnings = inventory(root)
    at = now()
    safe = protected(rows, at)
    for row in rows:
        row["can_delete"] = row["name"] not in safe and row["integrity"] == "recorded"
    state_path = root / ".backup-status.json"
    state = read_json(state_path) if state_path.exists() else {"status": "not_collected"}
    if state.get("status") not in {"not_collected", "queued", "running", "success", "failed"}:
        raise BackupError("备份运行状态格式无效")
    # Read actual flock state: a crash cannot leave a permanently green 'running'.
    try:
        with locked(root):
            if state.get("status") == "running":
                state = {**state, "status": "interrupted", "error": "进程已退出且未记录完成"}
    except BackupError:
        state = {**state, "status": "running"}
    valid = [r for r in rows if r["integrity"] == "recorded"]
    newest = valid[0]["created_at"] if valid else None
    if state.get("status") == "queued":
        queued_at = datetime.fromisoformat(state["started_at"])
        if at - queued_at > timedelta(minutes=5):
            warnings.append("手动请求等待超过5分钟，需检查宿主机执行器")
    if not newest:
        warnings.append("尚无带完整性校验记录的备份")
    elif at - datetime.fromisoformat(newest) > timedelta(hours=36):
        warnings.append("超过36小时没有新的已校验备份")
    if total > BUDGET_BYTES:
        warnings.append("本地备份超过8GiB预算；不会缩短14天保留窗口")
    days = {datetime.fromisoformat(r["created_at"]).astimezone(timezone.utc).date() for r in valid}
    # Last 14 completed UTC days. Today's pending run is not a missed day.
    completed = [(at - timedelta(days=i)).date() for i in range(1, RETENTION_DAYS + 1)]
    return {"as_of": at.isoformat(), "backups": rows[offset:offset+limit], "total": len(rows),
            "offset": offset, "limit": limit, "has_more": offset+limit < len(rows),
            "summary": {"used_bytes": total, "budget_bytes": BUDGET_BYTES, "free_bytes": shutil.disk_usage(root).free,
                        "retention_days": RETENTION_DAYS, "min_copies": MIN_COPIES, "last_success_at": newest,
                        "covered_days": sum(d in days for d in completed), "coverage_days": RETENTION_DAYS,
                        "coverage_timezone": "UTC", "warnings": sorted(set(warnings)), "last_run": state,
                        "offsite": "not_configured", "schedule": "每日03:20–03:30 Asia/Shanghai（宿主机systemd；本页不证明timer活跃）"}}


def remove(root: Path, name: str) -> None:
    dump, globals_file, meta = paths(root, name)
    with locked(root):
        regular(dump)
        rows, _, _ = inventory(root)
        row = next((r for r in rows if r["name"] == name), None)
        if not row or row["integrity"] != "recorded" or name in protected(rows, now()):
            raise ValueError("保留期内或最后两份已校验备份不能删除；未知备份需运维核验")
        verify_retained(root, rows, {name})
        delete_set(root, name)


def verify_retained(root: Path, rows: list[dict], excluding: set[str]) -> None:
    kept = [r for r in rows if r["name"] not in excluding and r["integrity"] == "recorded"][:MIN_COPIES]
    if len(kept) < MIN_COPIES:
        raise BackupError("不足两份完整备份，拒绝清理")
    for row in kept:
        dump, glob, meta = paths(root, row["name"])
        data = read_json(meta)
        if digest(dump) != data["dump_sha256"] or digest(glob) != data["globals_sha256"]:
            raise BackupError("保留备份校验和不一致，拒绝清理")


def delete_set(root: Path, name: str) -> None:
    dump, globals_file, meta = paths(root, name)
    related = [dump, globals_file, meta, root / (name[:-5] + ".manifest"), root / (name[:-5] + ".restore.json")]
    # Validate all before deleting any; remove completion marker first.
    for p in related:
        if p.exists() or p.is_symlink():
            regular(p)
    meta.unlink(missing_ok=True)
    sync_dir(root)
    for p in related:
        if p != meta:
            p.unlink(missing_ok=True)


def run_command(command: list[str], *, env: dict, output=None, stdin=None) -> bytes:
    try:
        result = subprocess.run(command, env=env, stdout=output or subprocess.PIPE,
                                stderr=subprocess.DEVNULL, stdin=stdin, timeout=300, check=True)
        return result.stdout or b""
    except subprocess.TimeoutExpired as exc:
        raise BackupError("数据库备份命令超时，子进程已回收") from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise BackupError("数据库备份命令失败，请检查主机日志及连接配置") from exc


def commands(transport: str, env: dict):
    if transport == "docker":
        prefix = ["docker", "exec", "-i", env.get("POSTGRES_CONTAINER_NAME", "companion-postgres")]
        connection = ["-U", env.get("POSTGRES_USER", "companion")]
    else:
        prefix = []
        connection = ["-h", env.get("PGHOST", "postgres"), "-p", env.get("PGPORT", "5432"), "-U", env.get("PGUSER", "companion")]
    database = env.get("PGDATABASE" if transport == "tcp" else "POSTGRES_DB", "companion")
    return (prefix + ["pg_dump", *connection, "-d", database, "-Fc", "--compress=6", "--no-owner", "--no-acl", "--lock-wait-timeout=10s"],
            prefix + ["pg_dumpall", *connection, "--globals-only"], prefix + ["pg_restore", "--list"])


def verify(root: Path, name: str, transport: str, env: dict, source="legacy_systemd") -> dict:
    """Also qualifies the previous checksum manifest format without rewriting it."""
    dump, globals_file, meta = paths(root, name)
    _, _, toc_cmd = commands(transport, env)
    regular(dump)
    regular(globals_file)
    with dump.open("rb") as src:
        table_count = run_command(toc_cmd, env=env, stdin=src).count(b"TABLE DATA")
    if table_count < 20 or regular(globals_file).st_size <= 0:
        raise BackupError("归档目录或角色备份不完整")
    hashes = {"dump_sha256": digest(dump), "globals_sha256": digest(globals_file)}
    if meta.exists():
        old = read_json(meta)
        if any(old.get(k) != v for k, v in hashes.items()):
            raise BackupError("备份校验和不一致")
        source = old.get("source", source)
    else:
        # Every old enhanced backup must have BOTH exact checksum entries.
        legacy = root / (name[:-5] + ".manifest")
        if regular(legacy).st_size > MAX_METADATA:
            raise BackupError("历史清单超过上限")
        lines = legacy.read_text().splitlines()
        for p, field in ((dump, "dump_sha256"), (globals_file, "globals_sha256")):
            if not any(line.startswith(hashes[field] + "  ") and Path(line[66:]).name == p.name for line in lines):
                raise BackupError("历史备份校验和不一致或缺失")
    data = {"schema_version": 1, "name": name, "source": source, "checked_at": stamp(),
            "validation": "checksum_and_toc", "table_count": table_count,
            "dump_bytes": regular(dump).st_size, "globals_bytes": regular(globals_file).st_size, **hashes}
    atomic_json(meta, data)
    return data


def cutover_ready(root: Path) -> dict:
    report = snapshot(root)
    rows, _, _ = inventory(root)
    recent = now() - timedelta(days=2)
    qualified = [r for r in rows if r["integrity"] == "recorded"]
    if len(qualified) < MIN_COPIES or not any(
        r["restore"]["status"] == "passed" and datetime.fromisoformat(r["created_at"]) >= recent
        for r in qualified
    ):
        raise BackupError("需至少两份已校验备份及两天内快照的隔离恢复回执，保留旧任务")
    verify_retained(root, qualified, set())
    return {"ready": True, "qualified_copies": len(qualified), "covered_days": report["summary"]["covered_days"]}


def request_backup(root: Path) -> dict:
    with locked(root):
        marker = read_json(root / ".scheduler.json")
        if marker.get("schema_version") != 1:
            raise BackupError("宿主机备份执行器未配置")
        request = root / ".manual-request.json"
        if request.exists():
            raise BackupError("已有手动备份等待执行，请稍候")
        data = {"schema_version": 1, "request_id": uuid.uuid4().hex, "requested_at": stamp()}
        atomic_json(root / ".backup-status.json", {"status": "queued", "started_at": data["requested_at"], "source": "manual"})
        atomic_json(request, data)
        return {"accepted": True, **data}


def process_request(root: Path, transport: str, env: dict) -> dict:
    with locked(root):
        request = root / ".manual-request.json"
        if not request.exists():
            return {"status": "no_request"}
        try:
            data = read_json(request)
            if data.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", data.get("request_id", "")):
                raise BackupError("手动备份请求格式无效")
            request.unlink()
            return {"name": create(root, transport, env, "manual", lock_held=True)}
        except (BackupError, OSError, ValueError, TypeError):
            if request.is_file() or request.is_symlink():
                request.unlink()
            atomic_json(root / ".backup-status.json", {"status": "failed", "finished_at": stamp(),
                        "source": "manual", "error": "手动备份执行失败，需检查主机日志"})
            raise


def create(root: Path, transport: str, env: dict, source: str, *, lock_held=False) -> str:
    with nullcontext() if lock_held else locked(root):
        rows, _, _ = inventory(root)  # Fail closed before mutation if catalog is abnormal.
        required = max(RESERVE_BYTES, 2 * max((r["size_bytes"] for r in rows), default=0))
        if shutil.disk_usage(root).free < required:
            atomic_json(root / ".backup-status.json", {"status": "failed", "finished_at": stamp(),
                        "source": source, "error": "可用空间不足备份预留要求，现有恢复点保留"})
            raise BackupError("备份分区可用空间不足预留要求，拒绝启动；现有恢复点保留")
        name = "companion_" + now().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8] + ".dump"
        dump, globals_file, meta = paths(root, name)
        partial_dump = root / ("." + name + ".tmp")
        partial_globals = root / ("." + name + ".globals.tmp")
        state = {"status": "running", "started_at": stamp(), "source": source, "name": name}
        atomic_json(root / ".backup-status.json", state)
        committed = False
        try:
            dump_cmd, globals_cmd, toc_cmd = commands(transport, env)
            for cmd, target in ((dump_cmd, partial_dump), (globals_cmd, partial_globals)):
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as out:
                    run_command(cmd, env=env, output=out)
                    out.flush()
                    os.fsync(out.fileno())
                if regular(target).st_size <= 0:
                    raise BackupError("生成了空备份文件")
            with partial_dump.open("rb") as src:
                tables = run_command(toc_cmd, env=env, stdin=src).count(b"TABLE DATA")
            if tables < 20:
                raise BackupError("归档目录校验失败：少于20张表")
            data = {"schema_version": 1, "name": name, "source": source, "checked_at": stamp(),
                    "validation": "checksum_and_toc", "table_count": tables,
                    "dump_bytes": regular(partial_dump).st_size, "globals_bytes": regular(partial_globals).st_size,
                    "dump_sha256": digest(partial_dump), "globals_sha256": digest(partial_globals)}
            os.replace(partial_dump, dump)
            os.replace(partial_globals, globals_file)
            atomic_json(meta, data)
            committed = True
            # Only expire known complete sets, after the new recovery point commits.
            current, _, _ = inventory(root)
            keep = protected(current, now())
            expired = {row["name"] for row in current if row["integrity"] == "recorded" and row["name"] not in keep}
            if expired:
                verify_retained(root, current, expired)
            for name_to_delete in expired:
                delete_set(root, name_to_delete)
            atomic_json(root / ".backup-status.json", {**state, "status": "success", "finished_at": stamp()})
            return name
        except Exception as exc:
            atomic_json(root / ".backup-status.json", {**state, "status": "failed", "finished_at": stamp(),
                        "error": str(exc) if isinstance(exc, BackupError) else "备份操作失败，请检查主机日志"})
            raise
        finally:
            for p in (partial_dump, partial_globals):
                p.unlink(missing_ok=True)
            if not committed:
                for p in (dump, globals_file):
                    p.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "snapshot", "verify", "cutover-ready", "process-request"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--transport", choices=["docker", "tcp"], default="docker")
    args = parser.parse_args()
    try:
        if args.action == "process-request":
            result = process_request(args.root, args.transport, dict(os.environ))
        elif args.action == "create":
            result = {"name": create(args.root, args.transport, dict(os.environ), "scheduled")}
        elif args.action == "verify":
            with locked(args.root):
                result = verify(args.root, args.name or "", args.transport, dict(os.environ))
        elif args.action == "cutover-ready":
            result = cutover_ready(args.root)
        else:
            result = snapshot(args.root)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (BackupError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc) if isinstance(exc, (BackupError, ValueError)) else "备份目录不可访问"}, ensure_ascii=False))
        return 75 if isinstance(exc, BackupError) and "已有备份操作" in str(exc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
