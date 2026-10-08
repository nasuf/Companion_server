"""Failure-only, Linux worker snapshots; no application imports or local values.

CPython's C-level faulthandler works even when a native call retains the GIL.
Only the launcher's spawned children opt in. A pidfd and a private, per-run
registration bind the request to the worker we actually supervise.
"""
from __future__ import annotations

import atexit
import faulthandler
import itertools
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time
from typing import Any

SESSION_ENV = "COMPANION_WORKER_DIAGNOSTIC_SESSION"
MAX_TRACE_BYTES = 64 * 1024
MAX_FRAMES = 32
MAX_THREADS = 16
CAPTURE_SECONDS = 0.2
_child_fd: int | None = None


def _read(path: Path, limit: int = 16384) -> str:
    with path.open("rb") as file:
        return file.read(limit).decode("ascii", errors="replace")


def _start_ticks(pid: int) -> int:
    # The comm field may contain spaces or parentheses.
    tail = _read(Path(f"/proc/{pid}/stat")).rsplit(") ", 1)[1].split()
    return int(tail[19])  # field22, with field3 at index0


def _handler_present(pid: int) -> bool:
    for line in _read(Path(f"/proc/{pid}/status")).splitlines():
        if line.startswith("SigCgt:"):
            return bool(int(line.split(":", 1)[1].strip(), 16) & (1 << (signal.SIGUSR2 - 1)))
    return False


def _secure_fd(path: Path, *, write: bool = False) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND if write else os.O_RDONLY
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            raise ValueError("invalid private diagnostic file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _private_json(path: Path) -> dict[str, Any]:
    fd = _secure_fd(path)
    try:
        blob = os.read(fd, 2049)
    finally:
        os.close(fd)
    if len(blob) > 2048:
        raise ValueError("oversized registration")
    result = json.loads(blob)
    if not isinstance(result, dict):
        raise ValueError("invalid registration")
    return result


def bootstrap_worker_trace() -> None:
    """Called only from the launcher's __mp_main__ spawn bootstrap."""
    global _child_fd
    if _child_fd is not None or sys.platform != "linux":
        return
    root = os.environ.get(SESSION_ENV)
    if not root:
        return
    fd = None
    registered = False
    try:
        folder = Path(root)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            return
        session = _private_json(folder / "session.json")
        if session["parent_pid"] != os.getppid() or session["parent_ticks"] != _start_ticks(os.getppid()):
            return
        if signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
            return
        pid = os.getpid()
        fd = _secure_fd(folder / f"{pid}.trace", write=True)
        faulthandler.register(signal.SIGUSR2, file=fd, all_threads=True)
        registered = True
        trace_info = os.fstat(fd)
        marker = {"pid": pid, "ticks": _start_ticks(pid), "parent_pid": os.getppid(),
                  "trace_inode": trace_info.st_ino, "trace_device": trace_info.st_dev}
        ready_fd = _secure_fd(folder / f"{pid}.ready", write=True)
        try:
            os.write(ready_fd, json.dumps(marker).encode("ascii"))
        finally:
            os.close(ready_fd)
        _child_fd = fd
        atexit.register(_close_child_trace)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        if registered:
            faulthandler.unregister(signal.SIGUSR2)
        if fd is not None:
            os.close(fd)
        # Diagnostics are optional; never turn their setup failure into an API
        # startup failure or expose exception messages/paths in normal logs.


def _close_child_trace() -> None:
    global _child_fd
    if _child_fd is not None:
        faulthandler.unregister(signal.SIGUSR2)
        os.close(_child_fd)
        _child_fd = None


def native_snapshot(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    root = Path(f"/proc/{pid}")
    try:
        fields = dict(line.split(":", 1) for line in _read(root / "status").splitlines() if ":" in line)
        for key in ["VmRSS", "VmSwap", "Threads"]:
            if key in fields:
                result[key] = int(fields[key].split()[0])
        result["state"] = fields["State"].strip()[0]
        threads = []
        for entry in itertools.islice((root / "task").iterdir(), MAX_THREADS):
            try:
                values = _read(entry / "stat").rsplit(") ", 1)[1].split()
                wchan = _read(entry / "wchan", 81).strip()
                threads.append({"tid": int(entry.name), "state": values[0],
                                "wchan": wchan if re.fullmatch(r"[A-Za-z_0-9]{1,80}", wchan) else "unknown",
                                "user_ticks": int(values[11]), "system_ticks": int(values[12])})
            except (OSError, ValueError, IndexError):
                continue
        result["threads"] = threads
    except (OSError, ValueError, KeyError, IndexError):
        result["process_unavailable"] = True
    memory = {}
    for name in ["memory.current", "memory.swap.current"]:
        try:
            memory[name] = int(_read(Path("/sys/fs/cgroup") / name, 40))
        except (OSError, ValueError):
            continue
    try:
        memory["events"] = {k: int(v) for k, v in (line.split() for line in _read(Path("/sys/fs/cgroup/memory.events"), 512).splitlines())
                            if k in {"oom", "oom_kill", "oom_group_kill", "high", "max"}}
    except (OSError, ValueError):
        pass
    result["cgroup_memory"] = memory
    return result


def frame_locations(blob: bytes) -> list[dict[str, Any]]:
    current = []
    others = []
    is_current = False
    pattern = r'  File "([^"\n]+)", line (\d+) in ([^\n]+)'
    for text in blob[:MAX_TRACE_BYTES].decode("ascii", errors="replace").splitlines():
        if text.startswith(("Thread ", "Current thread ")):
            is_current = text.startswith("Current thread ")
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        filename, line, function = match.groups()
        allowed = filename.startswith(("/app/", "/usr/local/lib/python3.13/", "/worker-test/"))
        frame = {"file": filename[:256] if allowed and re.fullmatch(r"[A-Za-z_0-9./-]+", filename) else "outside_application",
                 "line": int(line[:9]), "function": function[:80] if re.fullmatch(r"[A-Za-z_0-9.<>]+", function) else "unknown",
                 "thread": "current" if is_current else "other"}
        (current if is_current else others).append(frame)
    # faulthandler emits the current thread last. Preserve its top frame even
    # when other threads have deeper stacks than our public output limit.
    return (current[:16] + others)[:MAX_FRAMES]


class TraceSession:
    """A private parent-owned directory with no persistent business data."""

    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="companion-worker-diagnostic-", ignore_cleanup_errors=True)
        self.folder = Path(self._temp.name)
        self._previous = os.environ.get(SESSION_ENV)
        fd = _secure_fd(self.folder / "session.json", write=True)
        try:
            os.write(fd, json.dumps({"parent_pid": os.getpid(), "parent_ticks": _start_ticks(os.getpid())}).encode("ascii"))
        finally:
            os.close(fd)
        os.environ[SESSION_ENV] = str(self.folder)

    def close(self) -> None:
        if self._previous is None:
            os.environ.pop(SESSION_ENV, None)
        else:
            os.environ[SESSION_ENV] = self._previous
        self._temp.cleanup()

    def discard(self, pid: int) -> None:
        for suffix in ["ready", "trace"]:
            try:
                (self.folder / f"{pid}.{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def capture(self, pid: int, *, allow_signal: bool) -> dict[str, Any]:
        start = time.monotonic()
        result: dict[str, Any] = {"native": native_snapshot(pid), "stack_status": "not_registered", "frames": [], "diagnostic_signal_sent": False}
        pidfd = trace_fd = None
        try:
            if not allow_signal:
                result["stack_status"] = "already_exited"
                return result
            if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
                result["stack_status"] = "pidfd_unavailable"
                return result
            pidfd = os.pidfd_open(pid, 0)
            marker = _private_json(self.folder / f"{pid}.ready")
            if marker.get("pid") != pid or marker.get("parent_pid") != os.getpid() or marker.get("ticks") != _start_ticks(pid):
                result["stack_status"] = "identity_mismatch"
                return result
            trace_fd = _secure_fd(self.folder / f"{pid}.trace")
            info = os.fstat(trace_fd)
            if info.st_ino != marker.get("trace_inode") or info.st_dev != marker.get("trace_device"):
                result["stack_status"] = "sink_mismatch"
                return result
            if not _handler_present(pid):
                result["stack_status"] = "handler_unregistered"
                return result
            # One capture per failed worker. No signal when our total budget is
            # already consumed; unsupported/unregistered workers keep upstream
            # kill/replacement behavior without a diagnostic signal.
            if time.monotonic() - start >= CAPTURE_SECONDS:
                result["stack_status"] = "budget_exhausted"
                return result
            signal.pidfd_send_signal(pidfd, signal.SIGUSR2, None, 0)
            result["diagnostic_signal_sent"] = True
            previous_size = -1
            quiet_since = time.monotonic()
            while time.monotonic() - start < CAPTURE_SECONDS:
                size = os.fstat(trace_fd).st_size
                now = time.monotonic()
                if size != previous_size:
                    previous_size, quiet_since = size, now
                elif size > 0 and now - quiet_since >= 0.01:
                    break
                time.sleep(min(0.005, max(0, CAPTURE_SECONDS - (time.monotonic() - start))))
            # Prefer the tail of an oversized C dump: it includes the current
            # thread, emitted last. Read at most64KiB; never publish raw content.
            size = os.fstat(trace_fd).st_size
            os.lseek(trace_fd, max(0, size - MAX_TRACE_BYTES), os.SEEK_SET)
            blob = os.read(trace_fd, MAX_TRACE_BYTES)
            result["frames"] = frame_locations(blob)
            result["trace_read_bytes"] = len(blob)
            result["trace_truncated"] = size > len(blob)
            result["stack_status"] = "captured" if result["frames"] else "no_frames_before_deadline"
        except (OSError, ValueError, KeyError, IndexError):
            # Never replace a supervised process failure with a diagnostics
            # exception, and never expose raw paths/messages or stack contents.
            result["stack_status"] = "unavailable"
        finally:
            for fd in [pidfd, trace_fd]:
                if fd is not None:
                    os.close(fd)
            result["capture_ms"] = round((time.monotonic() - start) * 1000, 2)
        return result


def create_trace_session() -> TraceSession | None:
    if sys.platform != "linux":
        return None
    try:
        return TraceSession()
    except (OSError, ValueError, KeyError, IndexError):
        return None
