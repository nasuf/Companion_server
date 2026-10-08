"""Signal identity, confidentiality, bounded output and failure cleanup contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from unittest.mock import Mock

import pytest

from app import worker_diagnostics as diag


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "_start_ticks", lambda pid: 123)
    monkeypatch.setattr(diag, "native_snapshot", lambda pid: {"VmRSS": 300, "VmSwap": 100})
    monkeypatch.setattr(diag, "_handler_present", lambda pid: True)
    obj = diag.TraceSession()
    try:
        yield obj
    finally:
        obj.close()


def registration(session, pid=99, *, overrides=None, **changes):
    trace = session.folder / f"{pid}.trace"
    trace.write_bytes(b"")
    trace.chmod(0o600)
    info = trace.stat()
    data = {"pid": pid, "ticks": 123, "parent_pid": os.getpid(), "trace_inode": info.st_ino, "trace_device": info.st_dev, **changes, **(overrides or {})}
    marker = session.folder / f"{pid}.ready"
    marker.write_text(json.dumps(data))
    marker.chmod(0o600)
    return marker, trace


def pidfd(monkeypatch):
    # A real descriptor ensures every code path closes its acquired resources.
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(diag.os, "pidfd_open", lambda *args: fd, raising=False)
    send = Mock()
    monkeypatch.setattr(diag.signal, "pidfd_send_signal", send, raising=False)
    return fd, send


def test_private_session_environment_and_cleanup(monkeypatch):
    monkeypatch.setenv(diag.SESSION_ENV, "previous-private-session")
    monkeypatch.setattr(diag, "_start_ticks", lambda pid: 100)
    s = diag.TraceSession()
    folder = s.folder
    assert folder.stat().st_mode & 0o777 == 0o700
    assert (folder / "session.json").stat().st_mode & 0o777 == 0o600
    assert os.environ[diag.SESSION_ENV] == str(folder)
    s.close()
    assert not folder.exists()
    assert os.environ[diag.SESSION_ENV] == "previous-private-session"


def test_matching_registration_uses_pidfd_and_public_locations_only(session, monkeypatch):
    marker, trace = registration(session)
    fd, send = pidfd(monkeypatch)
    secret = "TEST_PRIVATE_VARIABLE_NOT_A_FRAME"
    def dump(*args):
        trace.write_text(f'Thread 0x1 (most recent call first):\n  File "/app/ping.py", line 42 in pong\nCurrent thread 0x2 (most recent call first):\n  File "/app/block.py", line 9 in blocked\n{secret}\n')
    send.side_effect = dump
    result = session.capture(99, allow_signal=True)
    send.assert_called_once_with(fd, signal.SIGUSR2, None, 0)
    assert result["stack_status"] == "captured"
    assert result["frames"][0] == {"file": "/app/block.py", "line": 9, "function": "blocked", "thread": "current"}
    assert secret not in json.dumps(result)
    with pytest.raises(OSError):os.fstat(fd)
    session.discard(99)
    assert not marker.exists() and not trace.exists()


@pytest.mark.parametrize("change", [{"pid": 100}, {"ticks": 124}, {"parent_pid": -1}])
def test_pid_start_clock_and_parent_mismatches_never_signal(session, monkeypatch, change):
    registration(session, overrides=change)
    fd, send = pidfd(monkeypatch)
    assert session.capture(99, allow_signal=True)["stack_status"] == "identity_mismatch"
    send.assert_not_called()
    with pytest.raises(OSError):os.fstat(fd)


@pytest.mark.parametrize("kind", ["missing", "symlink", "wrong_permissions", "oversized", "invalid_json", "non_object", "changed_sink", "linked_sink"])
def test_untrusted_registration_and_sink_never_signal(session, monkeypatch, kind):
    marker, trace = registration(session)
    if kind == "missing":marker.unlink()
    elif kind == "symlink":
        marker.unlink();marker.symlink_to(session.folder / "session.json")
    elif kind == "wrong_permissions":marker.chmod(0o644)
    elif kind == "oversized":marker.write_text("x" * 2049)
    elif kind == "invalid_json":marker.write_text("{")
    elif kind == "non_object":marker.write_text("[]")
    elif kind == "changed_sink":
        trace.unlink();trace.write_bytes(b"");trace.chmod(0o600)
        # Force inode mismatch even if a filesystem immediately reuses one.
        row=json.loads(marker.read_text());row["trace_inode"]=-1;marker.write_text(json.dumps(row))
    elif kind == "linked_sink":os.link(trace,session.folder / "second-link")
    fd, send = pidfd(monkeypatch)
    result=session.capture(99, allow_signal=True)
    assert result["stack_status"] in {"unavailable", "sink_mismatch"}
    send.assert_not_called()
    with pytest.raises(OSError):os.fstat(fd)


def test_already_exited_process_does_not_acquire_pidfd_or_signal(session, monkeypatch):
    open_fd = Mock();send = Mock()
    monkeypatch.setattr(diag.os,"pidfd_open",open_fd,raising=False)
    monkeypatch.setattr(diag.signal,"pidfd_send_signal",send,raising=False)
    assert session.capture(99,allow_signal=False)["stack_status"] == "already_exited"
    open_fd.assert_not_called();send.assert_not_called()


def test_missing_pidfd_support_never_falls_back_to_kill(session, monkeypatch):
    monkeypatch.delattr(diag.os,"pidfd_open",raising=False)
    kill=Mock();monkeypatch.setattr(diag.os,"kill",kill)
    assert session.capture(99,allow_signal=True)["stack_status"] == "pidfd_unavailable"
    kill.assert_not_called()


def test_stale_registration_after_handler_removal_never_signals(session, monkeypatch):
    registration(session)
    fd,send=pidfd(monkeypatch)
    monkeypatch.setattr(diag,"_handler_present",lambda pid:False)
    result=session.capture(99,allow_signal=True)
    assert result['stack_status']=='handler_unregistered' and not result['diagnostic_signal_sent']
    send.assert_not_called()
    with pytest.raises(OSError):os.fstat(fd)


def test_process_exit_race_returns_unavailable_and_closes_pidfd(session,monkeypatch):
    registration(session);fd,send=pidfd(monkeypatch)
    send.side_effect=ProcessLookupError()
    assert session.capture(99,allow_signal=True)["stack_status"] == "unavailable"
    with pytest.raises(OSError):os.fstat(fd)


def test_stopped_worker_has_bounded_capture_deadline(session,monkeypatch):
    registration(session);fd,send=pidfd(monkeypatch)
    # Isolate the clock object, never monkeypatch global stdlib time.
    tick=[0]
    def clock():tick[0]+=0.05;return tick[0]
    from types import SimpleNamespace
    monkeypatch.setattr(diag,"time",SimpleNamespace(monotonic=clock,sleep=lambda delay:None))
    result=session.capture(99,allow_signal=True)
    assert result["stack_status"] == "no_frames_before_deadline"
    assert send.call_count==1
    with pytest.raises(OSError):os.fstat(fd)


def test_budget_spent_on_native_collection_skips_signal(session,monkeypatch):
    registration(session);fd,send=pidfd(monkeypatch)
    from types import SimpleNamespace
    clock=iter([0,1,1])
    monkeypatch.setattr(diag,"time",SimpleNamespace(monotonic=lambda:next(clock)))
    assert session.capture(99,allow_signal=True)["stack_status"] == "budget_exhausted"
    send.assert_not_called()
    with pytest.raises(OSError):os.fstat(fd)


def test_oversized_dump_prefers_current_thread_and_bounds_read_and_frames(session,monkeypatch):
    _,trace=registration(session);fd,send=pidfd(monkeypatch)
    ordinary='  File "/app/other.py", line 1 in deeper\n'
    current='Current thread 0x2 (most recent call first):\n  File "/app/block.py", line 2 in blocked\n'
    send.side_effect=lambda *args:trace.write_text('Thread 0x1 (most recent call first):\n'+ordinary*3000+current)
    result=session.capture(99,allow_signal=True)
    assert result["trace_read_bytes"]==diag.MAX_TRACE_BYTES
    assert result["trace_truncated"] and len(result["frames"])==diag.MAX_FRAMES
    assert result["frames"][0]["function"]=='blocked'
    assert len(json.dumps(result))<16384


def test_frame_parser_discards_values_source_and_untrusted_names():
    blob=b'Current thread 0x2:\n  File "/private/secret-token.py", line 8 in user secret\n  File "/app/code.py", line 12 in safe\n    api_key = "SECRET_VALUE"\nSECRET_VALUE\n'
    result=diag.frame_locations(blob)
    assert result==[{'file':'outside_application','line':8,'function':'unknown','thread':'current'},{'file':'/app/code.py','line':12,'function':'safe','thread':'current'}]
    assert 'SECRET'not in json.dumps(result)


@pytest.mark.parametrize('condition',['no_session','not_linux','invalid_dir','bad_parent','existing_signal'])
def test_bootstrap_skip_never_changes_signal_or_opens_sink(session,monkeypatch,condition):
    monkeypatch.setattr(diag,'_child_fd',None)
    monkeypatch.setattr(diag.sys,'platform','linux')
    install=Mock();monkeypatch.setattr(diag.faulthandler,'register',install)
    if condition=='no_session':monkeypatch.delenv(diag.SESSION_ENV)
    elif condition=='not_linux':monkeypatch.setattr(diag.sys,'platform','darwin')
    elif condition=='invalid_dir':session.folder.chmod(0o755)
    elif condition=='bad_parent':pass  # session belongs to this process, not its parent
    else:
        monkeypatch.setattr(diag.os,'getppid',lambda:os.getpid())
        monkeypatch.setattr(diag.signal,'getsignal',lambda sig:signal.SIG_IGN)
    diag.bootstrap_worker_trace()
    install.assert_not_called()
    assert not (session.folder/f'{os.getpid()}.trace').exists()


def test_bootstrap_installs_before_registration_and_keeps_fd_until_exit(session,monkeypatch):
    monkeypatch.setattr(diag,'_child_fd',None);monkeypatch.setattr(diag.sys,'platform','linux')
    monkeypatch.setattr(diag.os,'getppid',lambda:os.getpid())
    monkeypatch.setattr(diag.signal,'getsignal',lambda sig:signal.SIG_DFL)
    install=Mock();remove=Mock();hook=Mock()
    monkeypatch.setattr(diag.faulthandler,'register',install)
    monkeypatch.setattr(diag.faulthandler,'unregister',remove)
    monkeypatch.setattr(diag.atexit,'register',hook)
    diag.bootstrap_worker_trace();fd=diag._child_fd
    assert fd is not None and os.fstat(fd).st_mode&0o777==0o600
    assert diag._private_json(session.folder/f'{os.getpid()}.ready')['pid']==os.getpid()
    install.assert_called_once_with(signal.SIGUSR2,file=fd,all_threads=True)
    diag.bootstrap_worker_trace();assert install.call_count==1
    diag._close_child_trace();remove.assert_called_once_with(signal.SIGUSR2)
    with pytest.raises(OSError):os.fstat(fd)


def test_missing_proc_start_clock_setup_is_optional(monkeypatch):
    monkeypatch.setattr(diag.sys,'platform','linux')
    monkeypatch.setattr(diag,'_start_ticks',Mock(side_effect=FileNotFoundError()))
    assert diag.create_trace_session()is None
