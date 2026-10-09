"""The diagnostics must preserve Uvicorn decisions and expose only bounded facts."""
from __future__ import annotations

import argparse
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import uvicorn.supervisors.multiprocess as upstream

from app import api_server


@pytest.fixture
def diagnostic_e2e(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("test_api_server_diagnostics_e2e")


def test_e2e_cleanup_waits_for_retiring_worker_without_weakening_checks(diagnostic_e2e, monkeypatch):
    ready = {"registrations": [{"pid": 2, "trace_bytes": 0}, {"pid": 3, "trace_bytes": 0}], "trace_files": 2}
    retiring = {"registrations": [*ready["registrations"], {"pid": 1, "trace_bytes": 32}], "trace_files": 3}
    probe = Mock(side_effect=[retiring, ready])
    clock = SimpleNamespace(monotonic=Mock(return_value=0), sleep=Mock())
    monkeypatch.setattr(diagnostic_e2e, "registrations", probe)
    monkeypatch.setattr(diagnostic_e2e, "time", clock)
    assert diagnostic_e2e.wait_clean_registrations("owned-container", {2, 3}) == ready
    assert probe.call_count == 2
    clock.sleep.assert_called_once_with(0.1)


def test_e2e_cleanup_still_fails_when_sinks_leak(diagnostic_e2e, monkeypatch):
    probe = Mock(return_value={"registrations": [{"pid": 2, "trace_bytes": 0}, {"pid": 3, "trace_bytes": 0}], "trace_files": 3})
    clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 0, 10]), sleep=Mock())
    monkeypatch.setattr(diagnostic_e2e, "registrations", probe)
    monkeypatch.setattr(diagnostic_e2e, "time", clock)
    with pytest.raises(AssertionError, match="did not settle"):
        diagnostic_e2e.wait_clean_registrations("owned-container", {2, 3})
    assert probe.call_count == 2


def test_e2e_cleanup_does_not_hide_private_permission_failure(diagnostic_e2e, monkeypatch):
    monkeypatch.setattr(diagnostic_e2e, "registrations", Mock(side_effect=AssertionError("unsafe modes")))
    with pytest.raises(AssertionError, match="unsafe modes"):
        diagnostic_e2e.wait_clean_registrations("owned-container", {2, 3})


class FakeProcess:
    def __init__(self, *, alive=True, exitcode=None, pid=7):
        self.alive = alive
        self.exitcode = exitcode
        self.pid = pid
        self.timeouts = []
        self.actions = []

    def is_alive(self, timeout=5):
        self.timeouts.append(timeout)
        return self.alive

    def kill(self):
        self.actions.append("kill")
        if self.exitcode is None:
            self.exitcode = -9

    def join(self):
        self.actions.append("join")

    def start(self):
        self.actions.append("start")

    def terminate(self):
        self.actions.append("terminate")
        self.exitcode = -15

    def wait_until_ready(self, timeout, should_exit):
        return self.alive


def diagnostics(caplog):
    return [json.loads(record.getMessage().split("worker_diagnostic ", 1)[1])
            for record in caplog.records if record.getMessage().startswith("worker_diagnostic ")]


def supervisor(processes):
    result = api_server.DiagnosedMultiprocess.__new__(api_server.DiagnosedMultiprocess)
    result.processes = processes
    result.config = SimpleNamespace(timeout_worker_healthcheck=60)
    result.sockets = []
    result.should_exit = threading.Event()
    return result


def test_healthy_probe_delegates_exactly_once_and_stays_silent(caplog):
    process = FakeProcess()
    observed = api_server.ObservedProcess(process)
    assert observed.is_alive(timeout=60)
    assert observed.pid == 7
    assert process.timeouts == [60]
    observed.join()
    assert diagnostics(caplog) == []


@pytest.mark.parametrize("exitcode,reason", [
    (None, "unresponsive_before_replacement"),
    (-9, "exited_before_replacement"),
    (3, "exited_before_replacement"),
    (0, "exited_before_replacement"),
])
def test_failure_records_before_and_after_join_without_diagnosing_unobserved_cause(caplog, exitcode, reason):
    process = FakeProcess(alive=False, exitcode=exitcode)
    process.credentials = "must-never-be-logged"
    observed = api_server.ObservedProcess(process)
    assert not observed.is_alive(timeout=60)
    observed.kill()
    observed.join()
    records = diagnostics(caplog)
    assert len(records) == 2
    assert records[0] == {
        "event": "api_worker_unhealthy", "worker_pid": 7, "reason": reason,
        "exitcode_before_replacement": exitcode, "healthcheck_timeout_seconds": 60,
    }
    assert records[1] == {**records[0], "event": "api_worker_failure_joined",
                          "exitcode_after_join": -9 if exitcode is None else exitcode}
    assert process.timeouts == [60]
    assert "must-never-be-logged" not in caplog.text
    observed.join()
    assert len(diagnostics(caplog)) == 2


def test_probe_exceptions_propagate_without_false_exit_report(caplog):
    process = FakeProcess()
    process.is_alive = Mock(side_effect=RuntimeError("probe error"))
    observed = api_server.ObservedProcess(process)
    with pytest.raises(RuntimeError, match="probe error"):
        observed.is_alive(timeout=60)
    assert diagnostics(caplog) == []


def test_healthy_worker_never_requests_a_snapshot():
    session = Mock()
    observed = api_server.ObservedProcess(FakeProcess(), session)
    assert observed.is_alive(timeout=60)
    session.capture.assert_not_called()
    observed.join()
    session.discard.assert_called_once_with(7)


@pytest.mark.parametrize("exitcode", [None, -9, 3])
def test_failure_snapshot_signal_permission_follows_actual_exit_state(exitcode, caplog):
    session = Mock()
    session.capture.return_value = {"stack_status": "captured", "frames": []}
    observed = api_server.ObservedProcess(FakeProcess(alive=False, exitcode=exitcode), session)
    assert not observed.is_alive(timeout=60)
    session.capture.assert_called_once_with(7, allow_signal=exitcode is None)
    assert diagnostics(caplog)[0]["failure_snapshot"] == session.capture.return_value


def test_snapshot_error_never_changes_worker_replacement_or_leaks_exception(monkeypatch, caplog):
    session = Mock()
    session.capture.side_effect = RuntimeError("private-diagnostic-secret")
    failed = FakeProcess(alive=False)
    replacement = FakeProcess(pid=9)
    monkeypatch.setattr(upstream, "Process", lambda config, sockets: replacement)
    parent = supervisor([failed])
    parent.trace_session = session
    parent.keep_subprocess_alive()
    assert failed.actions == ["kill", "join"]
    assert replacement.actions == ["start"]
    assert diagnostics(caplog)[0]["failure_snapshot"] == {"stack_status": "collection_failed"}
    assert "private-diagnostic-secret" not in caplog.text


def test_upstream_replaces_one_failure_and_observes_its_replacement(monkeypatch, caplog):
    failed = FakeProcess(alive=False)
    survivor = FakeProcess(pid=8)
    replacement = FakeProcess(pid=9)
    monkeypatch.setattr(upstream, "Process", lambda config, sockets: replacement)
    parent = supervisor([failed, survivor])
    parent.keep_subprocess_alive()
    assert failed.actions == ["kill", "join"]
    assert replacement.actions == ["start"]
    assert survivor.actions == []
    assert parent.processes[0] is replacement
    assert parent.processes[1]._process is survivor
    parent.keep_subprocess_alive()
    assert parent.processes[0]._process is replacement
    assert replacement.timeouts == [60]
    assert len(diagnostics(caplog)) == 2


def test_upstream_stops_on_startup_failure_instead_of_restart_loop(monkeypatch, caplog):
    failed = FakeProcess(alive=False, exitcode=upstream.STARTUP_FAILURE)
    create = Mock()
    monkeypatch.setattr(upstream, "Process", create)
    parent = supervisor([failed])
    parent.keep_subprocess_alive()
    assert parent.should_exit.is_set()
    create.assert_not_called()
    assert diagnostics(caplog)[-1]["exitcode_after_join"] == upstream.STARTUP_FAILURE


def test_upstream_shutdown_and_hup_keep_lifecycle_and_no_failure_warning(monkeypatch, caplog):
    old = FakeProcess()
    replacement = FakeProcess(pid=9)
    monkeypatch.setattr(upstream, "Process", lambda config, sockets: replacement)
    parent = supervisor([old])
    parent.keep_subprocess_alive()
    parent.restart_all()
    assert old.actions == ["terminate", "join"]
    assert replacement.actions == ["start"]
    parent.keep_subprocess_alive()
    parent.should_exit.set()
    parent.terminate_all()
    parent.join_all()
    assert replacement.actions == ["start", "terminate", "join"]
    assert diagnostics(caplog) == []


def test_qualification_guard_precedes_binding_or_start(monkeypatch):
    monkeypatch.setattr(api_server.uvicorn, "__version__", "unqualified")
    config = Mock(workers=2)
    with pytest.raises(RuntimeError, match="qualification"):
        api_server.run_server(config)
    config.bind_socket.assert_not_called()


def test_multiworker_run_uses_parent_and_closes_bound_socket(monkeypatch):
    config = Mock(workers=2)
    parent = Mock()
    create_parent = Mock(return_value=parent)
    monkeypatch.setattr(api_server, "DiagnosedMultiprocess", create_parent)
    api_server.run_server(config)
    config.load_app.assert_not_called()
    create_parent.assert_called_once_with(config, sockets=[config.bind_socket.return_value])
    parent.run.assert_called_once_with()
    config.bind_socket.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("started", [True, False])
def test_singleworker_retains_startup_failure_exit_code(monkeypatch, started):
    config = Mock(workers=1)
    server = Mock(started=started)
    monkeypatch.setattr(api_server, "Server", Mock(return_value=server))
    if started:
        api_server.run_server(config)
    else:
        with pytest.raises(SystemExit) as exc:
            api_server.run_server(config)
        assert exc.value.code == upstream.STARTUP_FAILURE
    config.load_app.assert_called_once_with()
    config.bind_socket.assert_not_called()
    server.run.assert_called_once_with()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-2", "invalid"])
def test_invalid_heartbeat_budget_rejected_before_start(value):
    with pytest.raises(argparse.ArgumentTypeError):
        api_server._positive_number(value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_invalid_worker_count_rejected_before_start(value):
    with pytest.raises(argparse.ArgumentTypeError):
        api_server._positive_integer(value)


def test_launcher_preserves_configured_workers_and_budget(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    monkeypatch.setenv("UVICORN_WORKER_HEALTHCHECK_TIMEOUT", "42")
    monkeypatch.setattr(api_server.sys, "argv", ["api_server"])
    monkeypatch.setattr(api_server.sys, "path", list(api_server.sys.path))
    run = Mock()
    monkeypatch.setattr(api_server, "run_server", run)
    monkeypatch.setattr(api_server, "Config", lambda app, **kwargs: SimpleNamespace(app=app, **kwargs))
    api_server.main()
    config = run.call_args.args[0]
    assert (config.app, config.host, config.port, config.workers, config.timeout_worker_healthcheck) == (
        "app.main:app", "0.0.0.0", 8000, 3, 42)
