"""Pinned Uvicorn launcher with bounded, credential-free worker diagnostics.

The probe delegates worker lifecycle decisions to Uvicorn 0.54.0. It does not
change polling, healthcheck deadlines, restart, startup-failure or signal rules.
Requalify this small supervisor adapter when upgrading the pinned dependency.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

from app.worker_diagnostics import bootstrap_worker_trace, create_trace_session, TraceSession

# CPython spawn reimports this -m entry point as __mp_main__ before invoking
# Uvicorn's unchanged child target. Register before potentially slow app imports.
if __name__ == "__mp_main__":
    bootstrap_worker_trace()

import uvicorn
from uvicorn import Config, Server
from uvicorn.config import STARTUP_FAILURE
from uvicorn.supervisors.multiprocess import Multiprocess, Process

SUPPORTED_UVICORN = "0.54.0"
logger = logging.getLogger("uvicorn.error")


class ObservedProcess:
    """Parent-only proxy; the spawned child still uses Uvicorn's own target."""

    def __init__(self, process: Process, trace_session: TraceSession | None = None) -> None:
        self._process = process
        self._failure: dict[str, Any] | None = None
        self._trace_session = trace_session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._process, name)

    def is_alive(self, timeout: float = 5) -> bool:
        alive = self._process.is_alive(timeout=timeout)
        if not alive:
            exitcode = self._process.exitcode
            self._failure = {
                "event": "api_worker_unhealthy",
                "worker_pid": self._process.pid,
                "reason": "exited_before_replacement" if exitcode is not None else "unresponsive_before_replacement",
                "exitcode_before_replacement": exitcode,
                "healthcheck_timeout_seconds": timeout,
            }
            if self._trace_session is not None:
                try:
                    self._failure["failure_snapshot"] = self._trace_session.capture(
                        self._process.pid, allow_signal=exitcode is None,
                    )
                except Exception:
                    # Optional observation must never prevent upstream worker
                    # replacement. Do not log raw diagnostic exceptions.
                    self._failure["failure_snapshot"] = {"stack_status": "collection_failed"}
            logger.warning("worker_diagnostic %s", json.dumps(self._failure, sort_keys=True))
        return alive

    def join(self) -> None:
        self._process.join()
        if self._failure is not None:
            logger.warning("worker_diagnostic %s", json.dumps({
                **self._failure,
                "event": "api_worker_failure_joined",
                "exitcode_after_join": self._process.exitcode,
            }, sort_keys=True))
            self._failure = None
        if self._trace_session is not None:
            self._trace_session.discard(self._process.pid)


class DiagnosedMultiprocess(Multiprocess):
    def _observe(self) -> None:
        # Uvicorn creates ordinary Process objects on startup and replacement.
        # Wrap them only in the parent; leave the upstream decision tree intact.
        for index, process in enumerate(self.processes):
            if not isinstance(process, ObservedProcess):
                self.processes[index] = ObservedProcess(process, getattr(self, "trace_session", None))  # type: ignore[assignment]

    def init_processes(self) -> None:
        super().init_processes()
        self._observe()

    def restart_all(self) -> None:
        self._observe()
        super().restart_all()
        self._observe()

    def keep_subprocess_alive(self) -> None:
        self._observe()
        super().keep_subprocess_alive()


def run_server(config: Config) -> None:
    if uvicorn.__version__ != SUPPORTED_UVICORN:
        raise RuntimeError("API supervisor dependency changed; runtime qualification is required")
    if config.workers == 1:
        config.load_app()
    server = Server(config=config)
    sock = None
    trace_session = None
    try:
        if config.workers > 1:
            trace_session = create_trace_session()
            sock = config.bind_socket()
            parent = DiagnosedMultiprocess(config, sockets=[sock])
            parent.trace_session = trace_session
            parent.run()
        else:
            server.run()
    except KeyboardInterrupt:
        pass
    finally:
        if sock is not None:
            sock.close()
        if trace_session is not None:
            trace_session.close()
    if not server.started and config.workers == 1:
        raise SystemExit(STARTUP_FAILURE)


def _positive_number(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Companion API supervisor")
    parser.add_argument("app", nargs="?", default="app.main:app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_positive_integer, default=8000)
    parser.add_argument("--workers", type=_positive_integer, default=os.getenv("WEB_CONCURRENCY") or "2")
    parser.add_argument("--timeout-worker-healthcheck", type=_positive_number,
                        default=os.getenv("UVICORN_WORKER_HEALTHCHECK_TIMEOUT") or "60")
    parser.add_argument("--app-dir", default="")
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("port must be at most 65535")
    sys.path.insert(0, args.app_dir)
    config = Config(args.app, host=args.host, port=args.port, workers=args.workers,
                    timeout_worker_healthcheck=args.timeout_worker_healthcheck)
    run_server(config)


if __name__ == "__main__":
    main()
