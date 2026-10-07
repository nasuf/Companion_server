"""Internal SQL worker contracts; none of these objects authenticate a caller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import math

from app.services.runtime.chat_ingress_contracts import _identity
from app.services.runtime.execution_scope import ExecutionScope


class LeaseLost(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Execution lease is no longer valid")


class WorkerStopRequired(RuntimeError):
    """A handler ignored cancellation; the owning worker must stop polling."""


def _seconds(value: float, maximum: int) -> None:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError("Invalid bounded duration")


@dataclass(frozen=True, slots=True)
class LeasePolicy:
    lease_seconds: float = 60
    heartbeat_seconds: float = 15
    retry_seconds: float = 5
    cancellation_seconds: float = 2
    scan_limit: int = 64

    def __post_init__(self) -> None:
        for name in (
            "lease_seconds",
            "heartbeat_seconds",
            "retry_seconds",
            "cancellation_seconds",
        ):
            _seconds(getattr(self, name), 3600)
        if self.heartbeat_seconds * 3 > self.lease_seconds:
            raise ValueError("Heartbeat must leave at least two renewal intervals")
        if type(self.scan_limit) is not int or not 1 <= self.scan_limit <= 256:
            raise ValueError("Invalid candidate scan bound")


@dataclass(frozen=True, slots=True)
class HandlerSpec:
    name: str
    job_key: str
    versions: tuple[tuple[str, int], ...]
    kind: str = "chat"
    payload_version: int = 1
    max_execution_seconds: float = 300
    retry_safe: bool = False
    executor: str = "langgraph"

    def __post_init__(self) -> None:
        _identity(self.name)
        _identity(self.job_key)
        if self.kind not in {"chat", "task", "background"}:
            raise ValueError("Unsupported run kind")
        if self.executor not in {"langgraph", "legacy"}:
            raise ValueError("Unsupported chat executor")
        if type(self.payload_version) is not int or self.payload_version < 1:
            raise ValueError("Invalid payload version")
        if type(self.versions) is not tuple or not 1 <= len(self.versions) <= 32:
            raise ValueError("Explicit graph/state compatibility pairs required")
        for pair in self.versions:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("Invalid compatibility pair")
            _identity(pair[0])
            if type(pair[1]) is not int or pair[1] < 1:
                raise ValueError("Invalid state version")
        if (
            len(set(self.versions)) != len(self.versions)
            or type(self.retry_safe) is not bool
        ):
            raise ValueError("Invalid handler contract")
        _seconds(self.max_execution_seconds, 3600)


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: str
    run_id: str
    scope: ExecutionScope
    handler: str
    worker_id: str
    fencing_token: int
    attempts: int
    lease_expires_at: datetime
    deadline_at: datetime | None
    payload_json: str
    config_json: str
    prompts_json: str
    budget_json: str
    graph_version: str
    state_version: int
    _revoked: asyncio.Event = field(
        default_factory=asyncio.Event, compare=False, repr=False
    )

    def revoke(self) -> None:
        self._revoked.set()

    def require_active(self) -> None:
        if self._revoked.is_set():
            raise LeaseLost()
