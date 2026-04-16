"""Shared Redis key prefixes for the memory-generation subsystem.

Keeping all `memgen:*` namespace builders in one place avoids drift between
`generation_lock` and `init_report`.
"""

from __future__ import annotations

_NAMESPACE = "memgen"


def lock_key(agent_id: str) -> str:
    return f"{_NAMESPACE}:lock:{agent_id}"


def report_key(agent_id: str) -> str:
    return f"{_NAMESPACE}:report:{agent_id}"
