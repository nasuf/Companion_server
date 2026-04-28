"""Fire-and-forget asyncio task scheduling with error logging.

Single source of truth replacing the duplicated `_fire_background` helpers
in `chat/orchestrator.py`, `chat/multi_intent.py`, and `chat/tracing.py`.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def fire_background(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Returns the created Task so callers can hold a reference if they want
    to await/cancel it. Uncaught exceptions are logged at WARNING level;
    cancellation is silently ignored.

    Background task runs in an isolated copy of the current context with
    request-scoped state (e.g. LLM usage_tracker session) cleared, so e.g.
    background memory-extraction LLM calls don't inflate the parent chat's
    token totals.
    """
    new_ctx = contextvars.copy_context()
    new_ctx.run(_isolate_request_scoped_state)
    task = asyncio.create_task(coro, context=new_ctx)
    task.add_done_callback(_on_task_error)
    return task


def _isolate_request_scoped_state() -> None:
    """Reset ContextVars that should not leak into background tasks.

    Imported lazily to keep this module a leaf — circular imports if we
    pulled them at module load.
    """
    from app.services.llm.usage_tracker import _current as _usage
    _usage.set(None)


def _on_task_error(t: asyncio.Task) -> None:
    if t.cancelled():
        return
    exc = t.exception()
    if exc is not None:
        logger.warning(f"Background task failed: {exc}")
