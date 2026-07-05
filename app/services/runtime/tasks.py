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

# Backpressure: track in-flight background tasks so a storm (high QPS × many
# fan-out tasks per turn) can't silently overwhelm the event loop unbounded.
# We never DROP tasks (dropping memory extraction would lose data) — we let
# them run but emit a warning once we cross the high-water mark, giving ops a
# clear signal to scale out.
_inflight: set[asyncio.Task] = set()
_overflow_warned = False


def _high_water_mark() -> int:
    from app.config import settings
    return max(1, settings.background_task_max_concurrency)


def background_inflight_count() -> int:
    """Current number of in-flight fire-and-forget tasks (for metrics/tests)."""
    return len(_inflight)


def fire_background(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Returns the created Task so callers can hold a reference if they want
    to await/cancel it. Uncaught exceptions are logged at WARNING level;
    cancellation is silently ignored.

    Background task runs in an isolated copy of the current context with
    request-scoped state (e.g. LLM usage_tracker session) cleared, so e.g.
    background memory-extraction LLM calls don't inflate the parent chat's
    token totals.

    In-flight tasks are tracked; crossing `background_task_max_concurrency`
    logs a throttled warning (tasks are never dropped).
    """
    new_ctx = contextvars.copy_context()
    new_ctx.run(_isolate_request_scoped_state)
    task = asyncio.create_task(coro, context=new_ctx)
    _inflight.add(task)
    task.add_done_callback(_on_task_done)
    _maybe_warn_overflow()
    return task


def _maybe_warn_overflow() -> None:
    global _overflow_warned
    hwm = _high_water_mark()
    count = len(_inflight)
    if count > hwm and not _overflow_warned:
        _overflow_warned = True
        logger.warning(
            f"Background task backlog high: {count} in-flight (> {hwm}); "
            "consider scaling out — tasks are queued, not dropped.",
        )
    elif count <= hwm // 2:
        # Reset the latch once we drain well below the mark, so a later spike warns again.
        _overflow_warned = False


def _isolate_request_scoped_state() -> None:
    """Reset ContextVars that should not leak into background tasks.

    Imported lazily to keep this module a leaf — circular imports if we
    pulled them at module load.
    """
    from app.services.llm.usage_tracker import _current as _usage
    _usage.set(None)


def _on_task_done(t: asyncio.Task) -> None:
    _inflight.discard(t)
    _on_task_error(t)


def _on_task_error(t: asyncio.Task) -> None:
    if t.cancelled():
        return
    exc = t.exception()
    if exc is not None:
        logger.warning(f"Background task failed: {exc}", exc_info=exc)
