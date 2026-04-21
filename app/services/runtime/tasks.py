"""Fire-and-forget asyncio task scheduling with error logging.

Single source of truth replacing the duplicated `_fire_background` helpers
in `chat/orchestrator.py`, `chat/multi_intent.py`, and `chat/tracing.py`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def fire_background(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Returns the created Task so callers can hold a reference if they want
    to await/cancel it. Uncaught exceptions are logged at WARNING level;
    cancellation is silently ignored.
    """
    task = asyncio.create_task(coro)
    task.add_done_callback(_on_task_error)
    return task


def _on_task_error(t: asyncio.Task) -> None:
    if t.cancelled():
        return
    exc = t.exception()
    if exc is not None:
        logger.warning(f"Background task failed: {exc}")
