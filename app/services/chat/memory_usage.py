"""One submission contract for memory usage in both chat executors.

The existing signal means retrieval-selected, not confirmed consumption by the
final prompt. Diagnostics describe background submission only: neither a queued
coroutine nor its completion proves a lifecycle transaction changed a score.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any

from app.services.memory.lifecycle.lazy_update import (
    USAGE_MAX_CANDIDATES,
    record_memory_usage,
)
from app.services.memory.retrieval.access_log import log_memory_access
from app.services.runtime.tasks import fire_background as schedule_background

logger = logging.getLogger(__name__)
BackgroundScheduler = Callable[[Coroutine[Any, Any, Any]], Any]


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _unique_ids(values: Sequence[str], excluded: set[str]) -> tuple[list[str], int]:
    """Keep stable order and stop after the shared distinct-ID limit is crossed."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [], 1
    result: list[str] = []
    invalid = 0
    for value in values:
        if not _valid_id(value):
            invalid += 1
        elif value not in excluded:
            excluded.add(value)
            result.append(value)
            if len(excluded) > USAGE_MAX_CANDIDATES:
                break
    return result, invalid


def _submit(
    coroutine: Coroutine[Any, Any, Any],
    scheduler: BackgroundScheduler,
    effect: str,
) -> bool:
    try:
        scheduler(coroutine)
    except BaseException as error:
        # A scheduler that rejects the coroutine leaves ownership here. Avoid
        # leaking an unawaited coroutine, including when cancellation propagates.
        coroutine.close()
        if not isinstance(error, Exception):
            raise
        logger.warning(
            "memory usage %s submission failed (%s)", effect, type(error).__name__
        )
        return False
    return True


def submit_memory_usage(
    *,
    event_id: str | None,
    user_id: str,
    workspace_id: str | None,
    selected_ids: Sequence[str],
    candidate_ids: Sequence[str] = (),
    fire_background: BackgroundScheduler = schedule_background,
) -> dict[str, Any]:
    """Submit independent audit/lifecycle effects with one explicit turn scope.

    ``event_id`` must identify the root persisted user message, including for
    sub-intents. Lifecycle storage revalidates this event and applies its receipt
    deduplication; submission diagnostics deliberately contain no resource IDs.
    """
    seen: set[str] = set()
    selected, invalid = _unique_ids(selected_ids, seen)
    candidates, invalid_candidates = (
        _unique_ids(candidate_ids, seen)
        if len(seen) <= USAGE_MAX_CANDIDATES
        else ([], 0)
    )
    diagnostics: dict[str, Any] = {
        "status": "not_requested",
        "semantics": "retrieval_selected",
        "sampled_at": datetime.now(UTC).isoformat(),
        "selected_count": len(selected),
        "candidate_only_count": len(candidates),
        "invalid_id_count": min(invalid + invalid_candidates, USAGE_MAX_CANDIDATES + 1),
        "counts_truncated": (
            len(seen) > USAGE_MAX_CANDIDATES
            or invalid + invalid_candidates > USAGE_MAX_CANDIDATES + 1
        ),
        "audit_submission": "not_requested",
        "lifecycle_submission": "not_requested",
    }
    if not seen:
        return diagnostics
    if not _valid_id(event_id) or not _valid_id(user_id) or (
        workspace_id is not None and not _valid_id(workspace_id)
    ):
        diagnostics["status"] = "missing_identity"
        return diagnostics
    if len(seen) > USAGE_MAX_CANDIDATES:
        diagnostics["status"] = "rejected_limit"
        return diagnostics

    results: list[bool] = []
    if selected:
        accepted = _submit(
            log_memory_access(user_id, selected, workspace_id=workspace_id),
            fire_background,
            "audit",
        )
        results.append(accepted)
        diagnostics["audit_submission"] = "submitted" if accepted else "failed"
    accepted = _submit(
        record_memory_usage(
            contributed_ids=selected,
            accessed_ids=candidates,
            event_id=event_id,
            user_id=user_id,
            workspace_id=workspace_id,
        ),
        fire_background,
        "lifecycle",
    )
    results.append(accepted)
    diagnostics["lifecycle_submission"] = "submitted" if accepted else "failed"
    diagnostics["status"] = (
        "submitted" if all(results)
        else "submission_partial" if any(results)
        else "submission_failed"
    )
    return diagnostics
