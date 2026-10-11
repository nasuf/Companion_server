"""Executor-independent usage submission, identity and failure contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.chat import memory_usage as usage


@pytest.fixture
def effects(monkeypatch):
    audit = AsyncMock()
    lifecycle = AsyncMock(return_value=0)
    monkeypatch.setattr(usage, "log_memory_access", audit)
    monkeypatch.setattr(usage, "record_memory_usage", lifecycle)
    queued = []

    def submit(**kwargs):
        return usage.submit_memory_usage(
            **{
                "event_id": "persisted-root-message",
                "user_id": "owner-private",
                "workspace_id": "workspace-private",
                "selected_ids": [],
                "fire_background": queued.append,
                **kwargs,
            }
        )

    yield SimpleNamespace(audit=audit, lifecycle=lifecycle, queued=queued, submit=submit)
    for coroutine in queued:
        coroutine.close()


@pytest.mark.asyncio
async def test_both_effects_receive_same_explicit_scope_and_selected_wins(effects):
    result = effects.submit(
        selected_ids=["b", "a", "b"], candidate_ids=["c", "a", "c", "d"]
    )
    assert result["status"] == "submitted"
    assert result["selected_count"] == result["candidate_only_count"] == 2
    assert result["semantics"] == "retrieval_selected"
    assert result["audit_submission"] == result["lifecycle_submission"] == "submitted"
    assert result["counts_truncated"] is False
    # Submission is asynchronous; no effect has been awaited yet.
    effects.audit.assert_not_awaited()
    effects.lifecycle.assert_not_awaited()
    await asyncio.gather(*effects.queued)
    effects.audit.assert_awaited_once_with(
        "owner-private", ["b", "a"], workspace_id="workspace-private"
    )
    effects.lifecycle.assert_awaited_once_with(
        contributed_ids=["b", "a"], accessed_ids=["c", "d"],
        event_id="persisted-root-message", user_id="owner-private",
        workspace_id="workspace-private",
    )
    # Even a zero-row lifecycle result is not converted into an applied claim.
    assert result["status"] == "submitted"


@pytest.mark.asyncio
async def test_candidate_only_has_no_audit_or_contribution(effects):
    result = effects.submit(candidate_ids=["candidate", "candidate"])
    assert result["audit_submission"] == "not_requested"
    assert result["candidate_only_count"] == 1
    assert len(effects.queued) == 1
    await effects.queued[0]
    effects.audit.assert_not_called()
    assert effects.lifecycle.await_args.kwargs["contributed_ids"] == []
    assert effects.lifecycle.await_args.kwargs["accessed_ids"] == ["candidate"]


def test_no_memory_ids_never_constructs_effects(effects):
    assert effects.submit()["status"] == "not_requested"
    assert effects.queued == []
    effects.audit.assert_not_called()
    effects.lifecycle.assert_not_called()


@pytest.mark.parametrize("overrides", [
    {"event_id": None}, {"event_id": ""}, {"event_id": " "},
    {"event_id": 42}, {"user_id": None}, {"user_id": ""},
    {"user_id": " owner-private"}, {"workspace_id": ""},
    {"workspace_id": ["workspace-private"]},
])
def test_invalid_identity_prevents_every_submission(effects, overrides):
    result = effects.submit(selected_ids=["selected"], **overrides)
    assert result["status"] == "missing_identity"
    assert effects.queued == []
    effects.audit.assert_not_called()
    effects.lifecycle.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_null_workspace_is_preserved(effects):
    result = effects.submit(selected_ids=["selected"], workspace_id=None)
    assert result["status"] == "submitted"
    await asyncio.gather(*effects.queued)
    assert effects.audit.await_args.kwargs["workspace_id"] is None
    assert effects.lifecycle.await_args.kwargs["workspace_id"] is None


@pytest.mark.asyncio
async def test_malformed_ids_are_not_coerced_into_database_ids(effects):
    result = effects.submit(
        selected_ids=[None, {}, [], 3, "", " ", " padded", "selected"],
        candidate_ids=[False, "trailing ", "selected", "candidate"],
    )
    assert result["invalid_id_count"] == 9
    assert result["selected_count"] == result["candidate_only_count"] == 1
    await asyncio.gather(*effects.queued)
    assert effects.lifecycle.await_args.kwargs["contributed_ids"] == ["selected"]
    assert effects.lifecycle.await_args.kwargs["accessed_ids"] == ["candidate"]


@pytest.mark.parametrize("malformed", ["whole-id", b"whole-id", None, 1, {}])
def test_malformed_collection_is_not_iterated_as_ids(effects, malformed):
    result = effects.submit(selected_ids=malformed)
    assert result["status"] == "not_requested"
    assert result["invalid_id_count"] == 1
    assert effects.queued == []


@pytest.mark.parametrize("distinct", [500, 501])
def test_combined_distinct_limit_is_shared_with_lifecycle(effects, distinct):
    result = effects.submit(
        selected_ids=[f"m-{i}" for i in range(250)],
        candidate_ids=[f"m-{i}" for i in range(distinct)],
    )
    assert usage.USAGE_MAX_CANDIDATES == 500
    assert result["status"] == ("submitted" if distinct == 500 else "rejected_limit")
    assert result["counts_truncated"] is (distinct > 500)
    assert len(effects.queued) == (2 if distinct == 500 else 0)
    if distinct > 500:
        effects.audit.assert_not_called()
        effects.lifecycle.assert_not_called()


def test_duplicate_ids_do_not_consume_distinct_limit(effects):
    result = effects.submit(selected_ids=["selected"] * 501)
    assert result["status"] == "submitted"
    assert result["selected_count"] == 1


def test_rejected_selection_is_bounded_and_does_not_submit_partial_subset(effects):
    result = effects.submit(selected_ids=[f"m-{i}" for i in range(900)])
    assert result["status"] == "rejected_limit"
    assert result["selected_count"] == 501
    assert result["counts_truncated"] is True
    assert effects.queued == []
    effects.audit.assert_not_called()
    effects.lifecycle.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_indices", [{0}, {1}, {0, 1}])
async def test_scheduler_failure_closes_rejected_coroutine_and_keeps_other_effect(
    effects, rejected_indices, caplog
):
    rejected = []
    calls = 0

    def scheduler(coroutine):
        nonlocal calls
        index = calls
        calls += 1
        if index in rejected_indices:
            rejected.append(coroutine)
            raise RuntimeError("private error text must not be reported")
        effects.queued.append(coroutine)

    result = effects.submit(selected_ids=["selected"], fire_background=scheduler)
    assert calls == 2
    assert all(coroutine.cr_frame is None for coroutine in rejected)
    assert result["status"] == (
        "submission_failed" if len(rejected_indices) == 2 else "submission_partial"
    )
    assert result["audit_submission"] == ("failed" if 0 in rejected_indices else "submitted")
    assert result["lifecycle_submission"] == ("failed" if 1 in rejected_indices else "submitted")
    await asyncio.gather(*effects.queued)
    assert effects.audit.await_count == (0 if 0 in rejected_indices else 1)
    assert effects.lifecycle.await_count == (0 if 1 in rejected_indices else 1)
    assert "private error text" not in caplog.text


@pytest.mark.parametrize("cancel_index", [0, 1])
def test_scheduler_cancellation_propagates_without_closing_accepted_effects(
    effects, cancel_index
):
    rejected = []
    calls = 0

    def scheduler(coroutine):
        nonlocal calls
        index = calls
        calls += 1
        if index == cancel_index:
            rejected.append(coroutine)
            raise asyncio.CancelledError()
        effects.queued.append(coroutine)

    with pytest.raises(asyncio.CancelledError):
        effects.submit(selected_ids=["selected"], fire_background=scheduler)
    assert calls == cancel_index + 1
    assert rejected[0].cr_frame is None
    assert all(coroutine.cr_frame is not None for coroutine in effects.queued)


@pytest.mark.asyncio
async def test_background_failure_does_not_suppress_independent_lifecycle(effects):
    effects.audit.side_effect = RuntimeError("synthetic audit failure")
    result = effects.submit(selected_ids=["selected"])
    completed = await asyncio.gather(*effects.queued, return_exceptions=True)
    assert isinstance(completed[0], RuntimeError)
    effects.lifecycle.assert_awaited_once()
    # The request cannot observe eventual persistence; keep honest submission status.
    assert result["status"] == "submitted"


def test_diagnostics_have_utc_sampling_time_and_no_resource_identifiers(effects):
    before = datetime.now(UTC)
    result = effects.submit(selected_ids=["sensitive-memory-id"])
    after = datetime.now(UTC)
    stamp = datetime.fromisoformat(result["sampled_at"])
    assert stamp.tzinfo is UTC
    assert before <= stamp <= after
    encoded = json.dumps(result)
    for value in ("sensitive-memory-id", "persisted-root-message", "owner-private", "workspace-private"):
        assert value not in encoded
    assert "applied" not in result and "committed" not in result


def test_invalid_count_is_bounded_and_marked_truncated(effects):
    result = effects.submit(selected_ids=[None] * 900)
    assert result["invalid_id_count"] == 501
    assert result["counts_truncated"] is True
