"""Shared chat submission against real, migrated, synthetic PostgreSQL.

Only the historical access audit is mocked. These tests await the submitted
lifecycle coroutines and inspect persisted scores and receipts; submission
diagnostics alone are deliberately not treated as persistence evidence.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.chat import memory_usage as usage
from tests.test_memory_lifecycle_postgres import event, memory, memory_flow, row
from tests.test_runtime_execution_foundation import flow


@pytest.fixture(autouse=True)
def access_audit(monkeypatch):
    audit = AsyncMock(return_value=None)
    monkeypatch.setattr(usage, "log_memory_access", audit)
    return audit


def submit(fixture, message, queue, *, selected=(), candidates=(), **overrides):
    return usage.submit_memory_usage(**{
        "event_id": message.id,
        "user_id": fixture.ids["owner"],
        "workspace_id": fixture.ids["workspace"],
        "selected_ids": selected,
        "candidate_ids": candidates,
        "fire_background": queue.append,
        **overrides,
    })


async def receipts(fixture, message):
    return await fixture.db.query_raw(
        "SELECT memory_side, memory_id, contributed, reward "
        "FROM memory_usage_receipts WHERE event_id=$1 ORDER BY memory_side, memory_id",
        message.id,
    )


@pytest.mark.parametrize("side", ["user", "ai"])
@pytest.mark.parametrize("legacy_scope", [False, True], ids=["workspace", "null-workspace"])
async def test_root_fragments_and_replay_share_one_persisted_reward(
    memory_flow, access_audit, side, legacy_scope
):
    f = memory_flow
    workspace = None if legacy_scope else f.ids["workspace"]
    conversation = f.ids["other_conversation" if legacy_scope else "conversation"]
    stored = await memory(f, side, workspaceId=workspace)
    root_message = await event(f, conversationId=conversation)
    queued = []

    # Root, a sub-intent, and replay all retain the persisted root event ID.
    diagnostics = [
        submit(
            f, root_message, queued,
            selected=[stored.id, stored.id], candidates=[stored.id],
            workspace_id=workspace,
        )
        for _ in range(3)
    ]
    assert all(item["status"] == "submitted" for item in diagnostics)
    assert all(item["selected_count"] == 1 for item in diagnostics)
    assert all(item["candidate_only_count"] == 0 for item in diagnostics)
    assert (await row(f, stored.id, side))["current_score"] == 0.5

    completed = await asyncio.gather(*queued)
    assert sorted(value for value in completed if value is not None) == [0, 0, 1]
    assert (await row(f, stored.id, side))["current_score"] == pytest.approx(0.62, abs=1e-5)
    assert await receipts(f, root_message) == [{
        "memory_side": side, "memory_id": stored.id,
        "contributed": True, "reward": pytest.approx(0.12),
    }]
    assert await f.db.message.count(where={"conversationId": conversation}) == 1
    assert access_audit.await_count == 3  # Audit idempotency is a separate contract.


@pytest.mark.parametrize("side", ["user", "ai"])
async def test_candidate_then_selected_upgrades_existing_receipt_without_double_credit(
    memory_flow, access_audit, side
):
    f = memory_flow
    stored = await memory(f, side)
    message = await event(f)
    queued = []
    candidate = submit(f, message, queued, candidates=[stored.id])
    assert candidate["audit_submission"] == "not_requested"
    assert await asyncio.gather(*queued) == [1]
    access_audit.assert_not_called()
    assert (await row(f, stored.id, side))["current_score"] == pytest.approx(0.514, abs=1e-5)
    assert await receipts(f, message) == [{
        "memory_side": side, "memory_id": stored.id,
        "contributed": False, "reward": pytest.approx(0.014, abs=1e-5),
    }]

    queued = []
    submit(f, message, queued, selected=[stored.id], candidates=[stored.id])
    assert await asyncio.gather(*queued) == [None, 1]
    assert (await row(f, stored.id, side))["current_score"] == pytest.approx(0.62, abs=1e-5)
    assert await receipts(f, message) == [{
        "memory_side": side, "memory_id": stored.id,
        "contributed": True, "reward": pytest.approx(0.12),
    }]

    queued = []
    submit(f, message, queued, candidates=[stored.id])
    submit(f, message, queued, selected=[stored.id])
    assert await asyncio.gather(*queued) == [0, None, 0]
    assert (await row(f, stored.id, side))["current_score"] == pytest.approx(0.62, abs=1e-5)
    assert len(await receipts(f, message)) == 1


async def test_one_event_updates_both_sides_and_new_event_receives_new_credit(memory_flow):
    f = memory_flow
    stored = {side: await memory(f, side) for side in ("user", "ai")}
    messages = [await event(f), await event(f)]
    for message in messages:
        queued = []
        submit(f, message, queued, selected=[item.id for item in stored.values()])
        assert await asyncio.gather(*queued) == [None, 2]
        persisted = await receipts(f, message)
        assert {(item["memory_side"], item["memory_id"]) for item in persisted} == {
            (side, item.id) for side, item in stored.items()
        }
    for side, item in stored.items():
        assert (await row(f, item.id, side))["current_score"] == pytest.approx(0.74, abs=1e-5)


@pytest.mark.parametrize("side", ["user", "ai"])
@pytest.mark.parametrize("invalid", [
    "wrong_owner", "memory_scope", "event_scope", "expired_event", "assistant_event",
])
async def test_submission_does_not_bypass_storage_identity_or_event_guards(
    memory_flow, side, invalid
):
    f = memory_flow
    memory_changes = {"workspaceId": None} if invalid == "memory_scope" else {}
    stored = await memory(f, side, **memory_changes)
    event_changes = {}
    if invalid == "event_scope":
        event_changes["conversationId"] = f.ids["other_conversation"]
    elif invalid == "expired_event":
        event_changes["createdAt"] = datetime.now(UTC) - timedelta(days=8)
    elif invalid == "assistant_event":
        event_changes["role"] = "assistant"
    message = await event(f, **event_changes)
    scope_changes = {"user_id": str(uuid4())} if invalid == "wrong_owner" else {}
    queued = []
    diagnostics = submit(f, message, queued, selected=[stored.id], **scope_changes)

    # Syntactically valid IDs can be submitted, but only storage can authorize
    # them. A zero-row result must not turn the diagnostic into "applied".
    assert diagnostics["status"] == "submitted"
    assert await asyncio.gather(*queued) == [None, 0]
    assert diagnostics["status"] == "submitted"
    assert (await row(f, stored.id, side))["current_score"] == 0.5
    assert await receipts(f, message) == []
