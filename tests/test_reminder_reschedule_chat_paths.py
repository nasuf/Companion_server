"""Chat entry and confirmation paths of versioned reminder rescheduling."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.services.memory.manual_write import MemoryWriteError
from tests.test_reminder_ask_back_flow import _make_preflight_ctx, _make_short_circuit_ctx

CANDIDATES = [{"id": "m1", "content": "提醒我喝水", "source": "user"}]


def _detected(*_args, **_kwargs):
    return {"is_deletion_request": True, "target_description": "喝水提醒",
            "intent": "reschedule", "new_time": "2030-01-01T00:00:00Z", "confidence": 0.9}


async def _drain(events):
    return [e async for e in events] if hasattr(events, "__aiter__") else events


async def test_unresolvable_reschedule_asks_again_without_a_pending_proposal():
    from app.services.chat import intent_handlers as ih

    ctx = _make_short_circuit_ctx()
    ctx.response_diagnostics = {}
    save = AsyncMock()
    with patch.object(ih, "detect_deletion_intent", side_effect=_detected), \
            patch.object(ih, "find_matching_memories", AsyncMock(return_value=CANDIDATES)), \
            patch.object(ih, "save_pending_action", save), \
            patch("app.services.reminder.reschedule.prepare_ticket",
                  AsyncMock(side_effect=MemoryWriteError(422, "reschedule_time_unresolved"))):
        handled, events = await ih.handle_deletion("把喝水提醒改一下", ctx)
        _ = [e async for e in events]
    assert handled
    assert ctx.last_short_circuit_reply.startswith("我还没改动提醒")
    assert ctx.response_diagnostics["reminder_reschedule"] == {
        "status": "needs_clarification", "reason": "reschedule_time_unresolved"}
    save.assert_not_awaited()


async def test_reschedule_proposal_pins_the_ticket_time_not_the_model_time():
    from app.services.chat import intent_handlers as ih

    ctx = _make_short_circuit_ctx()
    ticket = {"new_time": "2030-02-02T07:00:00+00:00", "targets": []}
    save = AsyncMock()
    with patch.object(ih, "detect_deletion_intent", side_effect=_detected), \
            patch.object(ih, "find_matching_memories", AsyncMock(return_value=CANDIDATES)), \
            patch.object(ih, "save_pending_action", save), \
            patch("app.services.reminder.reschedule.prepare_ticket", AsyncMock(return_value=ticket)):
        handled, _ = await ih.handle_deletion("把喝水提醒改到后天下午三点", ctx)
    assert handled
    kwargs = save.await_args.kwargs
    assert kwargs["new_time"] == ticket["new_time"] and kwargs["reschedule_ticket"] is ticket


@pytest.mark.parametrize(
    "outcome,expected_reason,updated",
    [
        (1, "applied", 1),
        (0, "proposal_unavailable", 0),
        (MemoryWriteError(409, "reschedule_target_changed"), "reschedule_target_changed", 0),
        (RuntimeError("db down"), "write_unavailable", 0),
    ],
)
async def test_confirmation_reports_each_reschedule_outcome(outcome, expected_reason, updated):
    from app.services.chat import preflight

    ctx = _make_preflight_ctx()
    ctx.response_diagnostics = {}
    ctx.message_ids = ("confirm",)
    pending = {"action": "reschedule", "candidates": CANDIDATES,
               "new_time": "2030-01-01T00:00:00Z", "reschedule_ticket": {"targets": []}}
    apply = AsyncMock(side_effect=outcome) if isinstance(outcome, Exception) else AsyncMock(return_value=outcome)
    with patch.object(preflight, "load_pending_action", AsyncMock(return_value=pending)), \
            patch.object(preflight, "apply_reschedule", apply), \
            patch.object(preflight, "clear_pending_deletion", AsyncMock()) as clear:
        await _drain(preflight.resolve_pending_deletion("好", ctx))
    assert ctx.response_diagnostics["reminder_reschedule"] == {
        "status": "applied" if updated else "not_applied", "reason": expected_reason, "updated": updated}
    assert ("挪到" in ctx.last_short_circuit_reply) == bool(updated)
    clear.assert_awaited_once()
    assert apply.await_args.kwargs["ticket"] == {"targets": []}
    assert apply.await_args.kwargs["message_ids"] == ("confirm",)


async def test_admin_temporal_repair_maps_contract_and_source_errors():
    from app.api.admin import memory_repairs as api

    admin = {"sub": "admin", "role": "admin"}
    with patch("app.services.memory.temporal_repair.preview",
               AsyncMock(side_effect=MemoryWriteError(404, "memory_not_found_in_scope"))):
        with pytest.raises(HTTPException) as missing:
            await api.preview_temporal_repair("user", "m1", "u", "w", admin)
    assert missing.value.status_code == 404
    payload = api.TemporalRepairRequest(user_id="u", workspace_id="w", evidence_id="e",
                                        expected_version="a" * 64)
    for failure, status, detail in (
        (MemoryWriteError(409, "temporal_repair_version_changed"), 409, "temporal_repair_version_changed"),
        (ValueError("evidence_source_version_changed"), 409, "temporal_repair_source_changed"),
    ):
        with patch("app.services.memory.temporal_repair.apply", AsyncMock(side_effect=failure)):
            with pytest.raises(HTTPException) as error:
                await api.apply_temporal_repair(payload, "user", "m1", admin)
        assert (error.value.status_code, error.value.detail) == (status, detail)
    with patch("app.services.memory.temporal_repair.apply", AsyncMock(return_value={"status": "applied"})) as apply:
        assert await api.apply_temporal_repair(payload, "user", "m1", admin) == {"status": "applied"}
    assert apply.await_args.kwargs["actor"] is admin
