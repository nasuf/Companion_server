"""Location ingress, recovery isolation, and bounded administrator diagnostics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from app.api.admin import runtime_jobs
from app.api.realtime import ws
from app.services import location_memory as loc, location_ingress as ingress
from app.services.auth import create_jwt
from tests.test_location_component_card import _location_card


@pytest.mark.parametrize(
    "value", [True, False, float("nan"), float("inf"), -float("inf"), 91, None]
)
def test_unusable_location_is_rejected(value):
    assert ws._sanitize_component_card(_location_card(latitude=value)) is None
    with pytest.raises(ValueError):
        ingress.location_request(
            text="",
            card=_location_card(latitude=value),
            client_id="draft",
            attachment_ids=[],
        )


def test_old_location_sentence_is_rendered_as_historical():
    from app.services.offerings_memory_text import render_component_card_line

    card = _location_card()
    old = "用户分享了当前位置：上海市黄浦区中山东一路，上海市，黄浦区（约 31.23040, 121.47370）"
    result = render_component_card_line(old, card)
    assert "当前位置" not in result
    assert result.count("31.23040") == 1
    assert "不证明现在的位置" in result
    assert render_component_card_line(result, card) == result


async def test_location_card_with_url_cannot_be_replaced_by_link():
    card = _location_card()
    assert await ws._ensure_link_card_for_turn(
        user_id="u",
        conversation_id="c",
        text="https://example.com",
        component_card=card,
    ) == (card, None)


@pytest.mark.parametrize("outcome", ["duplicate", "conflict", "unavailable", "invalid"])
async def test_location_retry_errors_are_scoped_and_do_not_close_socket(
    monkeypatch, outcome
):
    from app.services.admin_takeover import service

    monkeypatch.setattr(
        service, "is_conversation_taken_over", AsyncMock(return_value=False)
    )
    lookup = AsyncMock(return_value="saved")
    if outcome == "conflict":
        lookup.side_effect = ingress.ChatRequestConflict()
    elif outcome == "unavailable":
        lookup.side_effect = RuntimeError("private SQL contents")
    monkeypatch.setattr(ingress, "previous_location_message", lookup)
    quota = AsyncMock()
    monkeypatch.setattr(ws.chat_quota, "consume_one", quota)
    socket = SimpleNamespace(send_json=AsyncMock())
    card = _location_card(latitude=None) if outcome == "invalid" else _location_card()
    await ws._handle_message(
        socket,
        "c",
        "u",
        SimpleNamespace(id="a"),
        "",
        client_id="draft",
        component_card=card,
    )
    event = socket.send_json.call_args.args[0]
    assert event["type"] == ("ack" if outcome == "duplicate" else "error")
    assert event["data"]["client_id"] == "draft"
    assert "private" not in str(event)
    quota.assert_not_awaited()


async def test_dispatch_failure_does_not_starve_other_runtime_jobs(monkeypatch):
    from jobs import scheduler

    async def local(name, body):
        await body()

    monkeypatch.setattr(scheduler, "_run_local_job", local)
    processed = AsyncMock(return_value=1)
    monkeypatch.setattr(scheduler, "process_runtime_jobs", processed)
    monkeypatch.setattr(
        loc,
        "dispatch_location_events",
        AsyncMock(side_effect=RuntimeError("synthetic")),
    )
    failed = Mock()
    monkeypatch.setattr(scheduler, "_job_failed", failed)
    await scheduler._run_runtime_job_queue()
    processed.assert_awaited_once_with(max_jobs=20)
    assert failed.call_args.args[0] == "Location memory dispatch"


async def test_admin_requires_admin_token_and_bounded_query(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(
        settings, "jwt_secret", "synthetic-location-contract-secret-123"
    )
    app = FastAPI()
    app.include_router(runtime_jobs.router)
    read = AsyncMock(return_value={"items": [], "sampled_at": "now", "limit": 20})
    monkeypatch.setattr(loc, "inspect_location_events", read)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/admin-api/runtime-jobs/location-events"
        for token in (None, create_jwt("user", "user")):
            response = await client.get(
                path, headers={"Authorization": "Bearer " + token} if token else {}
            )
            assert response.status_code in (401, 403)
        read.assert_not_awaited()
        headers = {"Authorization": "Bearer " + create_jwt("admin", "admin")}
        assert (
            await client.get(path + "?limit=51", headers=headers)
        ).status_code == 422
        assert (await client.get(path, headers=headers)).status_code == 200
        read.assert_awaited_once_with(limit=20)
        read.side_effect = RuntimeError("private SQL contents")
        response = await client.get(path, headers=headers)
        assert response.status_code == 503 and "private" not in response.text


async def test_location_diagnostics_do_not_report_missing_or_failed_jobs_as_success(
    monkeypatch,
):
    from app.services.runtime import job_queue

    rows = [
        {
            "message_id": str(i),
            "status": status,
            "job_id": job,
            "dispatch_attempts": 1,
            "last_error": None,
            "created_at": "now",
            "updated_at": "now",
        }
        for i, (status, job) in enumerate(
            [
                ("completed", "done"),
                ("cancelled", "cancelled"),
                ("pending_dispatch", None),
                ("queued", "dead"),
                ("queued", "missing"),
                ("queued", "unreachable"),
                ("queued", "not-retried"),
                ("queued", "more"),
            ]
        )
    ]
    database = SimpleNamespace(query_raw=AsyncMock(return_value=rows))
    monkeypatch.setattr(loc, "db", database)
    inspect = AsyncMock(
        side_effect=[{"status": "dead_letter"}, None, ConnectionError("private")]
    )
    monkeypatch.setattr(job_queue, "inspect_runtime_job", inspect)
    result = await loc.inspect_location_events(limit=7)
    assert [r["queue_state"] for r in result["items"]] == [
        "not_needed",
        "not_needed",
        "not_dispatched",
        "dead_letter",
        "missing_or_expired",
        "unavailable",
        "unavailable",
    ]
    assert result["has_more"] and result["denominator"] == "recent_sample_only"
    assert database.query_raw.call_args.args[-1] == 8 and inspect.await_count == 3
    with pytest.raises(ValueError):
        await loc.inspect_location_events(limit=51)
    database.query_raw.return_value = []
    assert (await loc.inspect_location_events())["items"] == []


@pytest.mark.parametrize(
    "outcome", ["created", "concurrent_duplicate", "quota", "database_error"]
)
async def test_location_ws_ack_and_reply_follow_committed_admission(
    monkeypatch, outcome
):
    from app.services.admin_takeover import service
    from app.services.interaction.user_turn_aggregation import (
        UserMessageAggregationPlan,
    )

    monkeypatch.setattr(
        service, "is_conversation_taken_over", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        ingress, "previous_location_message", AsyncMock(return_value=None)
    )
    admit = AsyncMock(return_value=("saved", outcome == "created"))
    if outcome == "quota":
        admit.side_effect = ingress.ChatIngressQuotaBlocked(
            {"allowed": False, "reason": "paid_confirm"}
        )
    if outcome == "database_error":
        admit.side_effect = RuntimeError("private")
    monkeypatch.setattr(ingress, "accept_location_message", admit)
    quota = AsyncMock()
    monkeypatch.setattr(ws.chat_quota, "consume_one", quota)
    legacy = AsyncMock()
    monkeypatch.setattr(ws, "_persist_user_message", legacy)
    monkeypatch.setattr(
        ws.chat_media_repo, "get_message_attachments", AsyncMock(return_value=[])
    )
    bind = AsyncMock()
    monkeypatch.setattr(ws.chat_media_repo, "bind_attachments_to_message", bind)
    monkeypatch.setattr(ws, "ensure_vision_summaries", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        ws, "get_cached_schedule", AsyncMock(return_value={"synthetic": True})
    )
    monkeypatch.setattr(ws, "get_current_status", lambda _: {"status": "idle"})
    monkeypatch.setattr(ws, "build_reply_timing_context", AsyncMock(return_value={}))
    plan = UserMessageAggregationPlan(
        route="immediate",
        agent_id="a",
        user_id="u",
        conversation_id="c",
        text="",
        metadata={},
        final_message="",
        final_context={},
        fallback_message="",
        fallback_context={},
    )
    monkeypatch.setattr(
        ws, "plan_user_message_aggregation", AsyncMock(return_value=plan)
    )
    monkeypatch.setattr(ws, "mark_user_replied_for_conversation", AsyncMock())
    # Only outbound delivery/model and unrelated asynchronous side effects are replaced.
    monkeypatch.setattr(ws, "fire_background", lambda coroutine: coroutine.close())
    reply = AsyncMock()
    monkeypatch.setattr(ws, "_queue_reply_or_error", reply)
    socket = SimpleNamespace(send_json=AsyncMock())
    await ws._handle_message(
        socket,
        "c",
        "u",
        SimpleNamespace(id="a", name="Agent"),
        "",
        workspace_id="w",
        client_id="draft",
        component_card=_location_card(),
    )
    event = socket.send_json.call_args.args[0]
    assert (
        event["type"]
        == {
            "created": "ack",
            "concurrent_duplicate": "ack",
            "quota": "quota_blocked",
            "database_error": "error",
        }[outcome]
    )
    assert event["data"]["client_id"] == "draft"
    quota.assert_not_awaited()
    legacy.assert_not_awaited()
    bind.assert_not_awaited()
    assert reply.await_count == (1 if outcome == "created" else 0)
    if outcome == "created":
        assert reply.call_args.kwargs["reply_context"]["location_share"] is True
        assert "不能用作你的所在地" in reply.call_args.kwargs["user_message"]
