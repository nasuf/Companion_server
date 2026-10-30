"""Recovery of the Actions control plane must never replay a billed request."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from evals.quality import nightly
from app.services.quality import budget, jobs


@pytest.fixture
def http_control(monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.invalid?test=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "synthetic-secret")
    factory = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            nightly.httpx,
            "AsyncClient",
            lambda **kw: factory(transport=httpx.MockTransport(handler), **kw),
        )
        delay = AsyncMock()
        monkeypatch.setattr(nightly.asyncio, "sleep", delay)
        return delay

    return install


@pytest.mark.parametrize(
    "operation", ["renew", "prepare", "result", "baseline", "settle", "uncertain"]
)
@pytest.mark.parametrize("failure", [502, 503, 504, "transport"])
async def test_idempotent_actions_recover_without_changing_payload(
    http_control, operation, failure, capsys
):
    requests = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"value": "synthetic-jwt"})
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if failure == "transport":
                raise httpx.ReadError("private-text synthetic-secret", request=request)
            return httpx.Response(failure, text="private-text synthetic-secret")
        return httpx.Response(200, json={"ok": True})

    delay = http_control(respond)
    control = nightly.Control()
    control.job = {"id": "owned", "lease_token": "x" * 32}
    assert await control.action(operation, {"synthetic": "private-case"}) == {
        "ok": True
    }
    assert len(requests) == 2 and requests[0] == requests[1]
    delay.assert_awaited_once()
    output = capsys.readouterr().out
    assert "quality_control_retry" in output and operation in output
    assert not any(
        secret in output
        for secret in ["synthetic-secret", "synthetic-jwt", "private-case", "x" * 32]
    )


@pytest.mark.parametrize("operation", ["begin", "reserve", "finish", "unknown"])
async def test_non_idempotent_or_unknown_actions_are_never_retried(
    http_control, operation
):
    count = 0

    def respond(request):
        nonlocal count
        if request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        count += 1
        return httpx.Response(502, text="private")

    delay = http_control(respond)
    data = {} if operation == "begin" else {"operation": operation}
    path = "/begin" if operation == "begin" else "/jobs/owned"
    with pytest.raises(RuntimeError):
        await nightly.Control().request(path, data)
    assert count == 1
    delay.assert_not_awaited()


@pytest.mark.parametrize(
    "status,error",
    [
        (400, RuntimeError),
        (401, RuntimeError),
        (403, RuntimeError),
        (404, RuntimeError),
        (409, jobs.LostLease),
        (429, budget.BudgetExhausted),
    ],
)
async def test_terminal_rejections_do_not_retry(http_control, status, error):
    requests = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        requests.append(request)
        return httpx.Response(status, text="private")

    delay = http_control(respond)
    with pytest.raises(error):
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert len(requests) == 1
    delay.assert_not_awaited()


async def test_repeated_failure_stops_at_three_attempts(http_control, capsys):
    count = 0

    def respond(request):
        nonlocal count
        if request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        count += 1
        return httpx.Response(502, text="private")

    delay = http_control(respond)
    with pytest.raises(RuntimeError) as error:
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert count == 3 and delay.await_count == 2
    details = nightly.public_failure(error.value)
    assert details["http_status"] == 502 and details["attempts"] == 3
    assert details["operation"] == "renew" and details["phase"] == "control"
    assert details["status"] == "incomplete" and "private" not in json.dumps(details)


async def test_oidc_temporary_failure_recovers_before_any_post(http_control):
    methods = []

    def respond(request):
        methods.append(request.method)
        if len(methods) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"value": "jwt"} if request.method == "GET" else {"ok": True}
        )

    http_control(respond)
    assert await nightly.Control().request("/jobs/owned", {"operation": "renew"}) == {
        "ok": True
    }
    assert methods == ["GET", "GET", "POST"]


@pytest.mark.parametrize("phase", ["oidc", "control"])
@pytest.mark.parametrize("response", [b"private-not-json", b"[]", b"null"])
async def test_invalid_response_is_terminal_and_private(http_control, phase, response):
    methods = []

    def respond(request):
        methods.append(request.method)
        if phase == "control" and request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        return httpx.Response(200, content=response)

    delay = http_control(respond)
    with pytest.raises(nightly.ControlRequestError) as error:
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert error.value.phase == phase
    assert error.value.code == "invalid_response"
    assert "private" not in json.dumps(nightly.public_failure(error.value))
    assert methods == (["GET"] if phase == "oidc" else ["GET", "POST"])
    delay.assert_not_awaited()


async def test_total_deadline_cancels_inflight_request(http_control, monkeypatch):
    cancelled = asyncio.Event()

    async def respond(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    delay = http_control(respond)
    monkeypatch.setattr(nightly, "RETRY_WINDOW_SECONDS", 0.02)
    with pytest.raises(nightly.ControlRequestError) as error:
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert error.value.code == "deadline_exceeded"
    assert error.value.attempts == 1 and cancelled.is_set()
    delay.assert_not_awaited()


@pytest.mark.parametrize("during_backoff", [False, True])
async def test_parent_cancellation_is_not_retried(
    http_control, monkeypatch, during_backoff
):
    blocked = asyncio.Event()
    count = 0

    async def respond(request):
        nonlocal count
        count += 1
        if during_backoff:
            return httpx.Response(502)
        blocked.set()
        await asyncio.Event().wait()

    async def sleep(_):
        blocked.set()
        await asyncio.Event().wait()

    http_control(respond)
    if during_backoff:
        monkeypatch.setattr(nightly.asyncio, "sleep", sleep)
    task = asyncio.create_task(
        nightly.Control().request("/jobs/owned", {"operation": "renew"})
    )
    await asyncio.wait_for(blocked.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert count == 1


async def test_recovery_cannot_override_confirmed_lease_loss(http_control):
    count = 0

    def respond(request):
        nonlocal count
        if request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        count += 1
        return httpx.Response(502 if count == 1 else 409)

    delay = http_control(respond)
    with pytest.raises(jobs.LostLease):
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert count == 2
    delay.assert_awaited_once()


async def test_deadline_includes_retry_backoff(http_control, monkeypatch, capsys):
    real_sleep = asyncio.sleep
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503)

    http_control(respond)
    monkeypatch.setattr(nightly.asyncio, "sleep", real_sleep)
    monkeypatch.setattr(nightly, "RETRY_WINDOW_SECONDS", 0.02)
    with pytest.raises(nightly.ControlRequestError) as error:
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert error.value.code == "deadline_exceeded"
    assert len(requests) == 1
    assert (
        json.loads(capsys.readouterr().out.splitlines()[-1])["event"]
        == "quality_control_failure"
    )


@pytest.mark.parametrize("token", [None, "", 1, []])
async def test_invalid_oidc_token_never_reaches_control(http_control, token):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"value": token})

    delay = http_control(respond)
    with pytest.raises(nightly.ControlRequestError):
        await nightly.Control().request("/jobs/owned", {"operation": "renew"})
    assert len(requests) == 1 and requests[0].method == "GET"
    delay.assert_not_awaited()


async def test_ambiguous_reserve_logs_metadata_even_when_caller_catches_error(
    http_control, capsys
):
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"value": "jwt"})
        raise httpx.ReadError("private provider body", request=request)

    http_control(respond)
    with pytest.raises(nightly.ControlRequestError):
        await nightly.Control().request(
            "/jobs/owned", {"operation": "reserve", "data": {"private": "prompt"}}
        )
    event = json.loads(capsys.readouterr().out)
    assert event == {
        "event": "quality_control_failure",
        "status": "incomplete",
        "error_type": "ControlRequestError",
        "phase": "control",
        "operation": "reserve",
        "error_code": "transport_error",
        "http_status": None,
        "attempts": 1,
    }
