"""Availability state machine and real resilience boundary; no external calls."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from app.services.ops import provider_failing as f, service_health as h
from app.services.llm import resilience as r


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(f, "_pending", [])
    monkeypatch.setattr(f, "_batch", None)
    monkeypatch.setattr(f, "_lock", None)
    monkeypatch.setattr(f, "_lost", 0)
    r.reset_breakers_for_testing()
    r.reset_slots_for_testing()


def event(t, kind="failure", role="chat", model="m", **more):
    return {
        **dict(
            route=f"ark/{model}/{role}", request=str(t), kind=kind, started=t - 1, at=t
        ),
        **more,
    }


def reduce(events, state=None, now=1000, **config):
    return f.reduce_snapshot(state or {}, events, now, {**f.policy(), **config})


def test_exact_threshold_denominator_and_role_isolation():
    assert not reduce([event(990 + i) for i in range(4)])["routes"]["ark/m/chat"].get(
        "incident_at"
    )
    yes = reduce([event(990 + i) for i in range(5)])["routes"]["ark/m/chat"]
    assert yes["incident_at"] == 994
    both = [event(980 + i, "success", started=1) for i in range(6)] + [
        event(990 + i) for i in range(5)
    ]
    assert not reduce(both)["routes"]["ark/m/chat"].get("incident_at")
    assert reduce(both, failure_ratio_threshold=0.45)["routes"]["ark/m/chat"][
        "incident_at"
    ]
    separate = reduce(
        [event(990 + i, role="chat" if i % 2 else "utility") for i in range(6)]
    )
    assert all(not x.get("incident_at") for x in separate["routes"].values())


def test_empty_window_does_not_recover_and_old_inflight_success_does_not():
    s = reduce([event(990 + i) for i in range(5)])
    assert reduce([], s, now=2000)["routes"]["ark/m/chat"]["incident_at"]
    assert reduce([event(2001, "success", started=990)], s, now=2001)["routes"][
        "ark/m/chat"
    ]["incident_at"]
    recovered = reduce([event(2002, "success", started=1999)], s, now=2002)
    assert "incident_at" not in recovered["routes"]["ark/m/chat"]


def test_circuit_triggers_immediately_but_billing_failure_is_separate():
    assert reduce([event(999, "circuit")])["routes"]["ark/m/chat"]["incident_at"]
    assert reduce([event(999, circuit=True)])["routes"]["ark/m/chat"]["incident_at"]
    assert not reduce([event(990 + i, "billing", circuit=True) for i in range(6)])[
        "routes"
    ]["ark/m/chat"].get("incident_at")


def test_cross_worker_late_events_and_same_model_different_route_never_clear():
    s = reduce([event(990 + i) for i in range(5)])
    s = reduce([event(999, "success", model="backup")], s)
    assert s["routes"]["ark/m/chat"]["incident_at"]
    s = reduce([event(1000, "success", started=998)], s)
    s = reduce([event(995)], s)
    assert not s["routes"]["ark/m/chat"].get("incident_at")
    s = reduce([event(1002, "circuit")], s, now=1002)
    assert s["routes"]["ark/m/chat"]["incident_at"] == 1002


def test_fallback_evidence_is_for_exact_failure_and_never_recovers_primary():
    s = reduce([event(990 + i) for i in range(5)])
    s = reduce([event(996, "fallback", request="994", outcome="succeeded")], s)
    assert s["routes"]["ark/m/chat"]["fallback"] == "succeeded"
    assert s["routes"]["ark/m/chat"]["incident_at"]
    s = reduce([event(997, "fallback", request="993", outcome="failed")], s)
    assert s["routes"]["ark/m/chat"]["fallback"] == "succeeded"


def test_bounded_routes_counters_and_validated_identifiers(monkeypatch):
    s = reduce([event(999), event(999, model="other")], route_limit=1)
    assert s["incomplete"] and len(s["routes"]) == 1
    monkeypatch.setattr(f.settings, "provider_failing_buffer_limit", 1)
    f.Call("ark", "ark/m", "utility").record("failure", 1)
    f.Call("ark", "m", "utility").record("success", 2)
    assert f._lost == 1 and len(f._pending) == 1
    for args in [
        ("bad/route", "m", "chat"),
        ("ark", "SECRET\n", "chat"),
        ("ark", "m", "unknown"),
    ]:
        call = f.Call(*args)
        call.record("failure", 1)
        call.fallback("failed")
    assert f._lost == 4
    assert not reduce([], now=1000)["routes"]


async def test_hot_path_has_no_database_and_excludes_cancellation(monkeypatch):
    monkeypatch.setattr(f, "db", None)
    profile = r.CallProfile(timeout_s=1, max_retries=0, retry_backoff_s=())
    assert (
        await r.call_with_resilience(
            AsyncMock(return_value="ok"),
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
            role="utility",
        )
        == "ok"
    )
    assert [x["kind"] for x in f._pending] == ["success"]
    assert f._pending[0]["route"] == "ark/m/utility"
    with pytest.raises(asyncio.CancelledError):
        await r.call_with_resilience(
            AsyncMock(side_effect=asyncio.CancelledError),
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
        )
    assert len(f._pending) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "backup", ["succeeded", "failed", "not_attempted", "cancelled"]
)
async def test_real_resilience_reports_backup_outcome(stream, backup):
    profile = r.CallProfile(
        timeout_s=0.05, max_retries=0, retry_backoff_s=(), first_chunk_timeout_s=0.05
    )

    async def fail():
        raise RuntimeError("SECRET ERROR TEXT")

    async def good():
        return "ok"

    async def cancel():
        raise asyncio.CancelledError()

    async def fail_stream():
        raise RuntimeError("SECRET ERROR TEXT")
        yield

    async def good_stream():
        yield "ok"

    async def cancel_stream():
        raise asyncio.CancelledError()
        yield

    kwargs = dict(
        primary_provider="ark",
        primary_model_name="m",
        fallback_provider="deepseek",
        fallback_model_name="backup",
        profile=profile,
        op="test",
        role="utility",
    )
    backups = {
        "succeeded": good,
        "failed": fail,
        "not_attempted": None,
        "cancelled": cancel,
    }
    streams = {
        "succeeded": good_stream,
        "failed": fail_stream,
        "not_attempted": None,
        "cancelled": cancel_stream,
    }

    async def run():
        if stream:
            return "".join(
                [
                    x
                    async for x in r.astream_with_resilience(
                        fail_stream, fallback_factory=streams[backup], **kwargs
                    )
                ]
            )
        return await r.call_with_resilience(
            fail, fallback_factory=backups[backup], **kwargs
        )

    if backup == "succeeded":
        assert await run() == "ok"
    else:
        with pytest.raises(
            asyncio.CancelledError if backup == "cancelled" else r.LLMFailedError
        ):
            await run()
    outcomes = [
        x
        for x in f._pending
        if x["kind"] == "fallback" and x["route"] == "ark/m/utility"
    ]
    assert outcomes[-1]["outcome"] == backup
    assert "SECRET" not in repr(f._pending)
    assert f._pending[0]["kind"] == "failure"


async def test_timeouts_retry_denominator_and_circuit():
    profile = r.CallProfile(timeout_s=0.002, max_retries=1, retry_backoff_s=(0,))

    async def timeout():
        await asyncio.sleep(0.05)

    with pytest.raises(r.LLMFailedError):
        await r.call_with_resilience(
            timeout,
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
        )
    assert [x["kind"] for x in f._pending].count("failure") == 2
    r._get_breaker("ark")._opened_at = r.time.monotonic()
    with pytest.raises(r.LLMCircuitOpenError):
        await r.call_with_resilience(
            timeout,
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
        )
    assert f._pending[-2]["kind"] == "circuit"


async def test_partial_stream_and_cancel_never_count_complete_success():
    profile = r.CallProfile(timeout_s=1, max_retries=0, retry_backoff_s=())

    async def midfail():
        yield "first"
        raise RuntimeError("private")

    with pytest.raises(r.LLMFailedError):
        [
            x
            async for x in r.astream_with_resilience(
                midfail,
                primary_provider="ark",
                primary_model_name="m",
                profile=profile,
                op="test",
            )
        ]
    assert not any(x["kind"] == "success" for x in f._pending)
    assert f._pending[-1]["outcome"] == "not_attempted"
    f._pending.clear()

    async def forever():
        yield "first"
        await asyncio.sleep(10)

    stream = r.astream_with_resilience(
        forever,
        primary_provider="ark",
        primary_model_name="m",
        profile=profile,
        op="test",
    )
    assert await anext(stream) == "first"
    await stream.aclose()
    assert not f._pending


def test_notification_same_incident_dedup_and_backup_failure_escalation():
    def sample(outcome):
        return {
            "checks": {
                "provider_failing": {
                    "status": "measured",
                    "alerting": True,
                    "window_seconds": 600,
                    "incidents": [
                        {
                            "provider": "ark",
                            "model": "m",
                            "role": "chat",
                            "first_failure_at": "2026-10-28T00:00Z",
                            "calls": 10,
                            "failures": 5,
                            "fallback": outcome,
                        }
                    ],
                }
            }
        }

    good = sample("succeeded")
    notify, state = h.decide("warning", good, {}, 0)
    assert notify
    assert not h.decide("warning", good, state, 0)[0]
    assert "已由备用模型接住" in h.message_for("warning", good)[1]
    bad = sample("failed")
    assert h.decide("warning", bad, state, 0)[0]
    assert "备用也失败" in h.message_for("warning", bad)[1]


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("failed", [True, False])
async def test_kill_switch_observes_without_changing_result(
    monkeypatch, stream, failed
):
    monkeypatch.setattr(r.settings, "llm_resilience_enabled", False)
    profile = r.CallProfile(timeout_s=1, max_retries=0, retry_backoff_s=())

    async def unary():
        if failed:
            raise ValueError("synthetic")
        return "ok"

    async def source():
        yield await unary()

    async def run():
        if stream:
            return [
                x
                async for x in r.astream_with_resilience(
                    source,
                    primary_provider="ark",
                    primary_model_name="m",
                    profile=profile,
                    op="test",
                )
            ]
        return await r.call_with_resilience(
            unary,
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
        )

    if failed:
        with pytest.raises(ValueError):
            await run()
    else:
        assert await run()
    assert f._pending[0]["kind"] == ("failure" if failed else "success")


@pytest.mark.parametrize("phase", ["first", "idle", "overall", "circuit"])
async def test_stream_budget_failures_and_circuit_are_observed(phase):
    profile = r.CallProfile(
        timeout_s=0.01,
        max_retries=0,
        retry_backoff_s=(),
        first_chunk_timeout_s=0.002,
        idle_timeout_s=0.002 if phase == "idle" else 1,
    )

    async def source():
        if phase == "first":
            await asyncio.sleep(0.02)
        yield "one"
        await asyncio.sleep(0.02)
        yield "two"

    if phase == "circuit":
        r._get_breaker("ark")._opened_at = r.time.monotonic()
    with pytest.raises(r.LLMFailedError):
        await r.collect_stream(
            source,
            primary_provider="ark",
            primary_model_name="m",
            profile=profile,
            op="test",
        )
    assert f._pending[0]["kind"] == ("circuit" if phase == "circuit" else "failure")
    assert not any(x["kind"] == "success" for x in f._pending)


def test_billing_error_records_only_safe_category():
    class Billing(Exception):
        code = "AccountOverdueError"

    f.Call("ark", "m").record("failure", 1, error=Billing("PRIVATE"))
    assert f._pending[0]["kind"] == "billing" and "PRIVATE" not in repr(f._pending)
