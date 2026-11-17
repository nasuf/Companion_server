"""Q01.05: a caller budget bounds retries, backoff, fallback and streams.

Cutting an attempt short because the caller ran out of time is not a provider
failure, so the circuit breaker must not count it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.config import settings
from app.services.llm import resilience
from app.services.llm.resilience import CallProfile, LLMDeadlineExceeded, llm_deadline
# Warm the lazily imported observers so import time is not charged to a budget.
import app.services.ops.provider_budget  # noqa: F401,E402
import app.services.ops.provider_failing  # noqa: F401,E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "web_concurrency", 1)
    monkeypatch.setattr(settings, "llm_resilience_enabled", True)
    resilience.reset_slots_for_testing()
    resilience.reset_breakers_for_testing()
    yield
    resilience.reset_slots_for_testing()
    resilience.reset_breakers_for_testing()


def _profile(**kw):
    base = dict(timeout_s=5, max_retries=2, retry_backoff_s=(0.2, 0.2), allow_ollama_fallback=True,
                first_chunk_timeout_s=5, idle_timeout_s=5)
    base.update(kw)
    return CallProfile(**base)


def test_nested_deadline_only_tightens():
    with llm_deadline(10):
        outer = resilience._deadline.get()
        with llm_deadline(60):
            assert resilience._deadline.get() == outer
        with llm_deadline(1):
            assert resilience._deadline.get() < outer
    assert resilience._deadline.get() is None


async def test_attempt_cut_by_deadline_skips_fallback_and_spares_breaker():
    calls = []

    async def slow():
        calls.append("primary")
        await asyncio.sleep(5)

    async def fallback():
        calls.append("fallback")
        return "x"

    started = time.monotonic()
    with llm_deadline(0.8), pytest.raises(LLMDeadlineExceeded):
        await resilience.call_with_resilience(
            slow, primary_provider="ark", profile=_profile(), op="t", fallback_factory=fallback,
            fallback_provider="dashscope",
        )
    assert time.monotonic() - started < 1.5
    assert calls == ["primary"]
    assert resilience._get_breaker("ark").state() == "closed"
    assert not resilience._get_breaker("ark")._failures


async def test_no_retry_when_budget_cannot_cover_backoff():
    calls = []

    async def failing():
        calls.append(1)
        raise RuntimeError("provider 500")

    with llm_deadline(0.6), pytest.raises(resilience.LLMFailedError) as info:
        await resilience._run_with_retry(failing, provider="ark", profile=_profile(), op="t")
    assert len(calls) == 1 and not isinstance(info.value, LLMDeadlineExceeded)


async def test_exhausted_budget_fails_before_calling():
    async def never():
        raise AssertionError("must not be called")

    with llm_deadline(0.1), pytest.raises(LLMDeadlineExceeded):
        await resilience._run_with_retry(never, provider="ark", profile=_profile(), op="t")


class _Stream:
    def __init__(self, delays):
        self.delays = list(delays)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.delays:
            raise StopAsyncIteration
        await asyncio.sleep(self.delays.pop(0))
        return "a"

    async def aclose(self):
        self.closed = True


async def _drain(gen):
    return [t async for t in gen]


@pytest.mark.parametrize("delays", [[5], [0, 5]], ids=["before_first_chunk", "mid_stream"])
async def test_stream_deadline_closes_connection_and_spares_breaker(delays):
    stream = _Stream(delays)
    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await _drain(resilience.astream_with_resilience(
            lambda: stream, primary_provider="ark", profile=_profile(), op="t",
            fallback_factory=lambda: _Stream([0]), fallback_provider="dashscope",
        ))
    assert stream.closed
    assert not resilience._get_breaker("ark")._failures
    assert not resilience._slots.get("ark") or not resilience._slots["ark"].locked()


async def test_killswitch_stream_times_out_and_closes(monkeypatch):
    monkeypatch.setattr(settings, "llm_resilience_enabled", False)
    stalled = _Stream([0, 5])
    with pytest.raises(resilience.LLMFailedError):
        await _drain(resilience.astream_with_resilience(
            lambda: stalled, primary_provider="ark", profile=_profile(idle_timeout_s=0.2), op="t",
        ))
    assert stalled.closed
    ok = _Stream([0, 0])
    assert await _drain(resilience.astream_with_resilience(
        lambda: ok, primary_provider="ark", profile=_profile(), op="t")) == ["a", "a"]
    assert ok.closed


async def test_capped_attempt_without_observer_and_killswitch_unary(monkeypatch):
    async def slow():
        await asyncio.sleep(5)

    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await resilience._run_with_retry(slow, provider="ark", profile=_profile(), op="t")
    assert not resilience._get_breaker("ark")._failures

    monkeypatch.setattr(settings, "llm_resilience_enabled", False)
    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await resilience._run_with_retry(slow, provider="ark", profile=_profile(), op="t")

    async def quick():
        return "ok"

    assert await resilience._run_with_retry(quick, provider="ark", profile=_profile(), op="t") == "ok"


class _Broken(_Stream):
    async def __anext__(self):
        raise RuntimeError("upstream reset")


async def test_killswitch_stream_empty_overlong_and_broken(monkeypatch):
    monkeypatch.setattr(settings, "llm_resilience_enabled", False)
    empty = _Stream([])
    assert await _drain(resilience.astream_with_resilience(
        lambda: empty, primary_provider="ark", profile=_profile(), op="t")) == []
    assert empty.closed

    overlong = _Stream([0, 0.3, 0])
    with pytest.raises(resilience.LLMFailedError, match="overall timeout"):
        await _drain(resilience.astream_with_resilience(
            lambda: overlong, primary_provider="ark", profile=_profile(timeout_s=0.2), op="t"))
    assert overlong.closed

    broken = _Broken([])
    with pytest.raises(RuntimeError):
        await _drain(resilience._stream_provider(lambda: broken, provider="ark", profile=_profile(), op="t"))
    assert broken.closed


def _half_open(provider="ark"):
    breaker = resilience._get_breaker(provider)
    breaker._opened_at = time.monotonic() - breaker._cooldown_sec - 1
    assert breaker.state() == "half_open"
    return breaker


async def test_probe_cut_by_deadline_or_cancel_is_released():
    async def slow():
        await asyncio.sleep(5)

    breaker = _half_open()
    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await resilience._run_with_retry(slow, provider="ark", profile=_profile(), op="t")
    assert breaker.state() == "half_open" and breaker.try_acquire()  # probe allowed again
    breaker.release_probe()

    task = asyncio.create_task(resilience._run_with_retry(slow, provider="ark", profile=_profile(), op="t"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert breaker.try_acquire()
    breaker.release_probe()

    stream = _Stream([5])
    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await _drain(resilience._stream_provider(lambda: stream, provider="ark", profile=_profile(), op="t"))
    assert breaker.try_acquire() and not breaker._failures


async def test_stream_probe_released_on_cancel_and_abandon():
    breaker = _half_open()
    waiting = _Stream([5])
    task = asyncio.create_task(_drain(resilience._stream_provider(
        lambda: waiting, provider="ark", profile=_profile(), op="t")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert waiting.closed and breaker.try_acquire()
    breaker.release_probe()

    long = _Stream([0, 5])
    gen = resilience._stream_provider(lambda: long, provider="ark", profile=_profile(), op="t")
    assert await gen.__anext__() == "a"
    await gen.aclose()  # consumer stopped reading
    assert long.closed and breaker.try_acquire() and not breaker._failures


def _capture_trace(monkeypatch):
    from app.services.chat import local_tracer

    steps = []
    monkeypatch.setattr(local_tracer, "record_manual_step", lambda **kw: steps.append(kw))
    return steps


async def test_unary_deadline_is_recorded_in_trace_with_budget(monkeypatch):
    steps = _capture_trace(monkeypatch)

    async def slow():
        await asyncio.sleep(5)

    with llm_deadline(0.7), pytest.raises(LLMDeadlineExceeded):
        await resilience.call_with_resilience(
            slow, primary_provider="ark", profile=_profile(), op="t",
            fallback_factory=slow, fallback_provider="dashscope",
        )
    (step,) = steps
    assert step["name"] == "llm_deadline"
    assert step["inputs"] == {"op": "t", "provider": "ark", "budget_seconds": 0.7}
    assert "0.7 秒" in step["decision_label"]


async def test_stream_fallback_deadline_keeps_its_type_and_traces(monkeypatch):
    steps = _capture_trace(monkeypatch)
    with llm_deadline(1.2), pytest.raises(LLMDeadlineExceeded):
        await _drain(resilience.astream_with_resilience(
            _BrokenStream,
            primary_provider="ark", profile=_profile(), op="t",
            fallback_factory=lambda: _Stream([0, 5]), fallback_provider="dashscope",
        ))
    assert [s["inputs"]["op"] for s in steps] == ["t:fallback"]


class _BrokenStream(_Stream):
    def __init__(self):
        super().__init__([])

    async def __anext__(self):
        raise RuntimeError("provider down")


async def test_nested_budget_reports_the_tighter_one(monkeypatch):
    steps = _capture_trace(monkeypatch)

    async def slow():
        await asyncio.sleep(5)

    with llm_deadline(0.8):
        with llm_deadline(30), pytest.raises(LLMDeadlineExceeded):
            await resilience.call_with_resilience(
                slow, primary_provider="ark", profile=_profile(), op="t")
    assert steps[0]["inputs"]["budget_seconds"] == 0.8
    assert resilience._deadline_budget.get() is None


async def test_trace_failure_never_breaks_the_call(monkeypatch):
    from app.services.chat import local_tracer

    def boom(**_kw):
        raise RuntimeError("trace down")

    monkeypatch.setattr(local_tracer, "record_manual_step", boom)

    async def slow():
        await asyncio.sleep(5)

    with llm_deadline(0.6), pytest.raises(LLMDeadlineExceeded):
        await resilience.call_with_resilience(slow, primary_provider="ark", profile=_profile(), op="t")


async def test_unary_fallback_deadline_traces_the_fallback(monkeypatch):
    steps = _capture_trace(monkeypatch)

    async def down():
        raise RuntimeError("provider down")

    async def slow():
        await asyncio.sleep(5)

    with llm_deadline(1.2), pytest.raises(LLMDeadlineExceeded):
        await resilience.call_with_resilience(
            down, primary_provider="ark", profile=_profile(max_retries=0), op="t",
            fallback_factory=slow, fallback_provider="dashscope",
        )
    assert [s["inputs"]["op"] for s in steps] == ["t:fallback"]
