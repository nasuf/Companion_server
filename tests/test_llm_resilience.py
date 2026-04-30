"""LLM resilience layer 测试: CircuitBreaker 状态机 + unary retry + stream fallback."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.services.llm.resilience import (
    CircuitBreaker,
    CallProfile,
    LLMCircuitOpenError,
    LLMFailedError,
    astream_with_resilience,
    call_with_resilience,
    reset_breakers_for_testing,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """每个测试独立 CB 状态."""
    reset_breakers_for_testing()
    yield
    reset_breakers_for_testing()


# ═══════════════════════════════════════════════════════════════════
# CircuitBreaker 状态机
# ═══════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def _fresh(self, **overrides) -> CircuitBreaker:
        defaults = dict(failure_threshold=3, window_sec=10.0, cooldown_sec=5.0)
        defaults.update(overrides)
        return CircuitBreaker(**defaults)

    def test_closed_allows_calls(self):
        cb = self._fresh()
        assert cb.state() == "closed"
        assert cb.try_acquire() is True

    def test_opens_after_threshold_failures(self):
        cb = self._fresh()
        for _ in range(3):
            cb.record_failure()
        assert cb.state() == "open"
        assert cb.try_acquire() is False

    def test_open_rejects_calls_until_cooldown(self):
        cb = self._fresh(cooldown_sec=0.05)
        for _ in range(3):
            cb.record_failure()
        assert cb.try_acquire() is False
        import time as _t
        _t.sleep(0.06)
        # cooldown 过 → half_open
        assert cb.state() == "half_open"
        assert cb.try_acquire() is True  # 放行 probe
        assert cb.try_acquire() is False  # 同时只能一个 probe

    def test_half_open_probe_success_closes(self):
        cb = self._fresh(cooldown_sec=0.01)
        for _ in range(3):
            cb.record_failure()
        import time as _t
        _t.sleep(0.02)
        cb.try_acquire()
        cb.record_success()
        assert cb.state() == "closed"
        assert cb.try_acquire() is True

    def test_half_open_probe_failure_reopens(self):
        cb = self._fresh(cooldown_sec=0.01)
        for _ in range(3):
            cb.record_failure()
        import time as _t
        _t.sleep(0.02)
        cb.try_acquire()
        cb.record_failure()
        # 重新 open, cooldown 重置
        assert cb.state() == "open"

    def test_success_resets_window(self):
        cb = self._fresh()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # 窗口清空, 再来 2 次失败不应 open (阈值 3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state() == "closed"


# ═══════════════════════════════════════════════════════════════════
# call_with_resilience (unary)
# ═══════════════════════════════════════════════════════════════════

def _profile(**overrides) -> CallProfile:
    defaults = dict(timeout_s=1.0, max_retries=1, retry_backoff_s=(0.01,))
    defaults.update(overrides)
    return CallProfile(**defaults)


class TestCallWithResilience:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        calls = {"n": 0}

        async def ok():
            calls["n"] += 1
            return "hello"

        result = await call_with_resilience(
            ok, primary_provider="dashscope",
            profile=_profile(), op="test",
        )
        assert result == "hello"
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return "ok"

        result = await call_with_resilience(
            flaky, primary_provider="dashscope",
            profile=_profile(max_retries=1, retry_backoff_s=(0.01,)),
            op="test",
        )
        assert result == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_llm_failed(self):
        async def always_fail():
            raise RuntimeError("down")

        with pytest.raises(LLMFailedError) as ei:
            await call_with_resilience(
                always_fail, primary_provider="dashscope",
                profile=_profile(max_retries=1, retry_backoff_s=(0.01,)),
                op="test",
            )
        assert "down" in str(ei.value)

    @pytest.mark.asyncio
    async def test_timeout_counted_as_failure(self):
        async def slow():
            await asyncio.sleep(1.0)
            return "late"

        with pytest.raises(LLMFailedError):
            await call_with_resilience(
                slow, primary_provider="dashscope",
                profile=_profile(timeout_s=0.05, max_retries=0, retry_backoff_s=()),
                op="test",
            )

    @pytest.mark.asyncio
    async def test_fallback_to_ollama_when_primary_fails(self):
        primary_calls, fallback_calls = {"n": 0}, {"n": 0}

        async def primary_fail():
            primary_calls["n"] += 1
            raise RuntimeError("dashscope down")

        async def fallback_ok():
            fallback_calls["n"] += 1
            return "fallback-reply"

        result = await call_with_resilience(
            primary_fail, primary_provider="dashscope",
            profile=_profile(max_retries=0, retry_backoff_s=()),
            op="test",
            fallback_factory=fallback_ok,
        )
        assert result == "fallback-reply"
        assert primary_calls["n"] == 1
        assert fallback_calls["n"] == 1

    @pytest.mark.asyncio
    async def test_both_primary_and_fallback_fail_raises(self):
        async def fail():
            raise RuntimeError("x")

        with pytest.raises(LLMFailedError) as ei:
            await call_with_resilience(
                fail, primary_provider="dashscope",
                profile=_profile(max_retries=0, retry_backoff_s=()),
                op="test",
                fallback_factory=fail,
            )
        assert "both" in str(ei.value).lower() or "fallback" in str(ei.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        async def fail():
            raise RuntimeError("x")

        profile = _profile(max_retries=0, retry_backoff_s=())
        # 阈值跟 settings 同步 (避免 hardcode 跟 config 漂)
        from app.config import settings
        threshold = settings.llm_cb_failure_threshold
        for _ in range(threshold):
            try:
                await call_with_resilience(
                    fail, primary_provider="dashscope", profile=profile, op="test",
                )
            except LLMFailedError:
                pass

        # 下一次: CB open, 不进 retry, 直接快速失败
        with pytest.raises(LLMCircuitOpenError):
            await call_with_resilience(
                fail, primary_provider="dashscope", profile=profile, op="test",
            )


# ═══════════════════════════════════════════════════════════════════
# astream_with_resilience
# ═══════════════════════════════════════════════════════════════════

class _Chunk:
    """Minimal LangChain AIMessageChunk stand-in (has .content)."""
    def __init__(self, content: str) -> None:
        self.content = content


async def _stream_of(tokens: list[str]) -> AsyncIterator[_Chunk]:
    for t in tokens:
        yield _Chunk(t)


async def _slow_first_chunk(delay: float) -> AsyncIterator[_Chunk]:
    await asyncio.sleep(delay)
    yield _Chunk("late")


async def _stream_raises() -> AsyncIterator[_Chunk]:
    raise RuntimeError("stream broken")
    yield _Chunk("unreachable")  # noqa: unreachable pragma


class TestAstreamWithResilience:
    @pytest.mark.asyncio
    async def test_primary_stream_success(self):
        tokens = []
        async for t in astream_with_resilience(
            lambda: _stream_of(["h", "i"]),
            primary_provider="dashscope",
            profile=_profile(timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                             first_chunk_timeout_s=1.0),
            op="reply_stream",
        ):
            tokens.append(t)
        assert tokens == ["h", "i"]

    @pytest.mark.asyncio
    async def test_first_chunk_timeout_triggers_fallback(self):
        tokens = []
        async for t in astream_with_resilience(
            lambda: _slow_first_chunk(0.5),
            primary_provider="dashscope",
            profile=_profile(timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                             first_chunk_timeout_s=0.05),
            op="reply_stream",
            fallback_factory=lambda: _stream_of(["fb1", "fb2"]),
        ):
            tokens.append(t)
        assert tokens == ["fb1", "fb2"]

    @pytest.mark.asyncio
    async def test_primary_raise_pre_first_chunk_triggers_fallback(self):
        tokens = []
        async for t in astream_with_resilience(
            _stream_raises,
            primary_provider="dashscope",
            profile=_profile(timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                             first_chunk_timeout_s=0.5),
            op="reply_stream",
            fallback_factory=lambda: _stream_of(["fb1"]),
        ):
            tokens.append(t)
        assert tokens == ["fb1"]

    @pytest.mark.asyncio
    async def test_mid_stream_failure_no_fallback(self):
        """first chunk 到达后 mid-stream 抛异常: 不走 fallback (避免拼接错乱)."""
        async def half_then_fail():
            yield _Chunk("part1")
            raise RuntimeError("stream broke mid-way")

        with pytest.raises(LLMFailedError):
            tokens = []
            async for t in astream_with_resilience(
                half_then_fail,
                primary_provider="dashscope",
                profile=_profile(timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                                 first_chunk_timeout_s=1.0),
                op="reply_stream",
                fallback_factory=lambda: _stream_of(["fb"]),
            ):
                tokens.append(t)
        # 前半 token 已 yield 到调用方 (partial delivery)
        assert tokens == ["part1"]

    @pytest.mark.asyncio
    async def test_both_primary_and_fallback_fail(self):
        with pytest.raises(LLMFailedError):
            async for _ in astream_with_resilience(
                _stream_raises,
                primary_provider="dashscope",
                profile=_profile(timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                                 first_chunk_timeout_s=0.5),
                op="reply_stream",
                fallback_factory=_stream_raises,
            ):
                pass

    @pytest.mark.asyncio
    async def test_idle_timeout_triggers_failure_after_first_chunk(self):
        """首字节正常但相邻 chunk 间停顿 > idle_timeout_s 应抛 LLMFailedError.

        模拟 dashscope 首字节快但中间 hang 的场景 (背景生成最大顾虑).
        first_chunk 已落地, mid-stream 失败不再 fallback (避免拼接错乱).
        """
        async def first_then_hang():
            yield _Chunk("first")
            await asyncio.sleep(1.0)  # > idle_timeout_s
            yield _Chunk("never reaches consumer")

        tokens = []
        with pytest.raises(LLMFailedError) as exc_info:
            async for t in astream_with_resilience(
                first_then_hang,
                primary_provider="dashscope",
                profile=_profile(
                    timeout_s=5.0, max_retries=0, retry_backoff_s=(),
                    first_chunk_timeout_s=0.5,
                    idle_timeout_s=0.1,  # 极短 idle 触发
                ),
                op="reply_stream",
                fallback_factory=lambda: _stream_of(["fb"]),
            ):
                tokens.append(t)
        assert tokens == ["first"]
        assert "idle" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_idle_timeout_falls_back_to_first_chunk_when_unset(self):
        """idle_timeout_s=None 时, chunk 间停顿沿用 first_chunk_timeout_s 行为
        (与改动前等价, 老 chat_stream profile 不变)."""
        async def first_then_pause_within_first_chunk_window():
            yield _Chunk("a")
            await asyncio.sleep(0.05)  # < first_chunk_timeout_s
            yield _Chunk("b")

        tokens = []
        async for t in astream_with_resilience(
            first_then_pause_within_first_chunk_window,
            primary_provider="dashscope",
            profile=_profile(
                timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                first_chunk_timeout_s=0.5,
                idle_timeout_s=None,  # 默认: 沿用 first_chunk_timeout_s = 0.5s
            ),
            op="reply_stream",
        ):
            tokens.append(t)
        assert tokens == ["a", "b"]

    @pytest.mark.asyncio
    async def test_idle_timeout_unset_still_enforces_first_chunk_window_mid_stream(self):
        """idle_timeout_s=None 时停顿超过 first_chunk_timeout_s 仍然失败,
        证明 None fallback 是真的沿用而不是无限期等待."""
        async def first_then_long_pause():
            yield _Chunk("a")
            await asyncio.sleep(0.5)  # > first_chunk_timeout_s
            yield _Chunk("b")

        with pytest.raises(LLMFailedError):
            async for _t in astream_with_resilience(
                first_then_long_pause,
                primary_provider="dashscope",
                profile=_profile(
                    timeout_s=5.0, max_retries=0, retry_backoff_s=(),
                    first_chunk_timeout_s=0.1,
                    idle_timeout_s=None,
                ),
                op="reply_stream",
            ):
                pass
