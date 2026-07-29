"""LLM 并发上限: 防洪峰, 且后台不能把前台挤掉.

熔断器是事后止损 —— 等失败率上去了才切断。这层是事前的: provider 抖一下, 在途调用
集体重试, 请求量翻几倍, 于是真的撞上 rate limit, 熔断器打开, 全员降级。提醒触发那
条线早就用 Semaphore(8) 解决过同一个问题, 聊天路径一直没有。

最关键的一条是前台保底: 单一全局信号量挡不住夜间任务把槽位占满 —— 那等于拿白天真
实用户的延迟去换后台吞吐。
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.services.llm import resilience
from app.services.llm.resilience import CallProfile, _run_with_retry, reset_slots_for_testing


@pytest.fixture(autouse=True)
def _clean_slots(monkeypatch):
    # 上限会按 worker 数摊分 (见 _per_worker_share)。这些用例验证的是限流机制本身,
    # 固定成单 worker 让"配置多少就是多少", 摊分逻辑另有 test_multi_worker_safety
    # 专门覆盖。
    monkeypatch.setattr(settings, "web_concurrency", 1)
    reset_slots_for_testing()
    yield
    reset_slots_for_testing()


@pytest.fixture
def _profile():
    return CallProfile(
        timeout_s=5, max_retries=0, retry_backoff_s=(),
        allow_ollama_fallback=False,
    )


class _Tracker:
    """记录并发峰值."""

    def __init__(self):
        self.now = 0
        self.peak = 0
        self.gate = asyncio.Event()

    async def call(self):
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            await self.gate.wait()
            return "ok"
        finally:
            self.now -= 1


async def _fire(n: int, tracker: _Tracker, profile, scope: str = ""):
    from app.services.llm import usage_tracker

    async def _one():
        token = usage_tracker._current_scope.set(scope)
        try:
            return await _run_with_retry(
                tracker.call, provider="dashscope", profile=profile, op="test",
            )
        finally:
            usage_tracker._current_scope.reset(token)

    return [asyncio.create_task(_one()) for _ in range(n)]


class TestCap:
    @pytest.mark.asyncio
    async def test_in_flight_calls_are_capped(self, monkeypatch, _profile):
        monkeypatch.setattr(settings, "llm_max_concurrency", 4)
        tracker = _Tracker()
        tasks = await _fire(20, tracker, _profile)
        await asyncio.sleep(0.05)

        assert tracker.peak == 4, f"在途并发 {tracker.peak}, 应被夹在 4"

        tracker.gate.set()
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_zero_disables_the_cap(self, monkeypatch, _profile):
        """留一个关掉的口子: 限流本身出问题时不必回滚部署."""
        monkeypatch.setattr(settings, "llm_max_concurrency", 0)
        tracker = _Tracker()
        tasks = await _fire(12, tracker, _profile)
        await asyncio.sleep(0.05)

        assert tracker.peak == 12

        tracker.gate.set()
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_providers_get_independent_pools(self, monkeypatch, _profile):
        """rate limit 是按 provider 算的, 一家抖动不该拖累另一家."""
        monkeypatch.setattr(settings, "llm_max_concurrency", 2)
        a, b = _Tracker(), _Tracker()

        async def _run(tracker, provider):
            return await _run_with_retry(
                tracker.call, provider=provider, profile=_profile, op="test",
            )

        tasks = [asyncio.create_task(_run(a, "dashscope")) for _ in range(5)]
        tasks += [asyncio.create_task(_run(b, "doubao")) for _ in range(5)]
        await asyncio.sleep(0.05)

        assert a.peak == 2 and b.peak == 2, "两家各自 2 个槽位, 不共享"

        a.gate.set()
        b.gate.set()
        await asyncio.gather(*tasks)


class TestForegroundReservation:
    @pytest.mark.asyncio
    async def test_background_cannot_take_every_slot(self, monkeypatch, _profile):
        """夜间任务塞满时, 前台聊天仍要拿得到槽位.

        这是整个设计的重点。单一全局信号量在这个用例上会失败 —— 后台先到就全占了。
        """
        monkeypatch.setattr(settings, "llm_max_concurrency", 8)
        monkeypatch.setattr(settings, "llm_background_max_concurrency", 2)

        bg, fg = _Tracker(), _Tracker()
        bg_tasks = await _fire(20, bg, _profile, scope="schedule_cron")
        await asyncio.sleep(0.05)
        assert bg.peak == 2, f"后台占了 {bg.peak} 个槽位, 应被夹在 2"

        fg_tasks = await _fire(10, fg, _profile, scope="chat")
        await asyncio.sleep(0.05)
        assert fg.peak == 6, (
            f"后台占满后前台只拿到 {fg.peak} 个槽位, 应为 8-2=6 —— "
            "前台保底没生效, 夜间任务会把白天聊天的人挤到后面排队"
        )

        bg.gate.set()
        fg.gate.set()
        await asyncio.gather(*bg_tasks, *fg_tasks)

    @pytest.mark.asyncio
    async def test_post_process_counts_as_background(self, monkeypatch, _profile):
        """post_process 由用户消息触发, 但发生在回复推送之后 —— 慢一点没人感知."""
        monkeypatch.setattr(settings, "llm_max_concurrency", 8)
        monkeypatch.setattr(settings, "llm_background_max_concurrency", 2)

        tracker = _Tracker()
        tasks = await _fire(10, tracker, _profile, scope="post_process")
        await asyncio.sleep(0.05)

        assert tracker.peak == 2

        tracker.gate.set()
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_unknown_scope_is_treated_as_foreground(self, monkeypatch, _profile):
        """认不出来的 scope 按前台放行.

        宁可少限一点也不要误伤用户可见的调用 —— 新增 scope 时忘了登记, 后果应该是
        "保护弱了一点", 而不是"用户的回复排队了"。
        """
        monkeypatch.setattr(settings, "llm_max_concurrency", 8)
        monkeypatch.setattr(settings, "llm_background_max_concurrency", 2)

        tracker = _Tracker()
        tasks = await _fire(10, tracker, _profile, scope="brand_new_scope")
        await asyncio.sleep(0.05)

        assert tracker.peak == 8

        tracker.gate.set()
        await asyncio.gather(*tasks)


class TestSlotLifetime:
    @pytest.mark.asyncio
    async def test_backoff_sleep_does_not_hold_a_slot(self, monkeypatch):
        """重试等待期间必须放开槽位.

        不放的话, 一条重试链会把槽位按住十几秒, 限流自己成了瓶颈 —— 而重试恰恰
        发生在 provider 已经不稳的时候, 那时更不该自缚手脚。
        """
        monkeypatch.setattr(settings, "llm_max_concurrency", 1)
        profile = CallProfile(
            timeout_s=5, max_retries=2, retry_backoff_s=(0.05, 0.05),
            allow_ollama_fallback=False,
        )

        holding_during_backoff = False
        attempts = 0

        async def _flaky():
            nonlocal attempts, holding_during_backoff
            attempts += 1
            if attempts <= 2:
                raise RuntimeError("boom")
            return "ok"

        async def _probe():
            # 在第一次失败之后、重试之前插进来。拿得到槽位说明 backoff 没占着。
            await asyncio.sleep(0.02)
            nonlocal holding_during_backoff
            sem = resilience._slots.get("dashscope")
            holding_during_backoff = sem is not None and sem.locked()

        result, _ = await asyncio.gather(
            _run_with_retry(_flaky, provider="dashscope", profile=profile, op="test"),
            _probe(),
        )
        assert result == "ok"
        assert not holding_during_backoff, "backoff 期间仍占着槽位"

    @pytest.mark.asyncio
    async def test_slot_is_released_when_the_call_raises(self, monkeypatch, _profile):
        """失败路径也要还槽位, 否则几次异常就把容量耗光."""
        monkeypatch.setattr(settings, "llm_max_concurrency", 2)

        async def _boom():
            raise RuntimeError("boom")

        for _ in range(5):
            with pytest.raises(Exception):
                await _run_with_retry(
                    _boom, provider="dashscope", profile=_profile, op="test",
                )

        sem = resilience._slots["dashscope"]
        assert not sem.locked(), "异常路径漏还了槽位"

    @pytest.mark.asyncio
    async def test_cap_still_applies_when_resilience_is_killswitched(
        self, monkeypatch, _profile,
    ):
        """kill switch 关的是重试和熔断, 不该把限流一起关掉.

        那个开关是为了排查时少一层干扰, 而故障期恰恰最需要防止请求洪峰。
        """
        monkeypatch.setattr(settings, "llm_resilience_enabled", False)
        monkeypatch.setattr(settings, "llm_max_concurrency", 3)

        tracker = _Tracker()
        tasks = await _fire(10, tracker, _profile)
        await asyncio.sleep(0.05)

        assert tracker.peak == 3

        tracker.gate.set()
        await asyncio.gather(*tasks)
