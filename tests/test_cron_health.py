"""定时任务健康判读 + 失败上报口的回归防线.

两条最重要的不变量, 都对应生产上真实发生过的事:

1. 成功和失败必须记在同一个任务名下。原本外层用 job_name ('capsule_ready_
   notifications'), 任务体里用人类可读标签 ('Capsule ready notification scan'),
   11 个任务全部不一致 —— "fail_at 比 ok_at 新" 这条判读永远不会命中, 胶囊通知
   连崩一天也显示健康。

2. 任务体吞掉异常后不能再记成功。这些任务普遍自己 try/except 再调 _job_failed,
   外层看不到异常; 若无条件记 ok, 失败和成功会写在同一秒, 判读读成健康。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services.ops.cron_health import JobHealth, build_report, trigger_period

_NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


class _FakeTrigger:
    """按固定间隔触发。

    `next_at` 可以指定下次触发的绝对时刻 —— 真实 cron 的下次触发落在日程表上,
    不是"此刻加一个周期"。不指定就退化成后者, 那等于断言上次触发就在此刻, 会让
    任何 ok_at 都显得偏移。
    """

    def __init__(self, period: timedelta, next_at: datetime | None = None):
        self._period = period
        self._next_at = next_at

    def get_next_fire_time(self, previous, now):
        if previous is not None:
            return previous + self._period
        return self._next_at if self._next_at is not None else now + self._period


def _job(
    job_id: str,
    period: timedelta,
    next_run: datetime | None = None,
    *,
    trigger_next: datetime | None = None,
):
    return SimpleNamespace(
        id=job_id,
        trigger=_FakeTrigger(period, next_at=trigger_next),
        next_run_time=next_run,
    )


def _health(job_id: str, *, ok=None, fail=None, reason="") -> dict[str, str]:
    out: dict[str, str] = {}
    if ok:
        out[f"{job_id}:ok_at"] = ok.isoformat()
    if fail:
        out[f"{job_id}:fail_at"] = fail.isoformat()
    if reason:
        out[f"{job_id}:fail_reason"] = reason
    return out


def _verdict(report, job_id: str) -> str:
    for job in report.jobs:
        if job.job_id == job_id:
            return job.verdict
    raise AssertionError(f"{job_id} not in report")


class TestVerdicts:
    def test_recent_success_is_healthy(self):
        report = build_report(
            _health("daily", ok=_NOW - timedelta(hours=2)),
            [_job("daily", timedelta(days=1), next_run=_NOW + timedelta(hours=22))],
            now=_NOW,
        )
        assert _verdict(report, "daily") == "healthy"

    def test_failure_after_last_success_is_failing(self):
        report = build_report(
            _health(
                "capsule",
                ok=_NOW - timedelta(days=2),
                fail=_NOW - timedelta(hours=3),
                reason="Type <class 'datetime.date'> not serializable",
            ),
            [_job("capsule", timedelta(days=1))],
            now=_NOW,
        )
        assert _verdict(report, "capsule") == "failing"

    def test_failure_recorded_in_the_same_second_still_counts_as_failing(self):
        """生产上胶囊那两个时间戳完全相同 —— 失败先记, 外层紧接着记成功.

        判读必须把"同一秒"算成失败, 否则这种最常见的形态会被读成健康。真正的修
        法在 scheduler 侧 (失败过就不记 ok), 这里是判读层的第二道防线。
        """
        same = _NOW - timedelta(hours=1)
        report = build_report(
            _health("capsule", ok=same, fail=same, reason="boom"),
            [_job("capsule", timedelta(days=1))],
            now=_NOW,
        )
        assert _verdict(report, "capsule") == "failing"

    def test_failure_reason_is_merged_into_the_detail(self):
        """展示端只读 detail, 所以原因必须并进去 —— 否则界面上永远看不到.

        反面同样重要: 健康任务的 detail 要保持空, 展示端才不会把几十天前那条陈旧
        错误当成当前状态显示出来。
        """
        report = build_report(
            _health(
                "capsule",
                fail=_NOW - timedelta(hours=1),
                reason="Type <class 'datetime.date'> not serializable",
            ),
            [_job("capsule", timedelta(days=1))],
            now=_NOW,
        )
        detail = report.jobs[0].detail
        assert "not serializable" in detail

        recovered = build_report(
            _health(
                "capsule",
                ok=_NOW - timedelta(minutes=5),
                fail=_NOW - timedelta(days=30),
                reason="很久以前的旧错误",
            ),
            [_job("capsule", timedelta(days=1), next_run=_NOW + timedelta(hours=23))],
            now=_NOW,
        )
        assert recovered.jobs[0].verdict == "healthy"
        assert recovered.jobs[0].detail == ""

    def test_success_far_older_than_two_periods_is_stale(self):
        report = build_report(
            _health("hourly", ok=_NOW - timedelta(hours=9)),
            [_job("hourly", timedelta(hours=1))],
            now=_NOW,
        )
        assert _verdict(report, "hourly") == "stale"

    def test_one_missed_cycle_is_tolerated(self):
        """允许错过一轮: 重启、锁被别的实例拿走都会让某轮落空.

        漏跑一轮时偏移恰好约等于一个周期, 不能被"时刻偏移"抓走 —— 那两条规则
        会互相打架, 小时级任务偶尔落空一次就要报警。
        """
        report = build_report(
            _health("hourly", ok=_NOW - timedelta(minutes=95)),
            [_job(
                "hourly", timedelta(hours=1),
                next_run=_NOW + timedelta(minutes=25),   # 预期上次触发 = 35 分钟前
            )],
            now=_NOW,
        )
        assert _verdict(report, "hourly") == "healthy"

    def test_never_observed_is_unknown_not_failing(self):
        """新上线的任务确实还没到触发时刻, 一上来标红整张表就不可信了."""
        report = build_report({}, [_job("brand_new", timedelta(days=7))], now=_NOW)
        assert _verdict(report, "brand_new") == "unknown"

    def test_running_at_the_wrong_hour_is_drifted(self):
        """时区配错就长这样: 天天在跑, 但比预期时刻晚 8 小时."""
        next_run = _NOW + timedelta(hours=16)          # 预期上次触发 = 8 小时前
        report = build_report(
            _health("daily_schedule", ok=_NOW - timedelta(minutes=1)),
            [_job("daily_schedule", timedelta(days=1), next_run=next_run)],
            now=_NOW,
        )
        assert _verdict(report, "daily_schedule") == "drifted"

    def test_slow_job_finishing_late_is_not_drift(self):
        """任务结束才记 ok, 跑得久天然晚于触发时刻, 不该报 drift."""
        next_run = _NOW + timedelta(hours=23)
        report = build_report(
            _health("daily_schedule", ok=_NOW - timedelta(minutes=50)),
            [_job("daily_schedule", timedelta(days=1), next_run=next_run)],
            now=_NOW,
        )
        assert _verdict(report, "daily_schedule") == "healthy"

    def test_missing_job_definitions_never_mark_everything_retired(self):
        """调度器没在这个进程里跑时, 不能把全部任务说成"已下线".

        命令行巡检就是这个情形 —— 第一版把 17 个在跑的任务全标成 retired, 看起来
        像所有 cron 都被删了。"任务不在 scheduler 里"必须区分两件事: 代码里真的
        删了, 还是这个进程压根没注册。
        """
        raw = {
            **_health("daily_schedule", ok=_NOW - timedelta(hours=1)),
            **_health("l2_adjustment", ok=_NOW - timedelta(hours=2)),
        }
        report = build_report(raw, [], now=_NOW)

        assert report.definitions_available is False
        assert report.retired == []
        assert {j.job_id for j in report.jobs} == {"daily_schedule", "l2_adjustment"}
        assert all(j.verdict == "healthy" for j in report.jobs)

    def test_failure_is_still_detectable_without_job_definitions(self):
        """周期未知也照样能判"上一轮失败了" —— 这一档不需要周期."""
        report = build_report(
            _health("capsule", ok=_NOW - timedelta(days=1), fail=_NOW, reason="boom"),
            [],
            now=_NOW,
        )
        assert _verdict(report, "capsule") == "failing"

    def test_stale_is_not_claimed_without_job_definitions(self):
        """判不出周期就不能说"停跑", 否则是凭空断言."""
        report = build_report(
            _health("hourly", ok=_NOW - timedelta(days=30)), [], now=_NOW,
        )
        assert _verdict(report, "hourly") == "healthy"

    def test_next_fire_falls_back_to_the_trigger_when_not_started(self):
        """调度器只注册未启动时 next_run_time 是空的, 得自己问 trigger."""
        job = _job(
            "daily_schedule", timedelta(days=1),
            next_run=None,                              # 调度器没启动, 字段为空
            trigger_next=_NOW + timedelta(hours=16),    # 预期上次触发 = 8 小时前
        )
        report = build_report(
            _health("daily_schedule", ok=_NOW - timedelta(minutes=1)), [job], now=_NOW,
        )
        assert _verdict(report, "daily_schedule") == "drifted"

    def test_jobs_no_longer_registered_are_listed_separately(self):
        report = build_report(
            _health("memory_reflection", ok=_NOW - timedelta(days=30)),
            [_job("daily", timedelta(days=1))],
            now=_NOW,
        )
        assert [j.job_id for j in report.retired] == ["memory_reflection"]
        assert all(j.job_id != "memory_reflection" for j in report.jobs)

    def test_worst_verdicts_sort_first(self):
        raw = {
            **_health("good", ok=_NOW - timedelta(minutes=5)),
            **_health("bad", ok=_NOW - timedelta(days=3), fail=_NOW - timedelta(hours=1)),
        }
        report = build_report(
            raw,
            [_job("good", timedelta(hours=1)), _job("bad", timedelta(days=1))],
            now=_NOW,
        )
        assert report.jobs[0].job_id == "bad"
        assert report.unhealthy_count == 1


class TestTriggerPeriod:
    def test_reads_period_off_a_real_cron_trigger(self):
        from apscheduler.triggers.cron import CronTrigger

        period = trigger_period(CronTrigger(hour=3, minute=30), _NOW)
        assert period == timedelta(days=1)

    def test_unmeasurable_trigger_degrades_to_none(self):
        class _Dead:
            def get_next_fire_time(self, previous, now):
                return None

        assert trigger_period(_Dead(), _NOW) is None

    def test_period_unknown_never_reports_stale(self):
        """量不出周期就不能判"多久没跑算久", 只能保持 healthy."""
        job = SimpleNamespace(id="odd", trigger=None, next_run_time=None)
        report = build_report(
            _health("odd", ok=_NOW - timedelta(days=400)), [job], now=_NOW,
        )
        assert _verdict(report, "odd") == "healthy"


class TestJobNameConsistency:
    """失败必须记在 _run_distributed_job 声明的规范名下."""

    @pytest.mark.asyncio
    async def test_failure_inside_a_job_records_under_the_canonical_name(self, monkeypatch):
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)

        async def _body():
            # 任务体自己吞掉异常再上报 —— 这是本项目 11 个任务的通用写法
            try:
                raise ValueError("boom")
            except ValueError as e:
                sched._job_failed("Capsule ready notification scan", e)

        await sched._run_distributed_job("capsule_ready_notifications", 60, _body)
        await _drain()

        names = {name for name, _, _ in recorded}
        assert names == {"capsule_ready_notifications"}, (
            f"失败被记到了别的名字下: {names}。成功与失败分记两处会让"
            "'fail_at 比 ok_at 新'的判读永远不命中。"
        )

    @pytest.mark.asyncio
    async def test_swallowed_failure_prevents_recording_success(self, monkeypatch):
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)

        async def _body():
            try:
                raise ValueError("boom")
            except ValueError as e:
                sched._job_failed("some step", e)

        await sched._run_distributed_job("job_a", 60, _body)
        await _drain()

        assert not any(ok for _, ok, _ in recorded), (
            "任务体报过失败还记了成功 —— 两条会落在同一秒, 判读读成健康。"
        )
        assert any(not ok for _, ok, _ in recorded)

    @pytest.mark.asyncio
    async def test_clean_run_still_records_success(self, monkeypatch):
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)

        async def _body():
            return None

        await sched._run_distributed_job("job_b", 60, _body)
        await _drain()

        assert recorded == [("job_b", True, "")]

    @pytest.mark.asyncio
    async def test_failure_state_does_not_leak_into_the_next_job(self, monkeypatch):
        """ContextVar 必须逐轮重置, 否则一次失败会把后续任务全部染成失败."""
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)

        async def _failing():
            try:
                raise ValueError("boom")
            except ValueError as e:
                sched._job_failed("step", e)

        async def _clean():
            return None

        await sched._run_distributed_job("first", 60, _failing)
        await sched._run_distributed_job("second", 60, _clean)
        await _drain()

        assert ("second", True, "") in recorded

    @pytest.mark.asyncio
    async def test_uncaught_exception_is_recorded_before_propagating(self, monkeypatch):
        """任务体没兜住的异常也要留下失败记录.

        否则健康表上只看到"很久没成功", 查不到是为什么 —— 而这恰恰是最需要原因的
        场景。记录之后要继续往上抛, 不能把异常吞掉。
        """
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)

        async def _body():
            raise RuntimeError("unhandled")

        with pytest.raises(RuntimeError):
            await sched._run_distributed_job("job_c", 60, _body)
        await _drain()

        assert recorded == [("job_c", False, "unhandled")]

    @pytest.mark.asyncio
    async def test_outside_a_distributed_job_falls_back_to_the_label(self, monkeypatch):
        import jobs.scheduler as sched

        recorded: list[tuple[str, bool, str]] = []

        async def _fake_record(job_name, ok, detail=""):
            recorded.append((job_name, ok, detail))

        monkeypatch.setattr(sched, "_record_job_outcome", _fake_record)
        sched._job_failed("standalone scan", ValueError("boom"))
        await _drain()

        assert recorded and recorded[0][0] == "standalone scan"


async def _drain() -> None:
    """_job_failed 用 create_task 异步写记录, 让出一轮等它跑完."""
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestReportShape:
    def test_serialises_for_the_admin_api(self):
        report = build_report(
            _health("daily", ok=_NOW - timedelta(hours=1)),
            [_job("daily", timedelta(days=1))],
            now=_NOW,
        )
        payload = report.as_dict()
        assert payload["total"] == 1
        assert payload["unhealthy_count"] == 0
        assert payload["jobs"][0]["job_id"] == "daily"
        assert payload["jobs"][0]["last_ok"].startswith("2026-07-29T11:00")

    def test_empty_health_hash_still_lists_registered_jobs(self):
        """Redis 挂了也要能看到注册了哪些任务, 而不是一片空白."""
        report = build_report({}, [_job("daily", timedelta(days=1))], now=_NOW)
        assert [j.job_id for j in report.jobs] == ["daily"]
        assert isinstance(report.jobs[0], JobHealth)
