"""把 scheduler:health 里的裸时间戳翻译成"这个定时任务是不是活的".

为什么需要这层: 哈希里只有 ok_at / fail_at 两个时刻, 光看时刻判断不了健康与否
—— 一个周任务上次成功在 5 天前完全正常, 一个分钟级任务上次成功在 5 天前说明它
已经死了. 判读必须知道每个任务该多久跑一次.

周期不另存一份表, 直接问运行中的 scheduler 要 trigger 反推。抄一份周期表出来,
下次改 cron 表达式时一定会漂 —— 时区那个 bug 就是"逐个 job 传参数"漏了 14 处。

判读档位 (按严重程度):

    failing   上一轮失败了, 且之后没再成功过
    stale     离上次成功已经超过两个周期 —— 排程没触发, 或实例总在它之前重启
    drifted   一直在跑, 但触发时刻系统性偏离预期 (时区配错就长这样)
    unknown   从没观测到成功也没观测到失败 —— 可能是刚上线的新任务
    healthy   正常

注意 unknown 不算故障。健康记录留 90 天, 新部署的任务确实还没到过触发时刻, 一
上来标红会让整张表失去可信度; 而真正死掉的任务 (L2 那种每晚崩) 会留下 fail_at,
落进 failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 允许错过一个周期再报 stale: 实例重启、锁被别的实例拿走、任务本身跑得久, 都会
# 让某一轮落空, 连续两轮不成功才说明有事。
_STALE_PERIOD_MULTIPLIER = 2

# drifted 的判定容差。任务在结束时才记 ok, 所以 ok_at 天然晚于触发时刻, 容差要
# 覆盖任务自身耗时。取周期的 5% 与 15 分钟的较大者。
_DRIFT_MIN_TOLERANCE = timedelta(minutes=15)
_DRIFT_PERIOD_RATIO = 0.05

# 周期短于这个值的任务不做 drift 判定: 分钟级任务谈不上"触发时刻偏离", 而且
# ok_at 只有秒级精度, 噪声会淹没信号。
_DRIFT_MIN_PERIOD = timedelta(hours=1)


@dataclass
class JobHealth:
    """一个定时任务的健康判读结果."""

    job_id: str
    verdict: str
    last_ok: datetime | None = None
    last_fail: datetime | None = None
    # 原始失败信息。展示端不直接读它 (读 detail), 保留是为了排查时能看到未经组装
    # 的原文 —— detail 会截断和改写。
    fail_reason: str = ""
    period_seconds: float | None = None
    next_run: datetime | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "verdict": self.verdict,
            "last_ok": self.last_ok.isoformat() if self.last_ok else None,
            "last_fail": self.last_fail.isoformat() if self.last_fail else None,
            "fail_reason": self.fail_reason,
            "period_seconds": self.period_seconds,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "detail": self.detail,
        }


@dataclass
class CronHealthReport:
    jobs: list[JobHealth] = field(default_factory=list)
    retired: list[JobHealth] = field(default_factory=list)
    # 这个进程里拿不拿得到任务定义。拿不到时 stale/drifted/retired 三档全部不可
    # 判 —— 只有周期已知才能说"多久没跑算久", 只有确知任务表才能说"这个已下线"。
    definitions_available: bool = True

    @property
    def unhealthy_count(self) -> int:
        return sum(1 for j in self.jobs if j.verdict in ("failing", "stale", "drifted"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "jobs": [j.as_dict() for j in self.jobs],
            "retired": [j.as_dict() for j in self.retired],
            "unhealthy_count": self.unhealthy_count,
            "total": len(self.jobs),
            "definitions_available": self.definitions_available,
        }


def _parse_stamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def trigger_period(trigger: Any, now: datetime) -> timedelta | None:
    """问 trigger 要它的触发间隔.

    没有现成 API, 用"接下来两次触发的间隔"来量。对 cron 里写了多个不等距时刻的
    任务 (hour='3,15') 这个值会偏, 所以下游的容差都留得比较宽。
    """
    try:
        first = trigger.get_next_fire_time(None, now)
        if first is None:
            return None
        second = trigger.get_next_fire_time(first, first + timedelta(seconds=1))
        if second is None:
            return None
        span = second - first
    except Exception:  # trigger 类型五花八门, 量不出来就不判周期相关的档
        return None
    return span if span > timedelta(0) else None


def _classify(
    job_id: str,
    *,
    last_ok: datetime | None,
    last_fail: datetime | None,
    fail_reason: str,
    period: timedelta | None,
    next_run: datetime | None,
    now: datetime,
) -> JobHealth:
    base = JobHealth(
        job_id=job_id,
        verdict="healthy",
        last_ok=last_ok,
        last_fail=last_fail,
        fail_reason=fail_reason,
        period_seconds=period.total_seconds() if period else None,
        next_run=next_run,
    )

    if last_fail is not None and (last_ok is None or last_fail >= last_ok):
        base.verdict = "failing"
        # 原因必须并进 detail。展示端只读 detail —— 让它们各自去 fail_reason 兜底
        # 的话, 健康任务会把几十天前那条陈旧错误也显示出来。
        base.detail = (
            f"上一轮失败后没有再成功过: {fail_reason}"
            if fail_reason
            else "上一轮失败后没有再成功过"
        )
        return base

    if last_ok is None:
        base.verdict = "unknown"
        base.detail = "从未观测到成功 —— 可能是刚上线还没到触发时刻"
        return base

    if period is not None:
        overdue_after = period * _STALE_PERIOD_MULTIPLIER
        behind = now - last_ok
        if behind > overdue_after:
            base.verdict = "stale"
            base.detail = (
                f"已 {_humanize(behind)} 没成功, 该任务每 {_humanize(period)} 跑一次"
            )
            return base

        if period >= _DRIFT_MIN_PERIOD and next_run is not None:
            expected_prev = next_run - period
            offset = abs(last_ok - expected_prev)
            tolerance = max(_DRIFT_MIN_TOLERANCE, period * _DRIFT_PERIOD_RATIO)
            # 偏移必须小于一个周期才算"跑错时刻"。到了一个周期那是漏跑了一轮,
            # 归上面 stale 的容忍范围管 —— 两条规则都去抓同一个现象的话, 小时级
            # 任务偶尔错过一轮就会被报成时刻偏移。
            if tolerance < offset < period:
                base.verdict = "drifted"
                base.detail = (
                    f"上次成功比预期触发时刻偏了 {_humanize(offset)}"
                    f" (预期 {expected_prev:%H:%M}, 实际 {last_ok.astimezone(expected_prev.tzinfo):%H:%M})"
                )
                return base

    return base


def _humanize(delta: timedelta) -> str:
    total = int(abs(delta).total_seconds())
    if total < 90:
        return f"{total} 秒"
    if total < 5400:
        return f"{total // 60} 分钟"
    if total < 172800:
        return f"{total / 3600:.1f} 小时"
    return f"{total / 86400:.1f} 天"


def build_report(
    raw_health: dict[str, str],
    jobs: list[Any],
    *,
    now: datetime | None = None,
) -> CronHealthReport:
    """把 Redis 哈希 + scheduler 的 job 列表合成一份判读.

    `raw_health` 的键形如 "{job_id}:ok_at" / ":fail_at" / ":fail_reason"。
    `jobs` 是 APScheduler 的 job 对象 (需要 .id / .trigger / .next_run_time)。
    """
    now = now or datetime.now(timezone.utc)

    parsed: dict[str, dict[str, str]] = {}
    for key, value in (raw_health or {}).items():
        job_id, _, fieldname = str(key).rpartition(":")
        if not job_id:
            continue
        parsed.setdefault(job_id, {})[fieldname] = value

    report = CronHealthReport(definitions_available=bool(jobs))
    seen: set[str] = set()

    for job in jobs:
        job_id = getattr(job, "id", None)
        if not job_id:
            continue
        seen.add(job_id)
        entry = parsed.get(job_id, {})
        trigger = getattr(job, "trigger", None)
        report.jobs.append(
            _classify(
                job_id,
                last_ok=_parse_stamp(entry.get("ok_at")),
                last_fail=_parse_stamp(entry.get("fail_at")),
                fail_reason=entry.get("fail_reason", ""),
                period=trigger_period(trigger, now),
                next_run=_next_fire(job, trigger, now),
                now=now,
            )
        )

    if not report.definitions_available:
        # 没有任务定义时, 哈希里的每一条都还是"在跑的任务", 只是判不出周期相关的
        # 档。绝不能当成 retired —— 那会让"调度器没在这个进程里跑"看起来像"所有
        # 任务都被删了", 是最吓人的一种假告警。
        for job_id in sorted(parsed):
            entry = parsed[job_id]
            report.jobs.append(
                _classify(
                    job_id,
                    last_ok=_parse_stamp(entry.get("ok_at")),
                    last_fail=_parse_stamp(entry.get("fail_at")),
                    fail_reason=entry.get("fail_reason", ""),
                    period=None,
                    next_run=None,
                    now=now,
                )
            )
    else:
        # 哈希里还留着但 scheduler 已经不注册的任务。删掉 cron 后记录不会自己
        # 消失, 单独列出来免得跟在跑的任务混在一起被误读。
        for job_id in sorted(set(parsed) - seen):
            entry = parsed[job_id]
            report.retired.append(
                JobHealth(
                    job_id=job_id,
                    verdict="retired",
                    last_ok=_parse_stamp(entry.get("ok_at")),
                    last_fail=_parse_stamp(entry.get("fail_at")),
                    fail_reason=entry.get("fail_reason", ""),
                    detail="scheduler 里已无此任务, 是历史遗留记录",
                )
            )

    report.jobs.sort(key=lambda j: (_VERDICT_ORDER.get(j.verdict, 9), j.job_id))
    return report


def _next_fire(job: Any, trigger: Any, now: datetime) -> datetime | None:
    """下次触发时刻.

    优先用 scheduler 算好的 next_run_time; 调度器只注册未启动时那个字段是空的,
    退回自己问 trigger 要 —— 这样判读不依赖调度器有没有 start。
    """
    scheduled = getattr(job, "next_run_time", None)
    if scheduled is not None:
        return scheduled
    if trigger is None:
        return None
    try:
        return trigger.get_next_fire_time(None, now)
    except Exception:
        return None


_VERDICT_ORDER = {"failing": 0, "stale": 1, "drifted": 2, "unknown": 3, "healthy": 4}


async def collect_cron_health(now: datetime | None = None) -> CronHealthReport:
    """从 Redis + 运行中的 scheduler 取数并判读."""
    from app.redis_client import get_redis
    from jobs.scheduler import _JOB_HEALTH_KEY, scheduler

    raw: dict[str, str] = {}
    try:
        redis = await get_redis()
        raw = await redis.hgetall(_JOB_HEALTH_KEY) or {}
    except Exception as exc:  # Redis 挂了也要能出表, 至少能看到注册了哪些任务
        logger.warning(f"cron health: failed to read {_JOB_HEALTH_KEY}: {exc}")

    try:
        jobs = scheduler.get_jobs()
    except Exception as exc:
        logger.warning(f"cron health: failed to list scheduler jobs: {exc}")
        jobs = []

    return build_report(raw, jobs, now=now)
