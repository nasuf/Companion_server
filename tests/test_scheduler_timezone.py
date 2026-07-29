"""所有 cron 必须按配置的时区触发, 不是按容器时区.

APScheduler 缺省用**进程所在时区**。容器跑在 UTC 而配置写的是 Asia/Shanghai, 于是
每个 `hour=` 都被当成 UTC —— 比本意晚 8 小时。

生产实测 (2026-07-29): 15 个 cron 里只有 achievement_daily_rollup 一个显式传了
timezone=, 其余 14 个全在漂。后果不是"晚一点跑"这么轻:

    每日作息本该 03:30 生成, 实际 11:30。也就是每天 00:00-11:30 之间当天没有作息表,
    这段时间聊天走缓存 miss → 现场生成 → 那条路径不传 lifeOverview → 退化成通用
    模板。用户上午聊到的"我今天要做什么", 是一份跟这个 agent 无关的上班族作息。

失效方式也很典型: 任务照常跑、日志照常打、数据照常写, 只是时间点全错。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.config import settings

_SCHEDULER_PY = Path(__file__).resolve().parents[1] / "jobs" / "scheduler.py"


def _add_job_calls() -> list[ast.Call]:
    tree = ast.parse(_SCHEDULER_PY.read_text())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_job"
    ]


def _trigger_of(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "trigger":
            return getattr(kw.value, "value", "")
    if len(call.args) >= 2:
        return getattr(call.args[1], "value", "")
    return ""


def _job_id(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "id":
            return getattr(kw.value, "value", "?")
    return "?"


def test_scheduler_declares_its_timezone():
    """设在 scheduler 上而不是逐个 job 传 —— 逐个传下次新增 job 还会漏。"""
    source = _SCHEDULER_PY.read_text()
    assert "AsyncIOScheduler(timezone=" in source, (
        "AsyncIOScheduler 没设时区, 会用容器时区 (生产是 UTC), "
        "所有 cron 的 hour= 都会偏 8 小时"
    )
    assert "settings.schedule_timezone" in source.split("AsyncIOScheduler(")[1][:80], (
        "时区应取自配置, 不要写死"
    )


def test_scheduler_instance_actually_uses_the_configured_zone():
    """断言实例状态而不只是源码 —— 源码写对但被别处覆盖也要抓到。"""
    from jobs.scheduler import scheduler

    assert str(scheduler.timezone) == settings.schedule_timezone


@pytest.mark.parametrize(
    "job_id",
    [
        _job_id(call) for call in _add_job_calls()
        if _trigger_of(call) == "cron"
    ],
)
def test_every_cron_job_resolves_to_the_configured_zone(job_id):
    """每个 cron job 要么显式传时区, 要么继承 scheduler 的 —— 两者都指向配置值。

    这条参数化会随新增 job 自动覆盖, 所以下次加任务时忘了时区会直接失败。
    """
    call = next(
        c for c in _add_job_calls()
        if _trigger_of(c) == "cron" and _job_id(c) == job_id
    )
    explicit = next(
        (kw for kw in call.keywords if kw.arg == "timezone"), None,
    )
    if explicit is None:
        # 继承 scheduler 的全局时区, 由上面两条测试保证它是对的
        return
    assert ast.unparse(explicit.value) == "settings.schedule_timezone", (
        f"{job_id} 显式传了一个非配置的时区"
    )


def test_the_intended_local_hours_are_still_what_the_comments_say():
    """几个关键任务的本地时刻。时区修好之后它们才真的成立 —— 之前注释写的是本意,
    实际执行时刻全部 +8。"""
    from jobs import scheduler as sched

    source = inspect.getsource(sched)
    # 作息生成必须在凌晨: 它要在用户醒来之前把当天的表准备好
    assert "hour=3," in source and "minute=30," in source, (
        "daily_schedule 的时刻变了; 若是有意调整, 更新这条断言并确认它仍在凌晨"
    )
