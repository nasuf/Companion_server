"""定时任务的成败必须留下痕迹.

L2 动态分级 cron 因为一个 SQL 类型错每晚崩了几个月, 没被任何人发现. 失败走
logger.warning, 成功默认不出声 —— "从来没成功过"和"这次没事可做"在日志里长得
一模一样, 于是无从判断一个任务是不是活的.

修法不是逐个去试那 26 个任务 (试完一次, 下个月又可能悄悄死掉一个), 而是让两种
结局都被记下来. 这里守住这个机制本身: 它一旦失效不会有任何症状, 只是又变回
"看不见".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCHEDULER = Path(__file__).resolve().parent.parent / "jobs" / "scheduler.py"


@pytest.fixture(scope="module")
def source() -> str:
    return SCHEDULER.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """取出一个顶层函数的函数体.

    用 AST 而不是按 "\ndef " 切 —— 后面紧跟 async def 时字符串切分会把下一个
    函数一起吞进来, 断言就变成在检查别人的代码。
    """
    import ast

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"找不到函数 {name}")


def test_no_job_swallows_a_failure_into_a_bare_warning(source):
    """就是这个写法让 L2 那次故障隐身了几个月."""
    offenders = re.findall(r'logger\.warning\(f"[^"]*failed: \{e\}"\)', source)
    assert not offenders, (
        "定时任务的失败又被写成裸 logger.warning 了。改走 _job_failed(name, e) —— "
        "它按 error 级别上报并把失败时刻写进健康记录，否则一个任务可以坏上几个月"
        "而任何界面都看不出来：\n  " + "\n  ".join(offenders)
    )


def test_failures_go_through_the_shared_reporter(source):
    assert "_job_failed(" in source
    assert source.count("_job_failed(") >= 14, "应当所有任务的失败都走这个入口"


def test_reporter_logs_at_error_not_warning(source):
    """warning 在这个项目里量大到没人逐条看; 一个 cron 挂掉是整块功能没运行."""
    body = _function_body(source, "_job_failed")
    assert "logger.error(" in body
    assert "logger.warning(" not in body


def test_success_is_recorded_too(source):
    """只记失败不够 —— 一个从没被触发过的任务不会产生失败记录, 却同样是死的."""
    wrapper = _function_body(source, "_run_distributed_job")
    assert "_record_job_outcome(job_name, ok=True)" in wrapper


def test_health_write_cannot_raise(source):
    """记录健康状态本身失败时必须闭嘴 —— 掩盖掉原始故障比丢一条健康数据更糟."""
    body = _function_body(source, "_record_job_outcome")
    assert "except Exception:" in body and "pass" in body


def test_health_record_expires(source):
    """不设 TTL 的话, 早已删除的任务名会永远留在健康表里误导排查."""
    body = _function_body(source, "_record_job_outcome")
    assert "expire(" in body
