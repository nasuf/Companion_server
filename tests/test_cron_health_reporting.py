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
    """就是这个写法让 L2 那次故障隐身了几个月.

    只匹配单行 `logger.warning(f"...failed: {e}")` 是不够的 —— runtime_job_queue
    写成多行且带 extra=, 就这么从守卫底下溜过去了, 直到巡检上线才发现它的失败从来
    没进过健康记录。所以这里改成: 只要 warning 的消息里出现 "failed: {e}", 不管
    后面还跟着什么参数, 一律算数。
    """
    offenders = re.findall(r'logger\.warning\(\s*f"[^"]*failed: \{e\}"', source)
    assert not offenders, (
        "定时任务的失败又被写成裸 logger.warning 了。改走 _job_failed(name, e) —— "
        "它按 error 级别上报并把失败时刻写进健康记录，否则一个任务可以坏上几个月"
        "而任何界面都看不出来：\n  " + "\n  ".join(offenders)
    )


def test_every_registered_job_can_record_a_success(source):
    """每个注册的 job 都要经过某个会记录成功的包装.

    只记失败的任务在健康表上永远是"未观测" —— 跟"真的死了"完全无法区分, 等于没有
    监控。redis_health_recheck 和 runtime_job_queue 就是这么漏的: 它们刻意不走分布
    式锁 (每实例各自执行), 于是连带着也跳过了健康记录。
    """
    import ast

    tree = ast.parse(source)
    registered: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_job":
            if not node.args:
                continue
            job_id = next(
                (ast.literal_eval(k.value) for k in node.keywords
                 if k.arg == "id" and isinstance(k.value, ast.Constant)),
                None,
            )
            if job_id:
                registered[job_id] = ast.unparse(node.args[0])

    recording_wrappers = {"_run_distributed_job", "_run_local_job"}
    instrumented = {
        fn.name
        for fn in tree.body
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) in recording_wrappers
            for n in ast.walk(fn)
        )
    }

    missing = sorted(
        job_id for job_id, handler in registered.items() if handler not in instrumented
    )
    assert not missing, (
        "这些 job 的处理函数没经过 _run_distributed_job 或 _run_local_job，成功永远"
        "不会被记录，健康表上会一直显示「未观测」——真死了也是同一个显示：\n  "
        + "\n  ".join(missing)
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
    """只记失败不够 —— 一个从没被触发过的任务不会产生失败记录, 却同样是死的.

    只匹配 ok=True 而不锁死参数名: 记什么名字是会变的 (启动补跑要跟定时任务分开
    记), 变量名一变就误报的守卫会被人直接删掉。成功记录的**行为**由
    test_cron_health.py::test_clean_run_still_records_success 覆盖。
    """
    wrapper = _function_body(source, "_run_distributed_job")
    assert re.search(r"_record_job_outcome\([^)]*ok=True", wrapper), (
        "_run_distributed_job 不再记录成功了 —— 健康表将只剩失败, "
        "无法区分「一直没跑」和「跑了没事做」"
    )


def test_health_write_cannot_raise(source):
    """记录健康状态本身失败时必须闭嘴 —— 掩盖掉原始故障比丢一条健康数据更糟."""
    body = _function_body(source, "_record_job_outcome")
    assert "except Exception:" in body and "pass" in body


def test_health_record_expires(source):
    """不设 TTL 的话, 早已删除的任务名会永远留在健康表里误导排查."""
    body = _function_body(source, "_record_job_outcome")
    assert "expire(" in body
