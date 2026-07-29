"""带 LLM 的夜间任务必须只覆盖近期活跃的 agent.

不收敛的话, 夜间成本随**累计注册数**增长而不是随实际使用量。2026-07 实测: 每个
agent 每天约 6.5 次 cron LLM、100 秒串行延迟; 并发 3 时全部夜间任务在 648 个
agent 超过 6 小时窗口, 2592 个就塞不进一天 —— 按每周 2000 新用户是发布后第 1.3
周。而溢出的任务会全天候占用 LLM 配额, 真正被拖慢的是白天在聊天的真实用户。

反面同样要钉住: 纯计算的任务 (亲密度/耐心恢复) **不能**收敛。它们不烧 LLM, 而漏
算会让数值断档 —— 用户回来时看到的亲密度像是倒退了。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SCHEDULER = Path(__file__).resolve().parents[1] / "jobs" / "scheduler.py"

# 会走到 LLM 的 per-agent 任务, 必须带活跃度门槛
_LLM_TASKS = {
    "Daily schedule",
    "Schedule review",
    "Portrait update",
    "Monthly overview",
    "Memory consolidation",
}
# 纯 DB/Redis 计算, 必须全量跑
_COMPUTE_TASKS = {
    "Growth intimacy",
    "Topic intimacy",
    "Patience recovery",
}


def _fanout_calls() -> dict[str, ast.Call]:
    tree = ast.parse(_SCHEDULER.read_text(encoding="utf-8"))
    out: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_run_for_all_agents"):
            continue
        name = next(
            (ast.literal_eval(k.value) for k in node.keywords
             if k.arg == "task_name" and isinstance(k.value, ast.Constant)),
            None,
        )
        if name:
            out[name] = node
    return out


def _has_gate(call: ast.Call) -> bool:
    return any(kw.arg == "active_within_days" for kw in call.keywords)


def test_every_fanout_task_is_accounted_for():
    """新增 per-agent 任务时必须在这里表态是哪一类, 否则默认全量跑会悄悄回到老问题."""
    seen = set(_fanout_calls())
    known = _LLM_TASKS | _COMPUTE_TASKS
    assert seen == known, (
        f"新增/改名的 per-agent 任务: {sorted(seen - known)}；已消失的: "
        f"{sorted(known - seen)}。带 LLM 的要加 active_within_days，纯计算的不加，"
        "并在本测试的名单里登记。"
    )


@pytest.mark.parametrize("task", sorted(_LLM_TASKS))
def test_llm_tasks_are_gated_on_recent_activity(task: str):
    call = _fanout_calls()[task]
    assert _has_gate(call), (
        f"「{task}」会调 LLM 却对全部 agent 跑 —— 夜间成本会随累计注册数增长，"
        "发布后第 1.3 周就塞不进一天。加 active_within_days="
        "LLM_CRON_ACTIVE_WINDOW_DAYS。"
    )


@pytest.mark.parametrize("task", sorted(_COMPUTE_TASKS))
def test_pure_compute_tasks_stay_unfiltered(task: str):
    call = _fanout_calls()[task]
    assert not _has_gate(call), (
        f"「{task}」是纯 DB/Redis 计算，不烧 LLM。加了活跃度门槛会让休眠用户的"
        "数值断档——他们回来时会看到亲密度像是倒退了。"
    )


def test_gate_uses_the_shared_window_constant():
    """五处用同一个常量, 免得调窗口时漏掉某一个."""
    source = _SCHEDULER.read_text(encoding="utf-8")
    for task in _LLM_TASKS:
        call = _fanout_calls()[task]
        kw = next(k for k in call.keywords if k.arg == "active_within_days")
        assert ast.unparse(kw.value) == "LLM_CRON_ACTIVE_WINDOW_DAYS", (
            f"「{task}」写了字面量而不是共用常量，调窗口时会漏掉它"
        )
    assert "LLM_CRON_ACTIVE_WINDOW_DAYS = 7" in source


def test_activity_query_is_bounded_by_the_window_not_by_history():
    """活跃集合必须靠 (role, created_at) 索引按时间窗查, 不能全表扫.

    这决定了这套收敛能不能一直用下去: 扫描量只跟这几天的流量有关, 跟历史消息总量
    无关。
    """
    tree = ast.parse(_SCHEDULER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_recently_active_agent_ids"
    )
    sql = " ".join(
        node.value for node in ast.walk(fn)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert "role = 'user'" in sql, "应只看用户消息, AI 回复不代表用户还在用"
    assert "created_at >" in sql and "interval" in sql.lower(), "必须按时间窗过滤"
    assert "DISTINCT" in sql.upper(), "同一 agent 多条消息只应产出一次"


def test_dormant_agents_still_get_a_schedule_on_demand():
    """收敛的前提: 休眠 agent 回来时仍能拿到个性化作息.

    连接时后台预热 + 消息路径 await 兜底, 两条都必须在, 且都要带人设 —— 否则
    收敛的代价就变成了"回归用户那天的 AI 像换了个人"。
    """
    ws = (Path(__file__).resolve().parents[1] / "app" / "api" / "realtime" / "ws.py")
    source = ws.read_text(encoding="utf-8")
    assert "_warm_daily_schedule" in source, "缺少连接时的作息预热"

    tree = ast.parse(source)
    warm = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_warm_daily_schedule"
    )
    gen = [
        n for n in ast.walk(warm)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "generate_daily_schedule"
    ]
    assert gen, "预热里没有真的生成作息"
    assert any(kw.arg == "life_overview" for kw in gen[0].keywords), (
        "预热生成时没带人设，会产出通用模板作息"
    )
