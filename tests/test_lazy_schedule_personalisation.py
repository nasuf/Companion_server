"""缓存未命中时按需生成的作息, 必须带上生活画像.

generate_daily_schedule 在没有 life_overview 时会走通用模板分支, 产出一份跟这个
agent 的职业/性格无关的作息。而缓存未命中最典型的时刻, 恰恰是「久未上线的用户回来
说第一句话」—— 那天 AI 会显得完全换了个人, 而且极难归因: 作息本身看起来是正常的,
只是不像它自己。

聊天热路径 (ws.py) 原本一个参数都没传, 而 agents.py 那处自己判 dict 取
".description" —— 跟 save_life_overview 写入的纯字符串对不上, 实际也永远取到
None。两处统一走 get_life_overview。

这条约束在按活跃度收敛 cron 之后会更重要: 休眠 agent 不再有夜间预生成, 按需生成
就成了唯一来源。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CALL_SITES = (
    _ROOT / "app" / "api" / "realtime" / "ws.py",
    _ROOT / "app" / "api" / "public" / "agents.py",
)


def _is_generate_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        getattr(node.func, "id", None) == "generate_daily_schedule"
        or getattr(node.func, "attr", None) == "generate_daily_schedule"
    )


def _generate_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if _is_generate_call(node)]


def _cache_miss_generate_calls(path: Path) -> list[ast.Call]:
    """只取「因为缓存未命中才生成」的调用点.

    判据是所在函数同时调了 get_cached_schedule。agent 创建流程也会生成作息, 但它
    刚算出 life_overview 直接传下去 —— 再回头去读一遍缓存是多余的往返, 还可能跟
    刚写入的值抢时序。那种调用不该被这条规则管。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[ast.Call] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(ast.walk(fn))
        reads_cache = any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "get_cached_schedule"
            for n in body
        )
        if reads_cache:
            out.extend(n for n in body if _is_generate_call(n))
    return out


@pytest.mark.parametrize("path", _CALL_SITES, ids=lambda p: p.name)
def test_every_lazy_generation_passes_life_overview(path: Path):
    calls = _generate_calls(path)
    assert calls, f"{path.name} 里找不到 generate_daily_schedule 调用"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        assert "life_overview" in kwargs, (
            f"{path.name}:{call.lineno} 按需生成作息时没传 life_overview —— "
            "会退化成通用模板, 那天的 AI 跟它自己的职业/性格无关。"
        )


@pytest.mark.parametrize("path", _CALL_SITES, ids=lambda p: p.name)
def test_life_overview_is_read_through_the_shared_accessor(path: Path):
    """不要各自解析这个字段.

    它被写入时是纯字符串 (save_life_overview), 但历史上有地方按 dict 取
    ".description" —— 那种写法不会报错, 只是安静地永远返回 None。
    """
    source = path.read_text(encoding="utf-8")
    calls = _cache_miss_generate_calls(path)
    assert calls, f"{path.name} 里找不到「缓存未命中才生成」的调用点"
    for call in calls:
        for kw in call.keywords:
            if kw.arg != "life_overview":
                continue
            rendered = ast.unparse(kw.value)
            assert "get_life_overview" in rendered, (
                f"{path.name}:{call.lineno} 自己解析 lifeOverview ({rendered}) —— "
                "统一走 get_life_overview, 它带 Redis 缓存和 DB 兜底。"
            )
    assert "get_life_overview" in source


def test_generic_template_branch_still_exists_as_the_fallback():
    """人设确实缺失时仍要能出一份作息, 而不是抛异常让聊天中断."""
    schedule_src = (
        _ROOT / "app" / "services" / "schedule_domain" / "schedule.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(schedule_src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_daily_schedule"
    )
    arg_names = [a.arg for a in fn.args.args]
    assert "life_overview" in arg_names, "签名里应保留 life_overview 且可省略"
    # life_overview 有默认值 → 缺人设时不会 TypeError
    offset = len(fn.args.args) - len(fn.args.defaults)
    idx = arg_names.index("life_overview")
    assert idx >= offset, "life_overview 必须有默认值, 否则缺人设时直接抛异常"
