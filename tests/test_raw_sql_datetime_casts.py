"""裸 SQL 里跟 timestamp 列比较的参数必须显式转型.

这个文件是为一个跑了几个月没人发现的故障写的. L2 动态分级 cron (spec §1.5.2)
每晚崩在:

    operator does not exist: timestamp without time zone >= text

Prisma 的 query_raw 把 Python datetime 序列化成字符串传给 Postgres, 拿去跟
timestamp 列比较就报类型错. 结果是整个动态分级从上线起零产出 —— 457 条 L2
记忆一条都没算过分, 也没有任何升降级.

它活这么久有两个原因, 两个都值得记住:

  测试全 mock 数据库. `patch("...l2_dynamics.db")` 之后 SQL 字符串根本不会送到
  Postgres, 类型错误在 mock 下永远不可能暴露. 单元测试覆盖率对这类 bug 是零。

  失败被 `except Exception: logger.warning(...)` 吞掉. warning 会进日志, 但一个
  每晚定时任务静默失败几个月, 说明没人盯 warning, 而容器日志随部署轮转.

所以这里做静态检查而不是行为测试: 扫描裸 SQL, 凡是拿参数跟时间列比较的地方,
都必须带 ::timestamp / ::timestamptz / ::date 转型. 静态检查不需要真数据库,
能在 CI 里拦住同类写法.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

# 时间类列名 —— 跟这些列比较的参数必须显式转型
_TIME_COLUMNS = (
    "created_at", "updated_at", "occur_time", "statement_time",
    "last_accessed_at", "trigger_time", "last_fired", "deleted_at",
    "completed_at", "starts_at", "ends_at",
)

# `<time_column> <比较符> $n` 且 $n 后面没跟 :: 转型
_UNCAST = re.compile(
    r"\b(" + "|".join(_TIME_COLUMNS) + r")\s*(?:>=|<=|>|<|=)\s*(\$\d+)(?!\s*::)",
    re.IGNORECASE,
)


def _python_files() -> list[Path]:
    return [p for p in APP.rglob("*.py") if "__pycache__" not in p.parts]


def _offenders() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "query_raw" not in text and "execute_raw" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for match in _UNCAST.finditer(line):
                found.append((
                    str(path.relative_to(APP.parent)), i, line.strip()[:100]
                ))
    return found


def test_no_uncast_datetime_parameter_comparisons():
    offenders = _offenders()
    assert not offenders, (
        "裸 SQL 里有参数直接跟时间列比较, 没加显式转型。Prisma 会把 datetime "
        "当字符串传过去, Postgres 报 'operator does not exist: timestamp "
        "without time zone >= text', 整个查询在运行时崩掉 —— 而 mock 掉 db 的"
        "单元测试永远发现不了。加 ::timestamp 即可:\n\n"
        + "\n".join(f"  {f}:{n}  {src}" for f, n, src in offenders)
    )


def test_the_detector_actually_catches_the_original_bug():
    """守卫这个守卫: 正则若失效, 上面那条测试会变成永远通过的空断言."""
    original = "COUNT(*) FILTER (WHERE created_at >= $2)::int AS cnt,"
    assert _UNCAST.search(original), "检测器认不出当初那行有问题的 SQL"

    fixed = "COUNT(*) FILTER (WHERE created_at >= $2::timestamp)::int AS cnt,"
    assert not _UNCAST.search(fixed), "检测器误报了已修好的写法"


@pytest.mark.parametrize("sql", [
    "WHERE occur_time <= $1::timestamptz",
    "AND updated_at > $3::timestamp",
    "WHERE created_at >= $2::date",
])
def test_casted_forms_are_accepted(sql):
    assert not _UNCAST.search(sql)


@pytest.mark.parametrize("sql", [
    "WHERE occur_time <= $1",
    "AND updated_at > $3",
    "WHERE last_fired < $1 AND x = $2",
])
def test_uncasted_forms_are_rejected(sql):
    assert _UNCAST.search(sql)
