"""核对惰性衰减的 SQL 与 Python 实现算出同一个数.

两处实现分工不同 —— SQL 服务热路径 (让数据库基于当前行值做算术, 免读-改-写竞态),
Python 服务离线推演。它们一旦漂移, 推演结论就不适用于生产, 而推演正是判断
"改了会不会变差"的唯一依据。

单元测试只能比对 SQL 字符串结构; 数值等价必须真的让 Postgres 算一遍。这个脚本
用只读 SELECT 做, 不碰任何数据。
"""

from __future__ import annotations

import asyncio

from app.db import db
from app.services.memory.lifecycle.value import (
    ACCESS_CEILING,
    ACCESS_REWARD,
    CONTRIBUTION_REWARD,
    DECAY_LAMBDA,
    VALUE_MAX,
    apply_usage,
)

# (起始值, 闲置天数, 是否注入)
CASES = [
    (0.86, 0.0, True), (0.86, 0.0, False),
    (0.86, 30.0, True), (0.86, 30.0, False),
    (0.50, 180.0, True), (0.50, 180.0, False),
    (0.95, 365.0, True), (0.20, 730.0, False),
    (0.99, 0.0, True), (0.79, 1.0, False),
    (0.10, 5000.0, False), (0.0, 10.0, True),
]


async def main() -> None:
    await db.connect()
    print(f"{'起始值':>8}{'闲置天':>9}{'注入':>6}{'SQL':>10}{'Python':>10}{'差':>11}")
    worst = 0.0
    for value, days, contributed in CASES:
        rows = await db.query_raw(
            """
            SELECT CASE WHEN $3::bool
              THEN LEAST($7::float8, GREATEST(0.0, d.decayed + $5::float8))
              ELSE LEAST($7::float8, GREATEST(0.0,
                d.decayed + $4::float8 * GREATEST(0.0, $6::float8 - d.decayed)))
            END AS val
            FROM (SELECT $1::float8 * EXP(-$8::float8 * $2::float8) AS decayed) AS d
            """,
            value, days, contributed,
            ACCESS_REWARD, CONTRIBUTION_REWARD, ACCESS_CEILING, VALUE_MAX,
            DECAY_LAMBDA,
        )
        sql_value = float(rows[0]["val"])
        py = apply_usage(
            value=value, level=2, days_idle=days,
            accessed=not contributed, contributed=contributed,
        ).value
        delta = abs(sql_value - py)
        worst = max(worst, delta)
        flag = "" if delta < 1e-9 else "  ← 不一致"
        print(f"{value:>8.2f}{days:>9.0f}{str(contributed):>6}"
              f"{sql_value:>10.4f}{py:>10.4f}{delta:>11.2e}{flag}")

    await db.disconnect()
    print(f"\n最大偏差 {worst:.2e}")
    print("一致" if worst < 1e-9 else "不一致 —— 热路径与推演会给出不同结论")


if __name__ == "__main__":
    asyncio.run(main())
