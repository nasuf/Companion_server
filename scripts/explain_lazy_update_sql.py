"""在真实 Postgres 上验证惰性更新的 SQL 能解析、能走索引.

单元测试只能比对字符串 —— 一条语法错误的 SQL 照样能通过所有断言, 然后在生产上
每轮对话静默失败 (调用方吞异常, 因为效用值更新不该影响回复)。那会重演夜间 cron
死了几个月无人察觉的老故事, 只是这次更隐蔽。

所以上线前必须让数据库自己说一次。EXPLAIN 不执行 UPDATE, 只做解析和计划。
"""

from __future__ import annotations

import asyncio

from app.db import db
from app.services.memory.lifecycle.lazy_update import (
    _render_sql, _singleton_arrays, _TABLES,
)


async def main() -> None:
    await db.connect()
    ok = True
    for table in _TABLES:
        sql = _render_sql(table)
        try:
            sg_main, sg_sub = _singleton_arrays()
            rows = await db.query_raw(
                f"EXPLAIN {sql}", ["probe-id"], [True], sg_main, sg_sub,
            )
            plan = "\n".join(str(r.get("QUERY PLAN", r)) for r in rows)
            seq_scans = plan.count("Seq Scan")
            print(f"[{table}] 解析通过, {len(rows)} 行计划, Seq Scan × {seq_scans}")
            for line in plan.splitlines()[:6]:
                print(f"    {line}")
            if seq_scans:
                print(f"    ⚠ 有 {seq_scans} 处全表扫描 —— 热路径上会随表增长变慢")
        except Exception as e:
            ok = False
            print(f"[{table}] 解析失败: {e}")

    # 兜底扫描的 SQL 是改写出来的, 单独验一遍 —— 改写失败会静默跳过整张表。
    from app.services.memory.lifecycle import lazy_update

    calls: list = []

    async def _capture(sql, *args):
        calls.append((sql, args))
        return 0

    original = lazy_update.db.execute_raw
    lazy_update.db.execute_raw = _capture
    try:
        await lazy_update.sweep_stale_values(older_than_days=30, limit=10)
    finally:
        lazy_update.db.execute_raw = original

    if len(calls) != len(_TABLES):
        ok = False
        print(f"\n兜底扫描只产出 {len(calls)} 条 SQL, 期望 {len(_TABLES)} —— 改写失败")
    for sql, args in calls:
        try:
            await db.query_raw(f"EXPLAIN {sql}", *args)
            print(f"[sweep] 解析通过 ({len(args)} 个参数)")
        except Exception as e:
            ok = False
            print(f"[sweep] 解析失败: {e}")

    await db.disconnect()
    print("\n全部通过" if ok else "\n有失败项 —— 不要上线")


if __name__ == "__main__":
    asyncio.run(main())
