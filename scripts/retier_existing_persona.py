"""把存量 agent 的建号人设按现行分层规则重排一次.

2026-07 起新建 agent 的人设按事实种类分层: 核心身份 (L1_SINGLETON_SUBS) 留 L1,
其余降到 L2。但**存量 agent 仍是整份进 L1** —— 同一套代码下老号和新号的记忆结构
不一致, 而老号那批恰恰是实测最没用的一类 (建号人设检索有用率 11-20%, 聊天学到的
L2 是 29-37%, p=0.0001)。

## 哪些行算"建号人设"

    provenance = 'profile_seed'                     明确标记的
    provenance IS NULL 且 L1 且建号 10 分钟内写入    provenance 列引入前的历史数据

10 分钟这个窗口不是拍的。生产实测 5508 条 NULL/L1 行相对建号时刻的写入延迟:

    <1min      4602    建号时批量写入
    1-10min     899    _fill_main_gaps 补齐缺口 (要调 LLM, 慢)
    10-60min      1    《五子棋》对局记录 —— 聊天学到的
    1h-1d         1    "去手作店做陶瓷杯" —— 聊天学到的
    >1d           5    同上

10 分钟处有一道干净的分界: 之下 5501 条全是人设, 之上 7 条全是聊天学到的。

## 哪些**不**算

    knowledge_seed   admin 维护的知识, 新 agent 今天仍写 L1 —— 重排它反而会让
                     老号偏离新号, 方向正好反了
    非 L1 的行        本来就不在要修的问题里
    聊天学到的记忆    实测比人设更有用, 没有理由降级

## 可回滚

每条改动写 changelog `retier_persona`, old_value 记原始 "level=N importance=X",
所以任何一条都能按原值还原。加 --revert 即可整体回退。

用法:
    python retier_existing_persona.py              # 预览
    python retier_existing_persona.py --apply
    python retier_existing_persona.py --revert --apply
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections import Counter

from app.db import db
from app.services.life_story import _tiered_importance
from app.services.memory.config import level_for_importance

CREATION_WINDOW_MINUTES = 10
_TABLES = ("memories_ai", "memories_user")

_SCOPE_SQL = """
SELECT m.id, m.user_id, m.workspace_id, m.main_category, m.sub_category,
       m.content, m.importance, m.level
FROM {table} m
JOIN chat_workspaces w ON w.id = m.workspace_id
WHERE m.is_archived = false
  AND m.level = 1
  AND (
    m.provenance = 'profile_seed'
    OR (
      m.provenance IS NULL
      AND m.created_at <= w.created_at + INTERVAL '{window} minutes'
    )
  )
"""


async def _load_scope(table: str) -> list[dict]:
    return await db.query_raw(
        _SCOPE_SQL.format(table=table, window=CREATION_WINDOW_MINUTES)
    )


def _plan(rows: list[dict]) -> list[dict]:
    """算出每行的新分数与新层级, 只保留真的会变的。"""
    changes: list[dict] = []
    for r in rows:
        old_imp = float(r["importance"] or 0)
        new_imp = _tiered_importance(r["main_category"], r["sub_category"], old_imp)
        new_level = level_for_importance(new_imp)
        if new_level == r["level"] and abs(new_imp - old_imp) < 1e-9:
            continue
        changes.append({**r, "new_importance": new_imp, "new_level": new_level})
    return changes


# Postgres 单条预备语句最多 32767 个绑定参数。changelog 每行占 6 个, 5478 行会到
# 32868 —— 整条被拒 (好在是原子失败, 不会写一半)。分批的每一批仍然保持"先写
# changelog 再改写"的顺序, 所以任何被改的行都有还原依据。
_MAX_ROWS_PER_BATCH = 4000


async def _apply(table: str, changes: list[dict]) -> int:
    if not changes:
        return 0
    total = 0
    for start in range(0, len(changes), _MAX_ROWS_PER_BATCH):
        total += await _apply_batch(table, changes[start:start + _MAX_ROWS_PER_BATCH])
    return total


async def _apply_batch(table: str, changes: list[dict]) -> int:
    """一批的改写 + 留痕。changelog 先写, 保证任何被改的行都有还原依据。"""
    values = ",".join(
        f"(${i * 6 + 1}, ${i * 6 + 2}, ${i * 6 + 3}, ${i * 6 + 4}, "
        f"'retier_persona', ${i * 6 + 5}, ${i * 6 + 6})"
        for i in range(len(changes))
    )
    args: list = []
    for c in changes:
        args.extend((
            str(uuid.uuid4()), c["user_id"], c["workspace_id"], c["id"],
            f"level={c['level']} importance={float(c['importance']):.4f}",
            f"level={c['new_level']} importance={c['new_importance']:.4f}",
        ))
    await db.execute_raw(
        "INSERT INTO memory_changelogs "
        "(id, user_id, workspace_id, memory_id, operation, old_value, new_value) "
        f"VALUES {values}",
        *args,
    )

    return await db.execute_raw(
        f"""
        UPDATE {table} AS t SET level = u.lvl, importance = u.imp
        FROM (SELECT unnest($1::text[]) AS id, unnest($2::int[]) AS lvl,
                     unnest($3::float8[]) AS imp) AS u
        WHERE t.id = u.id
        """,
        [c["id"] for c in changes],
        [c["new_level"] for c in changes],
        [c["new_importance"] for c in changes],
    )


async def _revert(table: str, apply: bool) -> int:
    """按 changelog 的 old_value 还原。"""
    rows = await db.query_raw(
        f"""
        SELECT cl.memory_id, cl.old_value
        FROM memory_changelogs cl
        JOIN {table} m ON m.id = cl.memory_id
        WHERE cl.operation = 'retier_persona'
        """
    )
    if not rows:
        return 0
    ids, levels, imps = [], [], []
    for r in rows:
        parts = dict(
            piece.split("=") for piece in str(r["old_value"] or "").split() if "=" in piece
        )
        if "level" not in parts or "importance" not in parts:
            continue
        ids.append(r["memory_id"])
        levels.append(int(parts["level"]))
        imps.append(float(parts["importance"]))
    print(f"  {table}: 可还原 {len(ids)} 行")
    if not apply or not ids:
        return 0
    return await db.execute_raw(
        f"""
        UPDATE {table} AS t SET level = u.lvl, importance = u.imp
        FROM (SELECT unnest($1::text[]) AS id, unnest($2::int[]) AS lvl,
                     unnest($3::float8[]) AS imp) AS u
        WHERE t.id = u.id
        """,
        ids, levels, imps,
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="缺省只预览")
    ap.add_argument("--revert", action="store_true", help="按 changelog 还原")
    args = ap.parse_args()

    await db.connect()

    if args.revert:
        print(f"{'还原' if args.apply else '预览还原'} retier_persona 的全部改动")
        total = sum([await _revert(t, args.apply) for t in _TABLES])
        print(f"{'已还原' if args.apply else '将还原'} {total} 行")
        await db.disconnect()
        return

    grand_total = 0
    for table in _TABLES:
        rows = await _load_scope(table)
        changes = _plan(rows)
        moves = Counter(f"L{c['level']}→L{c['new_level']}" for c in changes)
        kept = len(rows) - len(changes)
        print(f"\n{table}: 命中人设 {len(rows)} 行")
        print(f"  保持 L1 (核心身份) {kept} 行")
        print(f"  层级变动 {len(changes)} 行 {dict(moves)}")
        for c in changes[:3]:
            print(f"    [{c['main_category']}/{c['sub_category']}] "
                  f"{float(c['importance']):.2f}→{c['new_importance']:.2f} "
                  f"L{c['level']}→L{c['new_level']}  {c['content'][:38]}")
        if args.apply:
            n = await _apply(table, changes)
            print(f"  已改写 {n} 行")
        grand_total += len(changes)

    if not args.apply:
        print(f"\n共 {grand_total} 行将被重排; 加 --apply 执行")
        print("撤销: --revert --apply")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
