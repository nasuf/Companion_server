"""撤销反思写入的判断.

反思写的是推断而不是陈述 —— 一条错误推断不会报错, 只会让 AI 带着它跟用户相处几个
月。所以必须有一键收回的手段, 而且要能按时间范围收 (通常是"上周那批不对")。

归档而不是删除, 与整合撤销一致: 记忆在这个系统里从不物理删除, 归档后既不参与检索
也仍可追溯。

用法 (生产容器内):
    python revoke_reflections.py                        # 预览全部
    python revoke_reflections.py --since 2026-07-01
    python revoke_reflections.py --user <id> --apply
"""

from __future__ import annotations

import argparse
import asyncio

from app.db import db
from app.services.memory.provenance import REFLECTED

_TABLES = ("memories_user", "memories_ai")


async def _collect(table: str, since: str | None, user_id: str | None) -> list[dict]:
    clauses = ["provenance = $1", "is_archived = false"]
    args: list = [REFLECTED]
    if since:
        args.append(since)
        clauses.append(f"created_at >= ${len(args)}::timestamp")
    if user_id:
        args.append(user_id)
        clauses.append(f"user_id = ${len(args)}")
    rows = await db.query_raw(
        f"SELECT id, user_id, content, created_at FROM {table} "
        f"WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
        *args,
    )
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD, 只收这天之后写的")
    ap.add_argument("--user", help="只收某个用户的")
    ap.add_argument("--apply", action="store_true", help="缺省只预览")
    args = ap.parse_args()

    await db.connect()
    total = revoked = 0
    for table in _TABLES:
        rows = await _collect(table, args.since, args.user)
        if not rows:
            continue
        total += len(rows)
        print(f"\n{table}: {len(rows)} 条")
        for row in rows[:8]:
            print(f"  {str(row['created_at'])[:16]}  {row['content'][:52]}")
        if len(rows) > 8:
            print(f"  … 另外 {len(rows) - 8} 条")

        if not args.apply:
            continue
        ids = [r["id"] for r in rows]
        # 先留痕再归档 —— 与整合撤销同一个顺序, 保证任何被动过的行都有记录。
        await db.execute_raw(
            "INSERT INTO memory_changelogs (id, user_id, memory_id, operation, old_value) "
            "SELECT gen_random_uuid()::text, user_id, id, 'reflection_revoked', "
            "LEFT(content, 200) "
            f"FROM {table} WHERE id = ANY($1::text[])",
            ids,
        )
        revoked += await db.execute_raw(
            f"UPDATE {table} SET is_archived = true WHERE id = ANY($1::text[])",
            ids,
        )

    print()
    if args.apply:
        print(f"已归档 {revoked}/{total} 条反思判断")
    else:
        print(f"共 {total} 条; 加 --apply 执行")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
