"""撤销一次 L3 簇压缩: 把原行取回, 把摘要归档.

整合会**归档原始记忆**, 这是全部记忆维护任务里唯一有实质破坏性的一步。所以在
开启它之前必须先有撤销手段 —— 不是"出问题再想办法", 那时候要面对的是散落在两张
表里的几百行, 而且没人记得哪些属于哪个簇。

依据是 `consolidated_into` changelog: 每条被归档的原行都有一条记录, new_value
指向摘要 ID。归档前先写 changelog 正是为了保证这条线索一定存在。

用法:
    # 看某次整合动了什么 (不改数据)
    python undo_consolidation.py --digest <digest-id>
    python undo_consolidation.py --run <run-id>

    # 真的撤销
    python undo_consolidation.py --digest <digest-id> --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.db import db

_TABLES = ("memories_user", "memories_ai")


async def _digests_of_run(run_id: str) -> list[str]:
    rows = await db.query_raw(
        "SELECT changes FROM memory_consolidation_runs WHERE id = $1", run_id,
    )
    if not rows:
        return []
    raw = rows[0].get("changes")
    payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
    if isinstance(payload, dict):
        return list(payload.get("digest_ids") or [])
    return []


async def _originals_of(digest_id: str) -> list[dict]:
    return await db.query_raw(
        """
        SELECT memory_id, old_value, created_at
        FROM memory_changelogs
        WHERE operation = 'consolidated_into' AND new_value = $1
        ORDER BY created_at
        """,
        digest_id,
    )


async def _restore(digest_id: str, apply: bool) -> tuple[int, int]:
    originals = await _originals_of(digest_id)
    if not originals:
        print(f"  {digest_id[:8]}: 没有 consolidated_into 记录 —— 无从撤销")
        return (0, 0)

    ids = [r["memory_id"] for r in originals]
    print(f"  {digest_id[:8]}: {len(ids)} 条原行")
    for row in originals[:3]:
        preview = (row.get("old_value") or "")[:48]
        print(f"      {row['memory_id'][:8]}  {preview}")
    if len(originals) > 3:
        print(f"      … 另外 {len(originals) - 3} 条")

    if not apply:
        return (len(ids), 0)

    restored = 0
    for table in _TABLES:
        # ID 全局唯一, 对另一张表是空操作。
        restored += await db.execute_raw(
            f"UPDATE {table} SET is_archived = false "
            "WHERE id = ANY($1::text[]) AND is_archived = true",
            ids,
        )
    archived = 0
    for table in _TABLES:
        archived += await db.execute_raw(
            f"UPDATE {table} SET is_archived = true WHERE id = $1", digest_id,
        )
    # 留痕: 撤销本身也要可追溯, 否则下次看到"原行活着且有 consolidated_into"
    # 会以为是整合出了 bug。
    await db.execute_raw(
        """
        INSERT INTO memory_changelogs (id, user_id, memory_id, operation, new_value)
        SELECT gen_random_uuid()::text, cl.user_id, cl.memory_id,
               'consolidation_undone', $1
        FROM memory_changelogs cl
        WHERE cl.operation = 'consolidated_into' AND cl.new_value = $1
        """,
        digest_id,
    )
    print(f"      取回 {restored} 条, 摘要归档 {archived} 条")
    return (len(ids), restored)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", help="摘要记忆 ID")
    ap.add_argument("--run", help="整合 run ID (撤销该次产出的全部摘要)")
    ap.add_argument("--apply", action="store_true", help="缺省只预览")
    args = ap.parse_args()
    if not args.digest and not args.run:
        raise SystemExit("需要 --digest 或 --run")

    await db.connect()
    digests = [args.digest] if args.digest else await _digests_of_run(args.run)
    if not digests:
        await db.disconnect()
        raise SystemExit("没找到要撤销的摘要")

    print(f"{'撤销' if args.apply else '预览'} {len(digests)} 个摘要")
    total_found = total_restored = 0
    for digest in digests:
        found, restored = await _restore(digest, args.apply)
        total_found += found
        total_restored += restored

    await db.disconnect()
    if args.apply:
        print(f"\n完成: 取回 {total_restored}/{total_found} 条原行")
    else:
        print(f"\n将取回 {total_found} 条原行; 加 --apply 执行")


if __name__ == "__main__":
    asyncio.run(main())
