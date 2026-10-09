"""撤销一次 L3 簇压缩: 把原行取回, 把摘要归档.

整合会**归档原始记忆**, 这是全部记忆维护任务里唯一有实质破坏性的一步。所以在
开启它之前必须先有撤销手段 —— 不是"出问题再想办法", 那时候要面对的是散落在两张
表里的几百行, 而且没人记得哪些属于哪个簇。

依据是 `consolidated_into` changelog: 每条被归档的原行都有一条记录, new_value
指向摘要 ID。审计与归档在同一事务提交；撤销会先重建被回收的向量。

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

    from app.services.memory.lifecycle.capacity import restore_consolidated_digest

    result = await restore_consolidated_digest(digest_id)
    print(f"      取回 {result['restored']} 条（已重建向量）")
    return (result["found"], result["restored"])

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
