"""一次性数据修复: 把历史上被错分到「身份/其他」的身份事实拉回正确子类.

背景见 app/services/memory/recording/identity_repair.py —— 抽取时的类目由 LLM
判定, 同一句式会给出不同结果. 该模块已在录入侧兜底, 但只对之后的抽取生效;
库里的历史行仍然是错的, 而它们正是最该永驻的事实 (姓名/性别/籍贯).

默认 dry-run, 只打印会改什么. 加 --apply 才写库.

    python -m scripts.repair_identity_memories                 # 预览
    python -m scripts.repair_identity_memories --apply         # 执行
    python -m scripts.repair_identity_memories --workspace X   # 限定范围

安全约束:

- 复用录入侧同一个 repair 函数, 不另写一套判定 —— 两边逻辑分叉是这类脚本最
  常见的坑.
- singleton 冲突不自动合并: 同一 (user, workspace) 下若已有正确的姓名行, 把
  另一行也改成姓名会造出两个姓名, 检索时无从取舍. 这类只报告, 交人判断.
- 每条改动写 changelog, 事后可追溯是脚本改的还是模型写的.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from app.db import db
from app.services.memory.recording.identity_repair import (
    repair_identity_classification,
)

# 与 pipeline 的分层推导保持一致 (importance → level).
_L1_THRESHOLD = 0.85
_L2_THRESHOLD = 0.50


def _level_for(importance: float) -> int:
    if importance >= _L1_THRESHOLD:
        return 1
    if importance >= _L2_THRESHOLD:
        return 2
    return 3


@dataclass
class Change:
    table: str
    row_id: str
    user_id: str
    workspace_id: str | None
    preview: str
    created_at: object
    old: tuple[str, str, float, int]
    new: tuple[str, str, float, int]
    reason: str
    blocked_by: str | None = None
    # singleton 冲突时占着位子的那些行 (id, content 预览, created_at)
    incumbents: tuple[tuple[str, str, object], ...] = ()


async def _collect(table: str, workspace: str | None) -> list[Change]:
    where = ["1=1"]
    params: list = []
    if workspace:
        params.append(workspace)
        where.append(f"workspace_id = ${len(params)}")
    rows = await db.query_raw(
        f"""
        SELECT id, user_id, workspace_id, content,
               main_category, sub_category, importance, level, created_at
        FROM {table}
        WHERE {' AND '.join(where)}
        ORDER BY created_at
        """,
        *params,
    )

    # 已有的 singleton 占位: (user, workspace, sub) → [(id, content 预览, created_at)]
    taken: dict[tuple[str, str, str], list[tuple[str, str, object]]] = {}
    for r in rows:
        if r["main_category"] == "身份":
            key = (r["user_id"], r["workspace_id"], r["sub_category"])
            taken.setdefault(key, []).append(
                (r["id"], r["content"] or "", r["created_at"])
            )

    changes: list[Change] = []
    for r in rows:
        main, sub, imp, reason = repair_identity_classification(
            summary=r["content"] or "",
            main_category=r["main_category"],
            sub_category=r["sub_category"],
            importance=float(r["importance"]),
        )
        if not reason:
            continue
        level = _level_for(imp)
        if (main, sub, imp, level) == (
            r["main_category"], r["sub_category"], float(r["importance"]), r["level"]
        ):
            continue

        blocked = None
        incumbents: tuple[tuple[str, str, object], ...] = ()
        if sub != r["sub_category"]:
            key = (r["user_id"], r["workspace_id"], sub)
            existing = taken.get(key) or []
            if existing:
                incumbents = tuple(existing)
                blocked = f"该用户已有 {len(existing)} 条 身份/{sub}"
            else:
                taken[key] = [(r["id"], r["content"] or "", r["created_at"])]

        changes.append(Change(
            table=table, row_id=r["id"], user_id=r["user_id"],
            workspace_id=r["workspace_id"], preview=(r["content"] or "")[:46],
            created_at=r["created_at"],
            old=(r["main_category"], r["sub_category"], float(r["importance"]), r["level"]),
            new=(main, sub, imp, level),
            reason=reason, blocked_by=blocked, incumbents=incumbents,
        ))
    return changes


async def _apply(change: Change) -> None:
    main, sub, imp, level = change.new
    await db.execute_raw(
        f"""
        UPDATE {change.table}
        SET main_category = $1, sub_category = $2, importance = $3, level = $4
        WHERE id = $5
        """,
        main, sub, imp, level, change.row_id,
    )
    old_main, old_sub, old_imp, old_level = change.old
    await _log(
        change,
        old=f"{old_main}/{old_sub} imp={old_imp:.2f} L{old_level}",
        new=f"{main}/{sub} imp={imp:.2f} L{level}",
    )


async def _log(change: Change, *, old: str, new: str, memory_id: str | None = None) -> None:
    """写 changelog. 列名按 memory_changelogs 真实 schema (old_value/new_value)."""
    import uuid

    await db.execute_raw(
        """
        INSERT INTO memory_changelogs
            (id, user_id, workspace_id, memory_id, operation,
             old_value, new_value, created_at)
        VALUES ($1, $2, $3, $4, 'update', $5, $6, now())
        """,
        str(uuid.uuid4()), change.user_id, change.workspace_id,
        memory_id or change.row_id,
        f"identity_repair script: {old}", f"{new} ({change.reason})",
    )


async def _demote_incumbent(change: Change, memory_id: str, preview: str) -> None:
    """把被顶下来的旧 singleton 降到 L2, 不删 —— 它仍是真实说过的话,
    只是不再是"当前答案"。跟矛盾处理里老 L1 降级的做法一致。"""
    await db.execute_raw(
        f"UPDATE {change.table} SET importance = 0.80, level = 2 WHERE id = $1",
        memory_id,
    )
    await _log(
        change, memory_id=memory_id,
        old="身份 singleton L1 (在位)",
        new=f"降为 L2, 让位给更晚的明确表述「{change.preview}」",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正写库 (默认只预览)")
    parser.add_argument("--workspace", help="只处理某个 workspace")
    parser.add_argument(
        "--resolve-singleton", action="store_true",
        help="冲突时让更晚的明确表述顶上, 把在位的旧行降为 L2",
    )
    args = parser.parse_args()

    await db.connect()
    try:
        all_changes: list[Change] = []
        for table in ("memories_user", "memories_ai"):
            all_changes += await _collect(table, args.workspace)

        doable = [c for c in all_changes if not c.blocked_by]
        blocked = [c for c in all_changes if c.blocked_by]

        print(f"命中 {len(all_changes)} 条, 可改 {len(doable)}, 需人工 {len(blocked)}\n")
        for c in doable:
            om, os_, oi, ol = c.old
            nm, ns, ni, nl = c.new
            print(f"  {c.table} {c.row_id[:8]} | {c.preview}")
            print(f"    {om}/{os_} imp={oi:.2f} L{ol}  →  {nm}/{ns} imp={ni:.2f} L{nl}")
        resolvable = [
            c for c in blocked
            if all(c.created_at > inc[2] for inc in c.incumbents)
        ]
        if blocked:
            print("\n singleton 冲突:")
            for c in blocked:
                print(f"  {c.table} {c.row_id[:8]} | {c.preview}")
                print(f"    {c.blocked_by}")
                for _mid, msum, mts in c.incumbents:
                    print(f"      在位: {str(mts)[5:16]} 「{msum[:40]}」")
                if c in resolvable:
                    print("      → 本行更晚, --resolve-singleton 可让它顶上")
                else:
                    print("      → 本行更早, 不动 (以在位的为准)")

        if not args.apply:
            print("\n(dry-run; 加 --apply 才写库)")
            return
        if not doable:
            print("\n无可执行改动")
            return
        for c in doable:
            await _apply(c)
        print(f"\n已更新 {len(doable)} 条, 并写入 changelog")

        if args.resolve_singleton and resolvable:
            for c in resolvable:
                for mid, msum, _ts in c.incumbents:
                    await _demote_incumbent(c, mid, msum)
                await _apply(c)
            print(f"已解决 {len(resolvable)} 处 singleton 冲突 (旧行降 L2, 未删除)")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
