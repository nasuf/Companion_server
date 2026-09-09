"""给存量事件类记忆 (生活/情绪) 补 occur_time。

背景 (2026-08 生产诊断): 事件类记忆里 user 侧只有 6-7% 填了 occur_time —— 抽取
产出的是剥掉时间词的摘要, 规则引擎抓不到。真正的杠杆是 statement_time 兜底
(event ≈ 说到它的时刻, Beyond Dialogue Time arXiv:2601.07468)。写入侧已在
store_memory 收口接了 resolve_occur_time; 这个脚本把存量 ~7400 行补上。

复用 store_memory 用的同一个 resolve_occur_time —— 存量与增量走同一份判定, 不会
出现"新记忆一套规则、老记忆另一套"。解析基准 = 该行的 statement_time (缺则
created_at), 即相对**说话时刻**解析, 不是相对现在 (否则 "明天面试" 会被错标)。

约定 (跟 scripts/split_oversized_memories.py 一致):
- 默认 dry-run, 只统计+抽样, 不写库
- --apply 才写; UPDATE 带 `occur_time IS NULL` 条件防并发覆盖
- 写库前把 (id, table) 存进 journal JSON, --rollback <journal> 可完整还原
  (回滚只把这些 id 的 occur_time 重新置 NULL —— 它们本来就是 NULL)

用法:
    .venv/bin/python -m scripts.backfill_occur_time                 # dry-run
    .venv/bin/python -m scripts.backfill_occur_time --apply --journal /tmp/occ.json
    .venv/bin/python -m scripts.backfill_occur_time --rollback /tmp/occ.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.db import connect_db, disconnect_db, db
from app.services.schedule_domain.time_parser import resolve_occur_time

DATEABLE = ("生活", "情绪")
TABLES = ("memories_user", "memories_ai")


def _as_dt(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(v, datetime) and v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v


async def _candidates(table: str) -> list[dict]:
    return await db.query_raw(
        f"SELECT id, content, statement_time, created_at, main_category, "
        f"sub_category, provenance FROM {table} "
        f"WHERE is_archived=false AND main_category=ANY($1) AND occur_time IS NULL",
        list(DATEABLE),
    )


def _resolve_for_row(r: dict) -> tuple[datetime | None, str]:
    """返回 (occur_time, tier). tier ∈ tierA_explicit / tierB_statement / none."""
    base = _as_dt(r["statement_time"]) or _as_dt(r["created_at"])
    if base is None:
        return None, "none"  # 不该发生 (created_at 恒有), 保险
    occ = resolve_occur_time(
        r["content"] or "", statement_time=base,
        main_category=r["main_category"], sub_category=r["sub_category"],
        provenance=r["provenance"],
    )
    if occ is None:
        return None, "none"
    # 与说话日不同 = content 里有显性日期被解析出来了 (Tier A); 否则是兜底 (Tier B)
    tier = "tierA_explicit" if occ.date() != base.date() else "tierB_statement"
    return occ, tier


async def run(apply: bool, journal_path: Path | None) -> None:
    await connect_db()
    try:
        journal: list[dict] = []
        for table in TABLES:
            rows = await _candidates(table)
            tiers: dict[str, int] = defaultdict(int)
            updates: list[tuple[str, datetime]] = []
            samples: list[str] = []
            for r in rows:
                occ, tier = _resolve_for_row(r)
                tiers[tier] += 1
                if occ is None:
                    continue
                updates.append((r["id"], occ))
                if tier == "tierA_explicit" and len(samples) < 8:
                    base = _as_dt(r["statement_time"]) or _as_dt(r["created_at"])
                    bd = base.date() if base else "?"
                    samples.append(
                        f"    [{bd}→{occ.date()}] 「{(r['content'] or '')[:44]}」")
            print(f"\n[{table}] 候选 {len(rows)} 条")
            print(f"   Tier A 显性日期解析 : {tiers['tierA_explicit']:>5}")
            print(f"   Tier B statement兜底: {tiers['tierB_statement']:>5}")
            print(f"   放弃(远过去/无基准) : {tiers['none']:>5}")
            print(f"   → 将补 occur_time   : {len(updates)} 条")
            if samples:
                print("   Tier A 样例 (说话日→解析出的事件日):")
                print("\n".join(samples))

            if apply and updates:
                for mid, occ in updates:
                    journal.append({"table": table, "id": mid})
                # 用 ORM update_many 而不是 raw SQL: occur_time 列是 timestamp
                # without time zone, 生产 create 路径靠 Prisma 处理 aware→naive 的
                # 时区转换 (create_data["occurTime"]=occur_time)。走同一条路, backfill
                # 的值才和 pipeline 写的值格式一致 —— raw SQL 手动 cast 极易把
                # +08 日历日期错移一天。where 带 occurTime=None 防并发覆盖。
                model = db.usermemory if table == "memories_user" else db.aimemory
                n = 0
                for mid, occ in updates:
                    n += await model.update_many(
                        where={"id": mid, "occurTime": None},
                        data={"occurTime": occ},
                    )
                print(f"   ✓ 已写入 {n} 条")

        if apply and journal_path:
            journal_path.write_text(json.dumps(
                {"created": datetime.now(timezone.utc).isoformat(), "entries": journal},
                ensure_ascii=False, indent=2))
            print(f"\njournal → {journal_path} ({len(journal)} 条, 可 --rollback 还原)")
        if not apply:
            print("\n(dry-run, 未写库。加 --apply --journal <path> 才写。)")
    finally:
        await disconnect_db()


async def rollback(journal_path: Path) -> None:
    await connect_db()
    try:
        data = json.loads(journal_path.read_text())
        by_table: dict[str, list[str]] = defaultdict(list)
        for e in data["entries"]:
            by_table[e["table"]].append(e["id"])
        for table, ids in by_table.items():
            model = db.usermemory if table == "memories_user" else db.aimemory
            n = 0
            for mid in ids:
                n += await model.update_many(
                    where={"id": mid}, data={"occurTime": None})
            print(f"[{table}] 回滚 {n}/{len(ids)} 条 occur_time → NULL")
    finally:
        await disconnect_db()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="真正写库 (默认 dry-run)")
    ap.add_argument("--journal", type=Path, help="写库时把改动 id 存到这里, 供 --rollback")
    ap.add_argument("--rollback", type=Path, help="按 journal 把 occur_time 还原成 NULL")
    args = ap.parse_args()
    if args.rollback:
        asyncio.run(rollback(args.rollback))
    else:
        if args.apply and not args.journal:
            ap.error("--apply 必须同时给 --journal <path> (否则没法回滚)")
        asyncio.run(run(args.apply, args.journal))


if __name__ == "__main__":
    main()
