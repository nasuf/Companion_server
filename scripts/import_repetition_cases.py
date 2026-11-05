"""一次性把人工复核过的复读门禁用例导入后台库 (CHAT01.10.2)。用例不进公开仓库。

用法 (生产容器内): PYTHONPATH=/app python scripts/import_repetition_cases.py \\
    --case-set rr-20261104 --cases /tmp/cases.json --contexts /tmp/contexts.json [--dry-run]

cases.json 为脱敏用例 (每例须是人工复核通过的), contexts.json 为同一批用例当时的作息状态 / 话题 /
延迟 (已脱敏), 键为用例编号。同一个 case_set 只能导入一次; 只保留回放与打分要用的字段。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

KEEP = ("id", "kind", "situation", "tags", "user", "history_rows", "message", "judge_user_text",
        "previous_ai_turns", "current_ids", "now", "received_at")
CONTEXT = ("ai_status", "topic", "delay")


def prepare(cases: list[dict], contexts: dict) -> list[dict]:
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case id")
    missing = [i for i in ids if i not in contexts]
    if missing:
        raise ValueError(f"context missing for {len(missing)} cases, e.g. {missing[0]}")
    for case in cases:
        absent = [k for k in KEEP if k not in case]
        if absent:
            raise ValueError(f"{case['id']}: missing {absent}")
    return [{"case": {k: c[k] for k in KEEP}, "context": {k: contexts[c["id"]][k] for k in CONTEXT
                                                           if contexts[c["id"]].get(k) is not None}}
            for c in cases]


async def main(args) -> dict:
    from datetime import timedelta

    from app.db import db
    from app.services.quality import repetition_gate

    items = prepare(json.loads(args.cases.read_text()), json.loads(args.contexts.read_text()))
    if args.dry_run:
        return {"case_set": args.case_set, "cases": len(items), "dry_run": True}
    await db.connect()
    try:
        async with db.tx(timeout=timedelta(seconds=60)) as tx:
            loaded = await repetition_gate.import_cases(tx, args.case_set, items)
    finally:
        await db.disconnect()
    return {"case_set": args.case_set, "cases": loaded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(asyncio.run(main(parser.parse_args())), ensure_ascii=False))
