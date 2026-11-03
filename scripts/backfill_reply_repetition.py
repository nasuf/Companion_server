"""一次性回填主回复重复表述打分 (CHAT01.10.1), 并重算对应日子的统计派生表。

用法 (生产容器内): PYTHONPATH=/app python scripts/backfill_reply_repetition.py [--days 30] [--dry-run]

只处理还没有当前版本完整打分的日子 (从近到远), 重跑会接着做没做完的。会调用 DashScope
向量与评判模型 (30 天约 3 千条气泡、几百次评判, 费用在几毛钱量级), 不写消息原文。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta


async def main(days: int, dry_run: bool) -> dict:
    from app.db import db
    from app.services import runtime_config
    from app.services.effect import daily_facts, definitions, repetition

    await db.connect()
    try:
        await runtime_config.ensure_loaded()
        yesterday = definitions.local_today() - timedelta(days=1)
        pending = await repetition.days_needing_scores(yesterday - timedelta(days=days - 1), yesterday)
        pending = sorted(pending, reverse=True)
        if dry_run:
            return {"pending": [d.isoformat() for d in pending], "dry_run": True}
        scored, failed = [], []
        for day in pending:
            try:
                result = await repetition.score_day(day)
            except Exception as exc:  # 一天失败不影响其余日子, 也不耽误已打完的日子重算
                failed.append(day.isoformat())
                print(json.dumps({"day": day.isoformat(), "error": type(exc).__name__}))
                continue
            print(json.dumps(result, ensure_ascii=False))
            scored.append(day)
        recomputed = await daily_facts.compute_days(scored)
        return {"scored": [d.isoformat() for d in scored], "failed": failed, "daily_facts": recomputed}
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(main(args.days, args.dry_run)), ensure_ascii=False))
