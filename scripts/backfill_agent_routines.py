"""一次性补齐活跃 agent 的固定作息 (TIME12)。

用法 (生产容器内): PYTHONPATH=/app python scripts/backfill_agent_routines.py [--dry-run] [--limit N]

只给 routine 为空的 active agent 推导; 写入用和线上同一个"只在仍为空时写"的路径,
与连接预热/夜间任务并发也只会留下先写入的那份。不改当天作息 (用户下次连接时由
预热按需重排, 当天有作息调整的不动)。推导失败的只记日志, 下次生成作息时还会再试。
"""
from __future__ import annotations

import argparse
import asyncio
import json


async def main(dry_run: bool, limit: int) -> dict:
    from app.db import db
    from app.services import runtime_config
    from app.services.llm.models import get_utility_model
    from app.services.schedule_domain import routine as routine_mod

    await db.connect()
    try:
        await runtime_config.ensure_loaded()
        model = get_utility_model()
        print(json.dumps({"utility_model": getattr(model, "model_name", None) or getattr(model, "model", None)}))
        # 只取还没有作息的: 重跑时自然从没做完的接着做, 不会反复扫已经有的
        pending = await db.query_raw(
            "SELECT id FROM ai_agents WHERE status = 'active' AND (routine IS NULL OR routine = 'null'::jsonb) "
            "ORDER BY created_at ASC LIMIT $1", limit,
        )
        agents = await db.aiagent.find_many(where={"id": {"in": [r["id"] for r in pending]}},
                                            order={"createdAt": "asc"})
        result = {"pending": len(agents), "already": 0, "derived": 0, "not_persisted": 0,
                  "failed": 0, "dry_run": dry_run}
        if dry_run:
            return result
        for agent in agents:
            if await routine_mod.get_agent_routine(agent.id) is not None:
                result["already"] += 1  # 期间被连接预热写上了
                continue
            routine = await routine_mod.ensure_agent_routine(agent)
            if routine is None:
                result["failed"] += 1
                print(json.dumps({"agent": agent.id[:8], "status": "failed"}))
            elif await routine_mod.get_agent_routine(agent.id) is None:
                result["not_persisted"] += 1  # 推导出来了但没写进库, 下次重跑还会再试
                print(json.dumps({"agent": agent.id[:8], "status": "not_persisted"}))
            else:
                result["derived"] += 1
                print(json.dumps({"agent": agent.id[:8], "status": "derived", "usual": routine.usual_text()},
                                 ensure_ascii=False))
        return result
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(main(args.dry_run, args.limit)), ensure_ascii=False))
