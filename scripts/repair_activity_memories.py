"""一次性修复线下活动相关的存量记忆 (VOICE01.10)。先出计划, 人工确认后按计划执行。

    PYTHONPATH=/app python scripts/repair_activity_memories.py --plan /tmp/plan.json     # 只读
    PYTHONPATH=/app python scripts/repair_activity_memories.py --apply /tmp/plan.json    # 按计划写

计划里两类动作:
- archive: 旧的活动记忆归档 (可恢复, 不删除)。
  * 系统替用户插的活动卡片被抽成了"用户提及活动卡片：X" —— 不是用户说的话;
  * 旧事件记忆 (忽略/到达/收好/接受/完成), 状态已由新的活动状态记忆取代, 而且可能已过时;
  * AI 日常总结里把"推荐给用户的卡片"写成"用户分享了…卡片" —— 方向写反了。
  只处理内容里出现该用户某个活动标题或地点的行; 用户自己表达的看法和 AI 自己的感受不动。
- sync: 用户回应过的活动, 按活动当前状态写一对活动状态记忆 (activity_memory.sync)。

执行时逐条核对内容哈希, 计划生成后内容变过、已归档的行一律跳过, 不会误伤。
输出只有计数; 计划文件含记忆原文, 只放受控目录。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CARD = re.compile(r"^用户(?:发送|提及|提到|分享)(?:了)?活动卡片")
STATUS_EVENT = re.compile(
    r"暂时忽略了线下活动推荐|确认到达线下活动地点|收好了(?:该活动|线下活动)的回忆|"
    r"(?:重新)?接受了线下活动推荐|完成了线下活动「")
WRONG_DIRECTION = re.compile(r"用户(?:分享|发送)(?:了)?.{0,30}活动卡片")


def classify(side: str, provenance: str | None, content: str) -> str | None:
    if side == "user" and provenance in (None, "user_stated"):
        if CARD.search(content):
            return "activity_card_not_user_words"
        if STATUS_EVENT.search(content):
            return "superseded_status_event"
    if side == "ai" and provenance == "daily_summary" and WRONG_DIRECTION.search(content):
        return "wrong_direction"
    return None


def version(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


async def build_plan(database) -> dict:
    from app.services.offline.activity_memory import render, responded

    activities = [dict(r) for r in await database.query_raw(
        """SELECT a.id, a.user_id, a.workspace_id, a.title, a.location_name, a.city, a.status, a.reached,
                  a.accepted_at, a.ignored_at, a.arrival_confirmed_at, a.completed_at, a.created_at, a.updated_at,
                  g.name AS agent_name
           FROM offline_activity_recommendations a LEFT JOIN ai_agents g ON g.id = a.agent_id
           ORDER BY a.user_id, a.created_at""")]
    archive: dict[str, dict] = {}
    for side, table in (("user", "memories_user"), ("ai", "memories_ai")):
        rows = await database.query_raw(
            f"""SELECT DISTINCT m.id, m.user_id, m.content, m.provenance
                FROM {table} m JOIN offline_activity_recommendations a ON a.user_id = m.user_id
                WHERE NOT m.is_archived AND COALESCE(m.provenance, '') <> 'activity_record'
                  AND (strpos(m.content, a.title) > 0
                       OR (length(COALESCE(a.location_name, '')) >= 4 AND strpos(m.content, a.location_name) > 0))""")
        for row in rows:
            reason = classify(side, row["provenance"], row["content"])
            if reason:
                archive[row["id"]] = dict(side=side, memory_id=row["id"], user_hash=user_hash(row["user_id"]),
                                          reason=reason, content_version=version(row["content"]),
                                          content=row["content"])
    sync = []
    for activity in activities:
        texts = render(activity, activity.get("agent_name")) if responded(activity) else None
        if texts:
            sync.append(dict(activity_id=activity["id"], user_id=activity["user_id"],
                             user_hash=user_hash(activity["user_id"]), status=activity["status"],
                             reached=bool(activity["reached"]), user=texts["user"], ai=texts["ai"]))
    return {"format": "activity-memory-repair-v1", "generated_at": datetime.now(UTC).isoformat(),
            "archive": sorted(archive.values(), key=lambda x: (x["user_hash"], x["reason"], x["memory_id"])),
            "sync": sync}


async def apply_plan(database, plan: dict) -> dict:
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.persistence import after_memory_changelog
    from app.services.offline import activity_memory

    if plan.get("format") != "activity-memory-repair-v1":
        raise ValueError("unknown_plan_format")
    stats = {"archived": 0, "archive_skipped_changed": 0, "sync": {}}
    for item in plan["archive"]:
        if item["side"] not in ("user", "ai") or item["reason"] not in {
                "activity_card_not_user_words", "superseded_status_event", "wrong_direction"}:
            raise ValueError("invalid_plan_item")
        table = "memories_ai" if item["side"] == "ai" else "memories_user"
        async with database.tx() as tx:
            rows = await tx.query_raw(
                f"""UPDATE {table} SET is_archived = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1 AND NOT is_archived
                      AND encode(sha256(convert_to(content, 'UTF8')), 'hex') = $2
                    RETURNING user_id, workspace_id, content""",
                item["memory_id"], item["content_version"])
            if not rows:
                stats["archive_skipped_changed"] += 1
                continue
            row = rows[0]
            await tx.memorychangelog.create(data={
                "userId": row["user_id"], "workspaceId": row["workspace_id"], "memoryId": item["memory_id"],
                "operation": "repair_archive", "oldValue": row["content"],
                "newValue": json.dumps({"action": "archive_memory", "reason": item["reason"],
                                        "task": "VOICE01.10"}, ensure_ascii=False)})
        stats["archived"] += 1
        await memory_repo.invalidate_scope(row["user_id"], row["workspace_id"])
        await after_memory_changelog(user_id=row["user_id"], memory_id=item["memory_id"],
                                     operation="repair_archive", workspace_id=row["workspace_id"])
    for item in plan["sync"]:
        result = await activity_memory.sync(item["activity_id"], item["user_id"])
        stats["sync"][result] = stats["sync"].get(result, 0) + 1
    return stats


def summary(plan: dict) -> dict:
    reasons: dict[str, int] = {}
    for item in plan["archive"]:
        reasons[f'{item["side"]}:{item["reason"]}'] = reasons.get(f'{item["side"]}:{item["reason"]}', 0) + 1
    return {"archive": len(plan["archive"]), "archive_by_reason": reasons, "sync": len(plan["sync"]),
            "users": len({i["user_hash"] for i in plan["archive"]} | {i["user_hash"] for i in plan["sync"]})}


async def main(args) -> dict:
    from app.db import db

    await db.connect()
    try:
        if args.plan:
            plan = await build_plan(db)
            args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=1, default=str))
            args.plan.chmod(0o600)
            return {"mode": "plan", **summary(plan)}
        plan = json.loads(args.apply.read_text())
        return {"mode": "apply", **await apply_plan(db, plan)}
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path, help="只读: 生成修复计划")
    group.add_argument("--apply", type=Path, help="按已确认的计划执行")
    print(json.dumps(asyncio.run(main(parser.parse_args())), ensure_ascii=False))
