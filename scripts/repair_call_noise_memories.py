"""一次性归档通话转写抽出的噪声记忆 (VOICE01.04)。先出计划, 人工确认后按计划执行。

    PYTHONPATH=/app python scripts/repair_call_noise_memories.py --plan /tmp/plan.json     # 只读
    PYTHONPATH=/app python scripts/repair_call_noise_memories.py --apply /tmp/plan.json    # 按计划写

只看来源批次里含通话转写 (messages.metadata.channel = realtime_voice) 的已写入记忆, 用与新录入
相同的规则 (app/services/memory/recording/call_noise.py) 判定噪声。归档可恢复, 不删除;
执行时逐条核对内容哈希, 计划生成后内容变过或已归档的行跳过。输出只有计数; 计划文件含记忆原文,
只放受控目录。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORMAT = "call-noise-repair-v1"


def version(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


async def build_plan(database) -> dict:
    from app.services.memory.recording.call_noise import CALL_CHANNEL, call_noise

    rows = await database.query_raw(
        """WITH call_batches AS (
             SELECT DISTINCT s.batch_id FROM memory_extraction_sources s JOIN messages m ON m.id = s.message_id
             WHERE m.metadata->>'channel' = $1
           ), written AS (
             SELECT r.memory_id, b.side FROM memory_write_receipts r JOIN memory_extraction_batches b ON b.id = r.batch_id
             WHERE r.batch_id IN (SELECT batch_id FROM call_batches) AND r.status = 'written' AND r.memory_id IS NOT NULL
           )
           SELECT 'user' AS side, m.id, m.user_id, m.content FROM memories_user m
             WHERE NOT m.is_archived AND m.id IN (SELECT memory_id FROM written WHERE side = 'user')
           UNION ALL
           SELECT 'ai', m.id, m.user_id, m.content FROM memories_ai m
             WHERE NOT m.is_archived AND m.id IN (SELECT memory_id FROM written WHERE side = 'ai')""",
        CALL_CHANNEL,
    )
    archive, kept = [], 0
    for row in rows:
        reason = call_noise(row["side"], row["content"])
        if not reason:
            kept += 1
            continue
        archive.append(dict(side=row["side"], memory_id=row["id"], user_hash=user_hash(row["user_id"]),
                            reason=reason, content_version=version(row["content"]), content=row["content"]))
    return {"format": FORMAT, "generated_at": datetime.now(UTC).isoformat(), "kept": kept,
            "archive": sorted(archive, key=lambda x: (x["user_hash"], x["reason"], x["memory_id"]))}


async def apply_plan(database, plan: dict) -> dict:
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.persistence import after_memory_changelog

    if plan.get("format") != FORMAT:
        raise ValueError("unknown_plan_format")
    stats = {"archived": 0, "archive_skipped_changed": 0}
    for item in plan["archive"]:
        if item["side"] not in ("user", "ai") or item["reason"] not in {
                "call_mechanics", "question_not_fact", "momentary_state"}:
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
                                        "task": "VOICE01.04"}, ensure_ascii=False)})
        stats["archived"] += 1
        await memory_repo.invalidate_scope(row["user_id"], row["workspace_id"])
        await after_memory_changelog(user_id=row["user_id"], memory_id=item["memory_id"],
                                     operation="repair_archive", workspace_id=row["workspace_id"])
    return stats


def summary(plan: dict) -> dict:
    reasons: dict[str, int] = {}
    for item in plan["archive"]:
        key = f'{item["side"]}:{item["reason"]}'
        reasons[key] = reasons.get(key, 0) + 1
    return {"archive": len(plan["archive"]), "archive_by_reason": reasons, "kept": plan["kept"],
            "users": len({i["user_hash"] for i in plan["archive"]})}


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
