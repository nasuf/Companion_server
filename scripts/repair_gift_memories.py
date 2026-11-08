"""一次性修复 AI 寄出礼物的存量记忆 (B01.08)。先出计划, 人工确认后按计划执行。

    PYTHONPATH=/app python scripts/repair_gift_memories.py --plan /tmp/plan.json     # 只读
    PYTHONPATH=/app python scripts/repair_gift_memories.py --apply /tmp/plan.json    # 按计划写

旧写法每个事件追加一条 (寄出/签收/道谢/取消), 签收后"等签收再揭晓"仍在, 取消后和"在路上"
并存。计划里两类动作:
- archive: 旧 gift_memory 模板写出的行 (只认这几句模板, 用户送 AI 的礼物、用户自己说的话不动),
  归档可恢复, 不删除。
- sync: 用户被告知过的礼物, 按礼物当前状态写一对状态记忆 (gift_memory.sync)。
旧行所属用户已经没有任何礼物记录的 (礼物被清空过), 单列 orphan, 默认不处理。

执行时逐条核对内容哈希, 计划生成后内容变过、已归档的行一律跳过。
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

FORMAT = "gift-memory-repair-v1"
_DAY = r"^\d{4}-\d{2}-\d{2} "
# 旧 gift_memory.py 的五句模板 (52a9a8f2 起), 按侧和 provenance 区分
TEMPLATES = {
    ("ai", "ai_authored"): [
        ("superseded_sent", re.compile(_DAY + r"我给用户寄了一份礼物，想等用户签收时再揭晓")),
        ("superseded_received", re.compile(_DAY + r"用户签收了我寄的礼物：")),
        ("superseded_thanks", re.compile(_DAY + r"用户收到我送的.+后特意跟我道谢：「")),
        ("superseded_cancelled", re.compile(_DAY + r"我寄给用户的那份礼物.*中途取消了")),
    ],
    ("user", "user_stated"): [
        ("superseded_received", re.compile(_DAY + r"我收到并签收了.+寄来的礼物：")),
    ],
}
REASONS = {reason for rules in TEMPLATES.values() for reason, _ in rules}


def classify(side: str, provenance: str | None, content: str) -> str | None:
    for reason, pattern in TEMPLATES.get((side, provenance or ""), []):
        if pattern.search(content):
            return reason
    return None


def version(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


async def build_plan(database) -> dict:
    from app.services.offline.gift_memory import _load, render

    gifts = await database.query_raw("SELECT id, user_id FROM real_world_gifts ORDER BY user_id, created_at")
    with_gifts = {g["user_id"] for g in gifts}
    archive, orphan = [], []
    for side, table in (("user", "memories_user"), ("ai", "memories_ai")):
        rows = await database.query_raw(
            f"""SELECT id, user_id, content, provenance FROM {table}
                WHERE NOT is_archived AND sub_category = '馈赠'
                  AND COALESCE(provenance, '') IN ('ai_authored', 'user_stated')""")
        for row in rows:
            reason = classify(side, row["provenance"], row["content"])
            if not reason:
                continue
            item = dict(side=side, memory_id=row["id"], user_hash=user_hash(row["user_id"]), reason=reason,
                        content_version=version(row["content"]), content=row["content"])
            (archive if row["user_id"] in with_gifts else orphan).append(item)
    sync = []
    for g in gifts:
        state = await _load(database, g["id"], g["user_id"])
        texts = render(state) if state else None
        if texts:
            sync.append(dict(gift_id=g["id"], user_id=g["user_id"], user_hash=user_hash(g["user_id"]),
                             status=state["status"], user=texts["user"], ai=texts["ai"]))
    order = lambda x: (x["user_hash"], x["reason"], x["memory_id"])  # noqa: E731
    return {"format": FORMAT, "generated_at": datetime.now(UTC).isoformat(),
            "archive": sorted(archive, key=order), "orphan": sorted(orphan, key=order), "sync": sync}


async def apply_plan(database, plan: dict) -> dict:
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.persistence import after_memory_changelog
    from app.services.offline import gift_memory

    if plan.get("format") != FORMAT:
        raise ValueError("unknown_plan_format")
    stats = {"archived": 0, "archive_skipped_changed": 0, "sync": {}}
    for item in plan["archive"]:
        if item["side"] not in ("user", "ai") or item["reason"] not in REASONS:
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
                                        "task": "B01.08"}, ensure_ascii=False)})
        stats["archived"] += 1
        await memory_repo.invalidate_scope(row["user_id"], row["workspace_id"])
        await after_memory_changelog(user_id=row["user_id"], memory_id=item["memory_id"],
                                     operation="repair_archive", workspace_id=row["workspace_id"])
    for item in plan["sync"]:
        result = await gift_memory.sync(item["gift_id"], item["user_id"])
        stats["sync"][result] = stats["sync"].get(result, 0) + 1
    return stats


def summary(plan: dict) -> dict:
    def by_reason(items):
        out: dict[str, int] = {}
        for item in items:
            out[f'{item["side"]}:{item["reason"]}'] = out.get(f'{item["side"]}:{item["reason"]}', 0) + 1
        return out

    sync_status: dict[str, int] = {}
    for item in plan["sync"]:
        sync_status[item["status"]] = sync_status.get(item["status"], 0) + 1
    return {"archive": len(plan["archive"]), "archive_by_reason": by_reason(plan["archive"]),
            "orphan": len(plan["orphan"]), "orphan_by_reason": by_reason(plan["orphan"]),
            "sync": len(plan["sync"]), "sync_by_status": sync_status,
            "users": len({i["user_hash"] for i in plan["archive"] + plan["orphan"] + plan["sync"]})}


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
