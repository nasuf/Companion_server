"""导出记忆的生命周期状态, 供离线推演使用.

推演要回答的是"改了分层规则之后, 有用的记忆还留不留得住"。这需要每条记忆的
层级、分数、以及**最后一次被用到的时间** —— 衰减完全由后者驱动。

访问时间取自 changelog 的 access 记录, 缺失时退回 created_at, 与
l2_dynamics._adjust_side 的取值口径一致 —— 推演和生产用不同口径的话, 推演结论
就不适用于生产。

用法 (在生产容器内):
    python export_lifecycle_snapshot.py /tmp/lifecycle.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

from app.db import db


async def main() -> None:
    out_path = sys.argv[1]
    await db.connect()

    rows = await db.query_raw(
        """
        SELECT m.id, m.content, m.level, m.importance, m.current_score,
               m.mention_count, COALESCE(m.provenance, 'init') AS provenance,
               m.main_category, m.sub_category, m.created_at,
               'ai' AS source,
               (SELECT MAX(cl.created_at) FROM memory_changelogs cl
                 WHERE cl.memory_id = m.id AND cl.operation = 'access') AS last_access
        FROM memories_ai m WHERE m.is_archived = false
        UNION ALL
        SELECT m.id, m.content, m.level, m.importance, m.current_score,
               m.mention_count, COALESCE(m.provenance, 'init') AS provenance,
               m.main_category, m.sub_category, m.created_at,
               'user' AS source,
               (SELECT MAX(cl.created_at) FROM memory_changelogs cl
                 WHERE cl.memory_id = m.id AND cl.operation = 'access') AS last_access
        FROM memories_user m WHERE m.is_archived = false
        """
    )
    await db.disconnect()

    def _iso(value) -> str | None:
        return value.isoformat() if hasattr(value, "isoformat") else value

    snapshot = [
        {
            "id": r["id"],
            "content": r["content"],
            "level": r["level"],
            "importance": float(r["importance"] or 0),
            "current_score": (
                float(r["current_score"]) if r["current_score"] is not None else None
            ),
            "mention_count": int(r["mention_count"] or 0),
            "provenance": r["provenance"],
            "main_category": r["main_category"],
            "sub_category": r["sub_category"],
            "created_at": _iso(r["created_at"]),
            "last_access": _iso(r["last_access"]),
            "source": r["source"],
        }
        for r in rows
    ]
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "memories": snapshot,
    }
    open(out_path, "w").write(json.dumps(payload, ensure_ascii=False))

    from collections import Counter
    print(f"导出 {len(snapshot)} 条 → {out_path}")
    print("  层级:", dict(sorted(Counter(m["level"] for m in snapshot).items())))
    print("  有访问记录的:", sum(1 for m in snapshot if m["last_access"]))
    print("  有 current_score 的:", sum(1 for m in snapshot if m["current_score"] is not None))


if __name__ == "__main__":
    asyncio.run(main())
