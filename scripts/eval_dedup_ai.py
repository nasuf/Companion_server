"""AI 侧 dedup 阈值实测 — 跑 paraphrase 重复落库的实测 cosine 表.

针对 conversation c0e6de3b-20fa-463c-8d12-e1fb108bca44 (用户报告的复读 bug):
1. 拉该 workspace 全部 L1 + L2 memories_ai
2. 对每条 L2 ai, 在 L1 ai + 其他 L2 ai 里找 top-1 cosine match
3. 输出 (L2 摘要, top-1 匹配, cosine, 是否人眼判定为同一事实) 表格

embeddings 已在 memory_embeddings 表里, 直接 SQL pgvector 算余弦, 不重新调 Ollama.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any

from app.db import connect_db, disconnect_db, db


CONVERSATION_ID = "c0e6de3b-20fa-463c-8d12-e1fb108bca44"


async def find_workspace() -> str:
    rows = await db.query_raw(
        'SELECT workspace_id FROM conversations WHERE id = $1',
        CONVERSATION_ID,
    )
    if not rows:
        print(f"conversation {CONVERSATION_ID} not found", file=sys.stderr)
        sys.exit(1)
    return rows[0]["workspace_id"]


async def fetch_l1_ai(workspace_id: str) -> list[dict[str, Any]]:
    return await db.query_raw(
        """
        SELECT m.id, m.summary, m.content, m.main_category, m.sub_category,
               m.created_at, me.embedding::text AS emb
        FROM memories_ai m
        JOIN memory_embeddings me ON me.memory_id = m.id
        WHERE m.workspace_id = $1 AND m.level = 1 AND m.is_archived = false
        ORDER BY m.created_at ASC
        """,
        workspace_id,
    )


async def fetch_l2_ai(workspace_id: str) -> list[dict[str, Any]]:
    return await db.query_raw(
        """
        SELECT m.id, m.summary, m.content, m.main_category, m.sub_category,
               m.created_at
        FROM memories_ai m
        WHERE m.workspace_id = $1 AND m.level = 2 AND m.is_archived = false
        ORDER BY m.created_at ASC
        """,
        workspace_id,
    )


async def top1_match(memory_id: str, workspace_id: str) -> dict[str, Any] | None:
    """对给定 L2 memory, 在同 workspace 的 L1 ai + 其他 L2 ai 里查 cosine 最高的.

    用 pgvector <=> (cosine distance) 算, 1 - distance = similarity.
    """
    rows = await db.query_raw(
        """
        WITH target AS (
            SELECT embedding FROM memory_embeddings WHERE memory_id = $1
        )
        SELECT m.id, m.summary, m.level, m.main_category, m.sub_category,
               1 - (me.embedding <=> (SELECT embedding FROM target)) AS sim
        FROM memories_ai m
        JOIN memory_embeddings me ON me.memory_id = m.id
        WHERE m.workspace_id = $2
          AND m.is_archived = false
          AND m.id != $1
        ORDER BY me.embedding <=> (SELECT embedding FROM target)
        LIMIT 1
        """,
        memory_id, workspace_id,
    )
    return rows[0] if rows else None


def short(s: str | None, n: int = 60) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


async def main() -> None:
    await connect_db()
    try:
        workspace_id = await find_workspace()
        print(f"workspace_id = {workspace_id}\n")

        l1 = await fetch_l1_ai(workspace_id)
        l2 = await fetch_l2_ai(workspace_id)
        print(f"L1 ai: {len(l1)} 条 | L2 ai: {len(l2)} 条\n")

        print("=" * 120)
        print("L1 ai 全部")
        print("=" * 120)
        for m in l1:
            print(f"  [{m['main_category']}/{m['sub_category']}] {short(m['summary'], 90)}")

        print("\n" + "=" * 120)
        print("L2 ai → top-1 cosine match (any level, any source=ai)")
        print("=" * 120)
        print(f"{'sim':>6}  {'目标 L2 摘要':<60}  →  {'top-1 命中 (level + 类别)':<70}")
        print("-" * 140)

        results: list[tuple[float, str, str, str]] = []
        for m in l2:
            match = await top1_match(m["id"], workspace_id)
            if not match:
                continue
            sim = float(match["sim"])
            l2_text = f"{short(m['summary'], 60)}"
            match_text = (
                f"L{match['level']} {match['main_category']}/{match['sub_category']}: "
                f"{short(match['summary'], 60)}"
            )
            results.append((sim, l2_text, match_text, m["id"]))

        # 按 sim 倒序看, 高 sim 在上
        results.sort(reverse=True)
        for sim, l2_text, match_text, _ in results:
            marker = "  HIT" if sim >= 0.9 else ("  ~~ " if sim >= 0.85 else "  ?  ")
            print(f"{sim:>6.3f}{marker} {l2_text:<60}  →  {match_text}")

        print("\n" + "=" * 120)
        print("分布统计")
        print("=" * 120)
        buckets = {">=0.95": 0, "0.90-0.95": 0, "0.85-0.90": 0, "0.80-0.85": 0, "0.75-0.80": 0, "<0.75": 0}
        for sim, *_ in results:
            if sim >= 0.95:
                buckets[">=0.95"] += 1
            elif sim >= 0.90:
                buckets["0.90-0.95"] += 1
            elif sim >= 0.85:
                buckets["0.85-0.90"] += 1
            elif sim >= 0.80:
                buckets["0.80-0.85"] += 1
            elif sim >= 0.75:
                buckets["0.75-0.80"] += 1
            else:
                buckets["<0.75"] += 1
        for k, v in buckets.items():
            print(f"  {k:>12}: {v} 条")

        # 按当前 0.9 阈值会被去重, 0.85 会被去重的数量对比
        catch_at_090 = sum(1 for s, *_ in results if s >= 0.9)
        catch_at_085 = sum(1 for s, *_ in results if s >= 0.85)
        catch_at_080 = sum(1 for s, *_ in results if s >= 0.80)
        print(f"\n  当前 0.90 阈值能拦截: {catch_at_090} / {len(results)} 条")
        print(f"  降到 0.85 能拦截:    {catch_at_085} / {len(results)} 条")
        print(f"  降到 0.80 能拦截:    {catch_at_080} / {len(results)} 条")

    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())
