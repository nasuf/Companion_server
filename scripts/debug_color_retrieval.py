"""调试脚本: 测试 "你喜欢什么颜色啊" 这条 query 对 AI 颜色记忆的向量召回.

运行:
    uv run python scripts/debug_color_retrieval.py [agent_id]

如果不传 agent_id, 会列出所有 active agent 让你选.
"""

from __future__ import annotations

import asyncio
import sys

from app.db import db, connect_db, disconnect_db
from app.services.llm.models import get_embedding_model
from app.services.memory.normalization import cosine_similarity


QUERY = "你喜欢什么颜色啊"


async def list_agents() -> None:
    rows = await db.aiagent.find_many(where={"status": "active"})
    print(f"\n{'agent_id':<40} {'name':<20} {'workspace':<40}")
    print("-" * 100)
    for a in rows:
        ws = await db.chatworkspace.find_first(
            where={"agentId": a.id, "status": "active"}
        )
        print(f"{a.id:<40} {a.name:<20} {(ws.id if ws else '-'):<40}")


async def main(agent_id: str) -> None:
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        print(f"agent {agent_id} not found")
        return
    workspace = await db.chatworkspace.find_first(
        where={"agentId": agent_id, "status": "active"}
    )
    if not workspace:
        print(f"no active workspace for agent {agent_id}")
        return

    print(f"agent: {agent.name} ({agent_id})")
    print(f"workspace: {workspace.id}")
    print(f"user_id: {agent.userId}")

    # 1. 拉所有 AI 颜色记忆
    rows = await db.query_raw(
        """
        SELECT m.id, m.summary, m.importance, m.sub_category,
               me.embedding::text AS emb_text
        FROM memories_ai m
        JOIN memory_embeddings me ON me.memory_id = m.id
        WHERE m.user_id = $1 AND m.workspace_id = $2
          AND m.sub_category IN ('审美爱好', '饮食喜好', '生活习惯')
        ORDER BY m.importance DESC
        LIMIT 50
        """,
        agent.userId, workspace.id,
    )
    print(f"\n找到 {len(rows)} 条 AI 偏好记忆 (审美/饮食/生活习惯)\n")

    if not rows:
        print("⚠️  没有偏好记忆 — agent 可能没生成完")
        return

    # 2. embed 用户查询
    model = get_embedding_model()
    [query_vec] = await model.aembed_documents([QUERY])
    print(f"query: '{QUERY}'")
    print(f"embedding model: {type(model).__name__}, dim={len(query_vec)}")

    # 3. 算余弦相似度
    scored: list[tuple[float, str, str]] = []
    for r in rows:
        emb_text = r["emb_text"]
        # pgvector text format: '[0.1,0.2,...]'
        vec = [float(x) for x in emb_text.strip("[]").split(",")]
        sim = cosine_similarity(query_vec, vec)
        scored.append((sim, str(r["sub_category"]), str(r["summary"])))

    scored.sort(reverse=True)

    print(f"\n{'sim':<8} {'sub':<10} summary")
    print("-" * 100)
    for sim, sub, summary in scored[:20]:
        marker = "✓" if sim >= 0.50 else "✗"
        print(f"{sim:<8.3f} {marker} {sub:<10} {summary[:80]}")

    above = sum(1 for s, _, _ in scored if s >= 0.50)
    print(f"\n>= 0.50 (检索阈值) 的记忆: {above} 条")
    if above == 0:
        print("\n🔴 问题确认: 向量模型没把 query 和任何颜色记忆关联到 0.50 以上.")
        print("   → 这是 bge-m3 对短文本的局限. 检索时会全部漏召, AI 没有相关记忆 → 自由发挥.")
    elif above < 3:
        print("\n🟡 召回偏少, 但有命中. 可能是 prompt 里相关度没高到注入合适数量.")
    else:
        print("\n🟢 向量召回正常, 问题不在 embedding. 看其他环节 (relevance / inject).")


if __name__ == "__main__":
    async def _run():
        await connect_db()
        try:
            if len(sys.argv) < 2:
                await list_agents()
                print("\n使用: uv run python scripts/debug_color_retrieval.py <agent_id>")
            else:
                await main(sys.argv[1])
        finally:
            await disconnect_db()

    asyncio.run(_run())
