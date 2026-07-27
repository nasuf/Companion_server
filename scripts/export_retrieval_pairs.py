"""导出真实 (用户消息, 召回记忆, 相似度) 配对, 用于标定相似度阈值.

生产阈值是 0.50 (retrieval/hybrid.py _SIMILARITY_THRESHOLD), 由 spec 的 0.7
下调而来, 理由是 bge-m3 对中文短文本召回弱. 但真实流量上 97% 的消息都能在
0.50 上匹配到东西, 包括 "Hello, hello" 命中 "我是汉族" 这种 —— 说明代价是精度.

这个脚本不做判断, 只把配对捞全, 判定交给下一步的评审. 不设阈值取 top-k, 这样
同一条消息在各个相似度档位上的候选都在, 才能算出精度随阈值的变化.
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.db import db
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.memory.storage.embedding import generate_embedding

TOP_K = 10


async def _scope(message: str) -> tuple[str, str] | None:
    rows = await db.query_raw(
        "SELECT conversation_id FROM messages WHERE role = $1 AND content = $2 LIMIT 1",
        "user", message,
    )
    if not rows:
        return None
    conv = await db.conversation.find_unique(where={"id": rows[0]["conversation_id"]})
    if not conv or not conv.userId:
        return None
    return conv.userId, conv.workspaceId


async def main() -> None:
    await db.connect()
    messages = json.loads(open(sys.argv[1]).read())
    out = []
    for message in messages:
        scope = await _scope(message)
        if scope is None:
            continue
        user_id, workspace_id = scope
        try:
            emb = await generate_embedding(message)
            hits = await search_by_embedding(
                emb, user_id, top_k=TOP_K, workspace_id=workspace_id, levels=[1, 2]
            )
        except Exception as exc:
            print(f"skip {message[:20]}: {exc}"[:120])
            continue
        for h in hits:
            out.append({
                "message": message,
                "memory": str(h.get("content") or ""),
                "sim": round(float(h.get("similarity") or 0.0), 4),
                "source": h.get("source"),
            })
    open(sys.argv[2], "w").write(json.dumps(out, ensure_ascii=False))
    print(f"exported {len(out)} pairs from {len(messages)} messages")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
