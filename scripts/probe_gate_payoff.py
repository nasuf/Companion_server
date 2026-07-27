"""相关度闸门改动的真实收益探针.

闸门判"中/强"只是**去查**记忆, 查得到才算数. 这个脚本对两组真实消息各跑一次
生产检索路径, 回答两个问题:

  翻转组 (改后才查的): 有多少条真的召回了记忆? 召不回就只是白付一次向量检索.
  仍判弱组: 有多少条其实召得回? 那是改完之后仍然漏掉的.

用生产的 search_by_embedding, 不手拼 SQL —— 裸 `<=>` 不在 search_path 里,
必须走 OPERATOR(extensions.<=>), 照抄函数是唯一不会写歪的方式.
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.db import db
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.memory.storage.embedding import generate_embedding

THRESHOLD = 0.50  # retrieval/hybrid.py _SIMILARITY_THRESHOLD


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


async def _probe(message: str) -> tuple[float, str] | None:
    scope = await _scope(message)
    if scope is None:
        return None
    user_id, workspace_id = scope
    emb = await generate_embedding(message)
    hits = await search_by_embedding(
        emb, user_id, top_k=3, workspace_id=workspace_id, levels=[1, 2]
    )
    if not hits:
        return 0.0, ""
    return float(hits[0].get("similarity") or 0.0), str(hits[0].get("content") or "")


async def main() -> None:
    await db.connect()
    data = json.loads(open(sys.argv[1]).read())
    for label, messages in (
        ("翻转为查", data["flipped"]),
        ("仍判弱", data["still_weak"]),
    ):
        hit = miss = skip = 0
        sims: list[float] = []
        samples: list[tuple[float, str, str]] = []
        for message in messages:
            try:
                result = await _probe(message)
            except Exception as exc:  # keep going; report at the end
                print(f"    ERROR {message[:24]}: {exc}"[:150])
                skip += 1
                continue
            if result is None:
                skip += 1
                continue
            sim, text = result
            sims.append(sim)
            if sim >= THRESHOLD:
                hit += 1
                if len(samples) < 8:
                    samples.append((round(sim, 3), message, text[:46]))
            else:
                miss += 1
        total = hit + miss
        rate = f"{hit / total:.0%}" if total else "n/a"
        print(f"\n[{label}] n={total}  召回到记忆(sim>={THRESHOLD}) {hit} ({rate})  "
              f"召不回 {miss}  无法定位 {skip}")
        # 0.50 的阈值几乎放行一切, 所以"召回到"这个二值量没有区分度.
        # 看相似度分布, 以及在更严阈值下两组还剩多少.
        if sims:
            ordered = sorted(sims)
            mid = ordered[len(ordered) // 2]
            print(f"    top-sim  中位 {mid:.3f}  均值 {sum(sims)/len(sims):.3f}  "
                  f"最高 {max(sims):.3f}  最低 {min(sims):.3f}")
            for cut in (0.60, 0.65, 0.70, 0.75):
                n_over = sum(1 for s in sims if s >= cut)
                print(f"    >= {cut:.2f}  {n_over:>3} / {len(sims)}  ({n_over/len(sims):.0%})")
        for sim, message, text in samples:
            print(f"    {sim}  {message[:30]:<32} ←  {text}")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
