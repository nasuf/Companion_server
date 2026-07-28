"""导出线上真实的近重复记忆对, 用于标定去重/删除阈值.

采样文本对做百分位或配对映射, 在分布中段是可靠的, 但去重阈值 (0.85) 卡在随机
记忆对分布的 99.9% 分位 —— 那一带几乎没有样本, 任何映射都是拿噪声外推.

要标定它只能用它真正服务的那类文本对: **确实高度相似的记忆对**. 库里现成就有
—— 8181 条记忆的 bge 向量都在 memory_embeddings 里, 直接查出旧模型认为超过阈值
的那些对, 把文本导出来, 再用新模型算一遍, 就知道新尺度上对应什么值.

同时导出一批"次相似"的对 (0.70-0.85) 作为负样本: 只看正样本会把阈值定得过低,
分不清"真重复"和"同话题但不同事".

用法 (在生产容器内):
    python export_near_duplicate_pairs.py /tmp/dup_pairs.json
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.db import db

# 旧模型 (bge-m3) 尺度上的判定区间
DUPLICATE_MIN = 0.85     # 现行 DEDUP_THRESHOLD, 视为真重复
NEAR_MISS_RANGE = (0.70, 0.85)   # 相似但不该判重
LIMIT_PER_BUCKET = 400


async def _pairs_in_range(low: float, high: float, limit: int) -> list[dict]:
    """成对取记忆文本 + 旧模型相似度. 只取 id 升序的一侧避免重复计对."""
    return await db.query_raw(
        """
        SELECT a.content AS text_a, b.content AS text_b,
               1 - (ea.embedding OPERATOR(extensions.<=>) eb.embedding) AS sim
        FROM memory_embeddings ea
        JOIN memory_embeddings eb ON eb.memory_id > ea.memory_id
        JOIN memories_ai a ON a.id = ea.memory_id
        JOIN memories_ai b ON b.id = eb.memory_id
        WHERE a.is_archived = false AND b.is_archived = false
          AND 1 - (ea.embedding OPERATOR(extensions.<=>) eb.embedding) >= $1
          AND 1 - (ea.embedding OPERATOR(extensions.<=>) eb.embedding) < $2
        LIMIT $3
        """,
        low, high, limit,
    )


async def main() -> None:
    out_path = sys.argv[1]
    await db.connect()
    duplicates = await _pairs_in_range(DUPLICATE_MIN, 1.01, LIMIT_PER_BUCKET)
    near_misses = await _pairs_in_range(*NEAR_MISS_RANGE, LIMIT_PER_BUCKET)
    await db.disconnect()

    payload = {
        "duplicates": [
            {"a": r["text_a"], "b": r["text_b"], "old_sim": float(r["sim"])}
            for r in duplicates
        ],
        "near_misses": [
            {"a": r["text_a"], "b": r["text_b"], "old_sim": float(r["sim"])}
            for r in near_misses
        ],
    }
    texts = sorted({p[k] for bucket in payload.values() for p in bucket for k in ("a", "b")})
    open(out_path, "w").write(json.dumps(payload, ensure_ascii=False))
    flat = out_path.replace(".json", "_flat.json")
    open(flat, "w").write(json.dumps(texts, ensure_ascii=False))

    print(f"旧模型 >= {DUPLICATE_MIN} 的对: {len(payload['duplicates'])}")
    print(f"{NEAR_MISS_RANGE[0]}-{NEAR_MISS_RANGE[1]} 的对: {len(payload['near_misses'])}")
    print(f"不重复文本: {len(texts)} → {flat}")
    for p in payload["duplicates"][:5]:
        print(f"  {p['old_sim']:.3f}  「{p['a'][:34]}」 ↔ 「{p['b'][:34]}」")


if __name__ == "__main__":
    asyncio.run(main())
