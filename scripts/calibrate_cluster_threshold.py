"""标定 L3 簇压缩的聚类阈值.

`_CLUSTER_SIMILARITY = 0.75` 是 bge-m3 时代定的。换到 qwen3-embedding 后同题记忆
的相似度整体下移, 生产实测最大桶内的最高两两相似度只有 0.567 —— 阈值从没跟着改,
因为当时整合是关闭的, 不在那轮阈值校准的范围内。结果是: 即使打开 flag 也一簇都
聚不出来, 完全空转。

这里不套用百分位映射, 而是直接扫阈值看**簇里装的是不是真同一件事** —— 聚类质量
是可以直接看的, 比映射一个分位数更贴合用途。太松会把不相干的日常混进一条摘要,
而原行归档后就找不回来了。

用法 (生产容器内):
    python calibrate_cluster_threshold.py
"""

from __future__ import annotations

import asyncio
from collections import Counter

from app.db import db
from app.services.memory.lifecycle.consolidation import (
    _MIN_CLUSTER_SIZE,
    _load_candidates,
)
from app.services.memory.normalization import cosine_similarity

THRESHOLDS = (0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40)


def cluster_at(candidates: list[dict], threshold: float) -> list[list[dict]]:
    buckets: dict[tuple, list[dict]] = {}
    for c in candidates:
        buckets.setdefault(
            (c.get("main_category") or "", c.get("sub_category") or ""), [],
        ).append(c)

    clusters: list[list[dict]] = []
    for rows in buckets.values():
        assigned: set[str] = set()
        for i, seed in enumerate(rows):
            if seed["id"] in assigned:
                continue
            cluster = [seed] + [
                other for other in rows[i + 1:]
                if other["id"] not in assigned
                and cosine_similarity(seed["_vec"], other["_vec"]) >= threshold
            ]
            if len(cluster) >= _MIN_CLUSTER_SIZE:
                assigned.update(c["id"] for c in cluster)
                clusters.append(cluster)
    return clusters


async def main() -> None:
    await db.connect()
    scopes = await db.query_raw(
        """
        SELECT user_id, workspace_id, COUNT(*)::int AS n FROM memories_ai
        WHERE is_archived = false AND level = 3
        GROUP BY user_id, workspace_id ORDER BY n DESC LIMIT 5
        """
    )
    pools: list[list[dict]] = []
    for s in scopes:
        pools.append(await _load_candidates(
            source="ai", user_id=s["user_id"], workspace_id=s["workspace_id"],
        ))
    total = sum(len(p) for p in pools)
    print(f"{len(pools)} 个 workspace, 合计 {total} 条候选\n")

    print(f"{'阈值':>6}{'簇数':>7}{'覆盖行数':>10}{'最大簇':>8}")
    per_threshold: dict[float, list[list[dict]]] = {}
    for threshold in THRESHOLDS:
        clusters = [c for pool in pools for c in cluster_at(pool, threshold)]
        per_threshold[threshold] = clusters
        covered = sum(len(c) for c in clusters)
        largest = max((len(c) for c in clusters), default=0)
        print(f"{threshold:>6.2f}{len(clusters):>7}{covered:>10}{largest:>8}")

    # 挑第一个能聚出簇的阈值, 把内容打出来人工看是否同题 —— 这一步不能省:
    # 数量对了不代表聚对了, 而原行归档后就找不回来。
    for threshold in THRESHOLDS:
        clusters = per_threshold[threshold]
        if not clusters:
            continue
        print(f"\n阈值 {threshold} 下前 2 簇的内容 (看是否真同题):")
        for cluster in clusters[:2]:
            cats = Counter(
                (c.get("main_category"), c.get("sub_category")) for c in cluster
            )
            print(f"  簇 ({len(cluster)} 条) {cats.most_common(1)[0][0]}:")
            for c in cluster[:6]:
                print(f"    - {(c.get('content') or '')[:56]}")
            if len(cluster) > 6:
                print(f"    … 另外 {len(cluster) - 6} 条")
        break

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
