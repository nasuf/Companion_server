"""生产候选池抽样器: 从 prod DB 拉真实记忆当"分散注意力"候选加进 P3 权重扫描.

## 为什么

原来的 temporal_recall 用 13 条手工种子 + 每题 ~10 个候选. 问题:
真实用户单次检索池子有几十到几百条候选, 里面掺杂**其它话题的近期新鲜事件**
——这些事件跟 query 语义不搭, 但 occur_time 很新, 会被 P3 加满 boost.
只有把这种噪音塞进候选池, 才能看到 P3 权重调高时"新鲜噪音是否会顶掉正确答案".

小池子看不到这层压力 —— 我们的合成种子里就 gym / 面试 / 居住 三个话题, 各条彼此
差异明显, 相似度很干净. 真实池子里有几千条 "工作/生活" 类候选, 相似度分布密得多.

## 怎么用

    python -m evals.temporal_recall.run_eval --isolate-ranking --sweep-recency \
        --prod-pool-size 100

它会:
  1. 从 prod DB 各抽样出 N/2 条 AI + N/2 条 user 记忆, 覆盖多种 sub_category,
     一半有 occur_time 一半没 (模拟真实混合).
  2. 用 Ollama 算真实 embedding, 存磁盘缓存 (跟种子共用一份缓存).
  3. 对每道题, 合并"种子池 + prod 池"作为候选, 走同样的相似度过滤和排序.
  4. 只在结果里打 ID (memory_XXXXXXXX), 不打内容 —— 减少无意的日志泄漏.

## 隐私

- 内容会通过 embedding 走 Ollama (本地, 不外发)
- 内容会被写进 .emb_cache.json (本地; 已在 .gitignore, 走 evals/**/*emb_cache.json*)
- 结果输出里只有 ID 前缀, 不写内容
- 只抽样一个 agent (最近的) 的记忆, 不做跨用户批量拉取
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _to_dt(value: Any) -> datetime | None:
    """query_raw 把 timestamp 返成 ISO 字符串, 不是 datetime —— 得手动 parse.

    (勾住这个坑: 直接 isinstance(value, datetime) 永远 False, sampler 会静默把所有
    行丢光, 池子空空如也, 权重扫描的 prod pool 变成没有效果的假装.)
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return None


@dataclass(frozen=True)
class ProdSeed:
    """一条从 prod DB 拉来的真实记忆, 用作候选池分散注意力.

    字段跟 TemporalSeed 对齐 —— run_eval._candidate 拿它同样能 build dict.
    """

    id: str            # memory_XXXXXXXX (前 8 位), 不透露完整 ID
    text: str          # 内容 (用来算相似度, 只在候选/embed 环节存在)
    main: str
    sub: str
    occur_time: datetime | None
    statement_time: datetime  # DB 里的 statement_time; 无则用 created_at
    source: str
    importance: float
    level: int


async def _query_pool_side(
    source: str, agent_id: str | None, half: int,
) -> list[dict[str, Any]]:
    """从 memories_{source} 抽样一半带 occur_time / 一半不带, 覆盖多话题.

    做法: 各 sub_category 取前若干条, 加起来到 half. 避免只从一个 sub 抽 (会让
    候选池全部集中在一个话题, 失去多样性).
    """
    from app.db import db

    table = f"memories_{source}"
    with_occur_half = half // 2
    without_occur_half = half - with_occur_half
    agent_where = "AND user_id = $1" if agent_id else ""
    args = [agent_id] if agent_id else []

    async def _sample(with_occur: bool, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        occur_filter = "IS NOT NULL" if with_occur else "IS NULL"
        # RANDOM() 而不是 ORDER BY id: 每次跑随机拉一批, 避免固定命中同一批 memory
        # 让 eval 结果对种子分布敏感.
        sql = f"""
            SELECT id, content, main_category AS main, sub_category AS sub,
                   occur_time, statement_time, created_at,
                   importance, level
            FROM {table}
            WHERE is_archived = FALSE
              AND occur_time {occur_filter}
              AND main_category IS NOT NULL
              {agent_where}
            ORDER BY RANDOM()
            LIMIT {int(limit)}
        """
        return await db.query_raw(sql, *args)

    with_occur = await _sample(with_occur=True, limit=with_occur_half)
    without_occur = await _sample(with_occur=False, limit=without_occur_half)
    return with_occur + without_occur


async def load_prod_pool(
    total_size: int, agent_id: str | None = None,
) -> list[ProdSeed]:
    """拉 total_size 条 prod 记忆, AI/user 各一半, 每半再一半带 occur_time.

    agent_id=None → 系统级抽样 (跨用户); 生产 eval 里建议指定, 减少 PII 面。
    """
    from app.db import db

    was_connected = getattr(db, "_connected", False)
    if not was_connected:
        await db.connect()

    try:
        per_side = max(1, total_size // 2)
        ai_rows = await _query_pool_side("ai", agent_id, per_side)
        user_rows = await _query_pool_side("user", agent_id, per_side)
    finally:
        if not was_connected:
            await db.disconnect()

    seeds: list[ProdSeed] = []
    for src, rows in [("ai", ai_rows), ("user", user_rows)]:
        for r in rows:
            content = (r.get("content") or "").strip()
            stmt_time = _to_dt(r.get("statement_time")) or _to_dt(r.get("created_at"))
            if not content or stmt_time is None:
                # 无内容或无时间戳的行直接跳过 —— 时间戳是排序层新鲜度的必要输入.
                continue
            # ID 只留前 8 位, 结果输出时不再暴露完整 ID
            mid = f"prod_{src}_{str(r['id'])[:8]}"
            seeds.append(ProdSeed(
                id=mid,
                text=content,
                main=str(r.get("main") or "其他"),
                sub=str(r.get("sub") or "其他"),
                occur_time=_to_dt(r.get("occur_time")),
                statement_time=stmt_time,
                source=src,
                importance=float(r.get("importance") or 0.5),
                level=int(r.get("level") or 2),
            ))
    logger.info(
        f"[temporal_recall/prod_pool] loaded {len(seeds)} seeds "
        f"(ai={len([s for s in seeds if s.source == 'ai'])}, "
        f"user={len([s for s in seeds if s.source == 'user'])}, "
        f"with_occur={len([s for s in seeds if s.occur_time])})",
    )
    return seeds
