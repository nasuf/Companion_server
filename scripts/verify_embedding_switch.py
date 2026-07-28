"""换 embedding 之后的线上实测: 走完整生产检索路径, 确认换对了.

"部署成功" 和 "检索还能用" 是两件事. 列换了、模型配了、迁移过了, 都不能说明
用户问一句话真能捞回相关记忆 —— 向量空间错配不会报错, 只会静默返回噪声.

这里拿真实用户消息走 hybrid_retrieve (生产热路径本身, 不是重写的近似), 看:

    召回率      多少条消息能捞到东西 (阈值定太高会归零)
    相似度分布  命中的相似度落在哪 (贴着阈值说明标定偏紧)
    实际内容    人眼能判断相不相关 —— 数字对了但内容驴唇不对马嘴, 说明空间还是错的

最后一项最要紧: 前两项在向量空间完全错配时也可能"看起来正常".

用法 (在生产容器内):
    python verify_embedding_switch.py
"""

from __future__ import annotations

import asyncio

from app.db import db
from app.services.memory.retrieval.hybrid import _SIMILARITY_THRESHOLD, hybrid_retrieve

SAMPLE = 25


async def main() -> None:
    await db.connect()
    rows = await db.query_raw(
        """
        SELECT DISTINCT ON (m.content) m.content, c.user_id, c.workspace_id
        FROM messages m JOIN conversations c ON c.id = m.conversation_id
        WHERE m.role = $1 AND char_length(m.content) BETWEEN 5 AND 60
          AND c.user_id IS NOT NULL
        ORDER BY m.content, m.created_at DESC
        LIMIT $2
        """,
        "user", SAMPLE,
    )
    print(f"阈值 {_SIMILARITY_THRESHOLD}  样本 {len(rows)} 条真实用户消息\n")

    hits = 0
    sims: list[float] = []
    for row in rows:
        try:
            result = await hybrid_retrieve(
                row["content"], row["user_id"], workspace_id=row["workspace_id"]
            )
        except Exception as exc:
            print(f"  ERROR 「{row['content'][:26]}」: {str(exc)[:70]}")
            continue
        memories = result.get("memories") or []
        if not memories:
            print(f"  ∅  「{row['content'][:26]}」")
            continue
        hits += 1
        top = memories[0]
        sim = float(getattr(top, "similarity", 0.0) or 0.0)
        sims.append(sim)
        print(f"  {sim:.3f} 「{row['content'][:26]}」 → 「{str(top.text)[:40]}」")

    print(f"\n召回到记忆的消息 {hits}/{len(rows)}")
    if sims:
        ordered = sorted(sims)
        print(f"命中相似度  中位 {ordered[len(ordered)//2]:.3f}  "
              f"最低 {ordered[0]:.3f}  最高 {ordered[-1]:.3f}")
        margin = ordered[0] - _SIMILARITY_THRESHOLD
        print(f"最低命中距阈值 {margin:+.3f} —— 贴得太近说明阈值偏紧, 差得太远说明偏松")
    print("\n最后一步靠人看: 上面每行的「消息 → 记忆」是否真的相关. "
          "向量空间若仍错配, 数字会正常但内容会驴唇不对马嘴.")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
