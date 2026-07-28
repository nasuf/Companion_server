"""给已达标但还没有画像的用户补生成一次.

画像的前置条件此前是 `L2 ≥ 20 AND L1 ≥ 5`, 生产上一次都没被满足过 —— 那两个数在
真实数据里此消彼长 (层级由 importance 推导), AND 起来就不可达。门槛放宽之后, 已经
够料的用户不必再等下一个周日的 cron。

这个脚本只补首次生成, 不动已有画像 —— 后续更新仍由 weekly_portrait 负责。

用法 (生产容器内):
    python backfill_user_portraits.py          # 预览谁会被生成
    python backfill_user_portraits.py --apply
"""

from __future__ import annotations

import argparse
import asyncio

from app.db import db
from app.services.portrait import check_portrait_preconditions, generate_portrait


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="缺省只预览")
    args = ap.parse_args()

    await db.connect()
    pairs = await db.query_raw(
        """
        SELECT DISTINCT c.user_id, c.agent_id
        FROM conversations c
        JOIN ai_agents a ON a.id = c.agent_id
        WHERE a.status = 'active'
          AND NOT EXISTS (
            SELECT 1 FROM user_portraits p
            WHERE p.user_id = c.user_id AND p.agent_id = c.agent_id
          )
        """
    )
    print(f"{len(pairs)} 个 (user, agent) 对还没有画像")

    eligible: list[tuple[str, str]] = []
    for row in pairs:
        try:
            if await check_portrait_preconditions(row["user_id"], row["agent_id"]):
                eligible.append((row["user_id"], row["agent_id"]))
        except Exception as e:
            print(f"  检查失败 {str(row['user_id'])[:8]}: {e}")

    print(f"其中 {len(eligible)} 个达到门槛")
    if not args.apply:
        for user_id, agent_id in eligible:
            print(f"  将生成: user={user_id[:8]} agent={agent_id[:8]}")
        print("\n加 --apply 执行")
        await db.disconnect()
        return

    ok = 0
    for user_id, agent_id in eligible:
        try:
            # 每个用户一次 LLM 调用; 失败不影响其余人。
            portrait = await generate_portrait(user_id, agent_id)
        except Exception as e:
            print(f"  生成失败 user={user_id[:8]}: {e}")
            continue
        if portrait:
            ok += 1
            print(f"  user={user_id[:8]} ({len(portrait)} 字) {portrait[:60]}…")
        else:
            print(f"  user={user_id[:8]} 生成返回空")

    print(f"\n完成: {ok}/{len(eligible)}")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
