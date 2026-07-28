"""开反思 flag 之前, 先看它会写出什么.

反思是记忆系统里唯一会写入**推断**的路径。推断错了不会报错, 只会让 AI 带着一个
错误的判断跟用户相处几个月。所以开之前必须先看一眼产出 —— 这个脚本只算不写。

它同时也是评估"数据量够不够"的工具: 如果多数用户连事实都凑不满 3 条, 说明还不到
开启的时候, 硬开只会逼模型编。

用法 (生产容器内):
    python preview_reflection.py              # 看所有活跃会话
    python preview_reflection.py --limit 3
"""

from __future__ import annotations

import argparse
import asyncio

from app.db import db
from app.services.memory.reflection.reflect import MIN_FACTS_TO_REFLECT, reflect_for_user
from app.services.memory.reflection.signals import (
    collect_behavioural_facts,
    format_facts_for_prompt,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--with-llm", action="store_true",
                    help="也跑归纳 (会调 LLM); 缺省只看事实")
    args = ap.parse_args()

    await db.connect()
    scopes = await db.query_raw(
        """
        SELECT c.user_id, c.agent_id, c.workspace_id,
               COUNT(*) FILTER (WHERE m.role = 'user')::int AS n
        FROM conversations c
        JOIN messages m ON m.conversation_id = c.id
        GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT $1
        """,
        args.limit,
    )

    enough = 0
    for scope in scopes:
        print(f"\n{'─' * 62}")
        print(f"user {str(scope['user_id'])[:8]}  ({scope['n']} 条用户消息)")
        facts = await collect_behavioural_facts(
            user_id=scope["user_id"], agent_id=scope["agent_id"],
            workspace_id=scope["workspace_id"],
        )
        if len(facts) < MIN_FACTS_TO_REFLECT:
            print(f"  事实 {len(facts)} 条, 不足 {MIN_FACTS_TO_REFLECT} 条 —— 会跳过")
            continue
        enough += 1
        print(f"  行为事实 ({len(facts)} 条):")
        for line in format_facts_for_prompt(facts).splitlines():
            print(f"    {line}")

        if not args.with_llm:
            continue
        stats = await reflect_for_user(
            user_id=scope["user_id"], agent_id=scope["agent_id"],
            workspace_id=scope["workspace_id"], dry_run=True,
        )
        print(f"  归纳出 {stats['insights']} 条:")
        for item in stats.get("preview") or []:
            print(f"    「{item['text']}」  依据 {item['based_on']}")

    print(f"\n{'─' * 62}")
    print(f"{len(scopes)} 个会话里, {enough} 个有足够事实可反思")
    if enough == 0:
        print("现在开启没有意义 —— 先让互动数据积累一段时间")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
