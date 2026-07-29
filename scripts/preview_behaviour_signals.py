"""看互动行为观察会给画像带来什么.

行为观察是画像的第二类输入: 记忆记录用户**说过**的话, 观察记录他**怎么跟 AI 相处**
—— 什么时候来、情绪如何、习惯长句还是短句。

这些观察曾经被做成独立记忆条目写进检索池, 实测行不通 (见 portrait._behaviour_section
的说明): 72 条真实消息只有 7% 能召回, 且多是"时间"这类表层词误配。改成喂给画像之后
不需要被检索到 —— 画像每轮必然注入。

用法 (生产容器内):
    python preview_reflection.py                # 只看观察
    python preview_reflection.py --with-portrait  # 也生成一份画像看效果 (不写库)
"""

from __future__ import annotations

import argparse
import asyncio

from app.db import db
from app.services.llm.models import get_utility_model, invoke_text
from app.services.memory.behaviour_signals import (
    collect_behavioural_facts,
    format_facts_for_prompt,
)
from app.services.memory.storage import repo as memory_repo
from app.services.prompting.store import get_prompt_text


async def _draft_portrait(user_id: str, workspace_id: str | None, behaviour: str) -> str:
    """按当前画像提示词生成一份, 不写库 —— 只为对比看效果。"""
    memories = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id, "workspaceId": workspace_id,
            "level": {"in": [1, 2]}, "isArchived": False,
        },
        order={"importance": "desc"},
        take=30,
    )
    if not memories:
        return "(无记忆, 画像不会生成)"
    memories_text = "\n".join(
        f"- [L{m.level}] [{m.mainCategory or '未分类'}/{m.subCategory or '其他'}] {m.content}"
        for m in memories
    )
    prompt = (await get_prompt_text("portrait.generation")).format(
        memories=memories_text, behaviour=behaviour,
    )
    return await invoke_text(get_utility_model(), prompt)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--with-portrait", action="store_true",
                    help="也生成一份画像看效果 (会调 LLM, 不写库)")
    args = ap.parse_args()

    await db.connect()
    scopes = await db.query_raw(
        """
        SELECT c.user_id, c.agent_id, c.workspace_id,
               COUNT(*) FILTER (WHERE m.role = 'user')::int AS n
        FROM conversations c
        JOIN messages m ON m.conversation_id = c.id
        WHERE c.is_deleted = false
        GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT $1
        """,
        args.limit,
    )

    with_facts = 0
    for scope in scopes:
        print(f"\n{'─' * 62}")
        print(f"user {str(scope['user_id'])[:8]}  ({scope['n']} 条用户消息)")
        facts = await collect_behavioural_facts(
            user_id=scope["user_id"], agent_id=scope["agent_id"],
            workspace_id=scope["workspace_id"],
        )
        if not facts:
            print("  互动数据还不够, 画像里这一段会是占位符")
            continue
        with_facts += 1
        for line in format_facts_for_prompt(facts).splitlines():
            print(f"    {line}")

        if args.with_portrait:
            behaviour = "\n".join(f"- {f.statement}" for f in facts)
            draft = await _draft_portrait(
                scope["user_id"], scope["workspace_id"], behaviour,
            )
            print(f"\n  画像草稿 ({len(draft)} 字):")
            print(f"    {draft}")

    print(f"\n{'─' * 62}")
    print(f"{len(scopes)} 个会话里, {with_facts} 个有可用的互动观察")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
