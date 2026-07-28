"""巡检模板 agent 的记忆分层是否符合现行规则.

模板是每个新用户的源头 —— H5 登录时它的记忆被整份复制过去, level 和 importance
逐字照搬。所以模板一旦分层不对, 之后每个新注册用户都会拿到一份错的, 而且不会有
任何报错。

单元测试管不到这件事: 它检查的是代码, 而模板是**数据**。模板可能被 admin 重新
生成、从文档导入、或手工编辑过。这个脚本检查数据本身。

判据 (与 life_story._tiered_importance 一致):
    核心身份 (L1_SINGLETON_SUBS)  应当在 L1
    其余人设                       应当在 L2
    knowledge_seed                 应当在 L1 (admin 维护的知识, 刻意例外)

用法 (生产容器内):
    python check_template_tiering.py
退出码非 0 表示模板需要修 —— 可挂到部署后的冒烟检查里。
"""

from __future__ import annotations

import asyncio
import sys

from app.db import db
from app.services.agent_template.registry import get_default_template_agent_id
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.taxonomy import L1_SINGLETON_SUBS


async def main() -> None:
    await db.connect()
    template_id = await get_default_template_agent_id()
    if not template_id:
        print("没有配置默认模板 —— 新用户走直接创建路径, 无需巡检")
        await db.disconnect()
        return

    rows = await db.query_raw(
        """
        SELECT m.id, m.level, m.main_category, m.sub_category, m.content,
               m.importance, COALESCE(m.provenance, '') AS provenance
        FROM memories_ai m
        JOIN chat_workspaces w ON w.id = m.workspace_id
        WHERE w.agent_id = $1 AND m.is_archived = false
        """,
        template_id,
    )
    print(f"模板 {template_id[:8]} 共 {len(rows)} 条未归档记忆")

    problems: list[str] = []
    for r in rows:
        is_core = (r["main_category"], r["sub_category"]) in L1_SINGLETON_SUBS
        is_knowledge = r["provenance"] == KNOWLEDGE_SEED
        level = r["level"]

        if is_knowledge:
            if level != 1:
                problems.append(
                    f"knowledge_seed 掉出 L1 (现 L{level}): {r['content'][:40]}"
                )
        elif is_core:
            if level != 1:
                problems.append(
                    f"核心身份不在 L1 (现 L{level}) "
                    f"[{r['main_category']}/{r['sub_category']}]: {r['content'][:36]}"
                )
        elif level == 1:
            problems.append(
                f"非核心人设仍在 L1 "
                f"[{r['main_category']}/{r['sub_category']}] "
                f"imp={float(r['importance']):.2f}: {r['content'][:36]}"
            )

    if problems:
        print(f"\n发现 {len(problems)} 处不符合现行分层:")
        for p in problems[:15]:
            print(f"  {p}")
        if len(problems) > 15:
            print(f"  … 另外 {len(problems) - 15} 处")
        print("\n修复: python retier_existing_persona.py --apply")
        print("影响: 修之前每个新注册用户都会克隆到这份错误分层")
    else:
        print("分层符合现行规则 —— 新用户克隆到的是正确的分层")

    await db.disconnect()
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    asyncio.run(main())
