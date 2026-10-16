"""把存量 agent 的 L1 姓名记忆补上昵称 —— 「我叫陈砚」→「我叫陈砚，你可以叫我阿砚」.

## 为什么需要这个

新建 agent 在初始化时就会写出带昵称的姓名记忆 (life_story.name_memory_content)。
迁移 20261016090000_agent_nickname 已按姓名库给存量 agent 回填了 ai_agents.nickname,
但它们的姓名记忆是建号那天写的, 不带昵称 —— 用户问"我该怎么叫你"时 AI 不知道。

## 怎么改

逐条走 manual_edit.edit_memory_with_evidence, 与后台/用户手动改记忆是同一条事务路径:
CAS 防并发、先算新 embedding、写 admin_edit 证据来源、记 changelog、失效作用域缓存。
不直接 UPDATE content —— 那样 embedding 和证据版本都会跟内容对不上。

只改能确定形状的记忆: 内容必须是 `我叫{name}` 或 `我叫{name}。{补充}`。
已经带「你可以叫我」的视为已完成 (幂等, 可重复跑); 其他形状 (改过名、人工编辑过)
只报告不动, 留给人工判断。模板 agent 也在范围内 —— 之后克隆出去的副本会直接带上。

## 用法 (生产容器内)

    python scripts/backfill_agent_nickname_memories.py                    # 只读预检, 打印计划
    python scripts/backfill_agent_nickname_memories.py --apply \\
        --actor-admin-id <后台管理员 user id>                              # 执行

stdout 每行一个 JSON, 保存为执行回执。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import db  # noqa: E402
from app.services.life_story import name_memory_content  # noqa: E402
from app.services.memory.manual_edit import MemoryEditError, edit_memory_with_evidence  # noqa: E402
from app.services.memory.storage import repo as memory_repo  # noqa: E402

NICKNAME_MARK = "，你可以叫我"
REASON = "agent_nickname_backfill"


def plan_content(content: str, name: str, nickname: str) -> tuple[str, str | None]:
    """(status, new_content). status: update / done / skip_unrecognized."""
    prefix = f"我叫{name}"
    if not content.startswith(prefix):
        return "skip_unrecognized", None
    rest = content[len(prefix):]
    if rest.startswith(NICKNAME_MARK):
        return "done", None
    if rest == "":
        detail = None
    elif rest.startswith("。"):
        detail = rest[1:]
    else:
        return "skip_unrecognized", None
    target = name_memory_content(name, nickname=nickname, detail=detail)
    return ("done", None) if target == content else ("update", target)


async def candidates() -> list[dict]:
    # 作用域谓词与 manual_write.memory_scope 一致, 选出来的都是能被编辑的行。
    return await db.query_raw(
        """SELECT m.id, m.content, a.id AS agent_id, a.name, a.nickname
           FROM memories_ai m
           JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
           JOIN ai_agents a ON a.id=w.agent_id AND a.user_id=w.user_id
           JOIN users u ON u.id=w.user_id
           WHERE m.main_category='身份' AND m.sub_category='姓名' AND m.level=1
             AND NOT m.is_archived
             AND w.status='active' AND w.archived_at IS NULL
             AND a.status='active' AND a.archived_at IS NULL
             AND u.status='active' AND u.archived_at IS NULL
             AND a.nickname IS NOT NULL AND btrim(a.nickname)<>'' AND a.nickname<>a.name
           ORDER BY a.created_at, m.id"""
    )


async def run(apply: bool, actor_admin_id: str | None) -> None:
    await db.connect()
    try:
        if apply:
            actor = await db.user.find_unique(where={"id": actor_admin_id})
            if actor is None or actor.role != "admin" or actor.status != "active":
                raise SystemExit("--actor-admin-id 必须是在用的后台管理员账号")
        counts: dict[str, int] = {}
        for row in await candidates():
            status, target = plan_content(row["content"], row["name"].strip(), row["nickname"].strip())
            line = {"memory_id": row["id"], "agent_id": row["agent_id"], "status": status,
                    "before": row["content"], "after": target}
            if status == "update" and apply and target is not None:
                record = await memory_repo.find_unique(row["id"])
                if record is None or record.content != row["content"]:
                    line["status"] = "skip_changed_concurrently"
                else:
                    try:
                        await edit_memory_with_evidence(
                            record=record, content=target,
                            actor={"sub": actor_admin_id, "role": "admin"}, reason=REASON,
                        )
                        line["status"] = "updated"
                    except MemoryEditError as exc:
                        line["status"] = f"failed:{exc.detail}"
            counts[line["status"]] = counts.get(line["status"], 0) + 1
            print(json.dumps(line, ensure_ascii=False), flush=True)
        print(json.dumps({"phase": "applied" if apply else "preflight", "counts": counts},
                         ensure_ascii=False), flush=True)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--actor-admin-id")
    args = parser.parse_args()
    if args.apply and not args.actor_admin_id:
        parser.error("--apply 需要 --actor-admin-id")
    asyncio.run(run(args.apply, args.actor_admin_id))
