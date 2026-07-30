#!/usr/bin/env python
"""把存量的超长记忆按句子/分号拆成能被检索到的多条.

## 为什么需要它

检索注入有单条 token 上限 (`MAX_MEMORY_TOKENS_PER_ITEM`)，超过的条目会被
`context_selector` **整条跳过**。它们躺在库里，占着统计口径，但任何对话都不会用到
——等于不存在。2026-07-30 盘点：`memories_ai` 7954 条里有 606 条 (7.6%) 处于这个
状态，全部落在 L2（只走检索，没有常驻注入兜底）。

这 606 条其实只有 63 段不同文本，是两份 agent template 被克隆 11-22 次的结果。
生成侧已经收紧字数不再产出新的，这个脚本处理存量。

## 做法

对每条超限记忆跑 `split_multi_fact`（先按「；」拆多事实，仍超限的按句子边界拆并补
回「标题：」前缀），然后：

    原行  → content 改成第 1 片（保留 id，changelog / access log / 实体引用不断）
    新行  → 第 2 片起各插一行，元数据全部继承
    全部重新 embedding（内容变了，旧向量指向的是不存在的文本）

拆不出两片的（整段就是一句、或引号不配对）**跳过不动**——硬切会切在句子中间，
读者拿到半句话比拿不到更糟。

## 安全性

- 默认 dry-run，`--apply` 才写库
- 写库前把每条原文存进 journal JSON，`--rollback <journal>` 可完整还原
- 幂等：修完的条目不再超限，重跑扫不到
- `--workspace` 可先在单个 workspace 上试
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import db  # noqa: E402
from app.services.memory.recording.splitting import split_multi_fact  # noqa: E402
from app.services.memory.retrieval.context_selector import (  # noqa: E402
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)
from app.services.memory.storage.embedding import (  # noqa: E402
    generate_embedding,
    store_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("split-oversized")

TABLES = ("memories_ai", "memories_user")

# 元数据字段：新行要从原行继承的列。mention_count / current_score 刻意不继承——
# 那是"这条被用过多少次"的历史，复制到 N 个新行会把统计放大 N 倍。原行保留自己的。
_INHERITED = (
    "user_id", "workspace_id", "type", "level", "importance",
    "occur_time", "main_category", "sub_category", "statement_time",
    "recurrence", "provenance",
)

# 时间列是 `timestamp without time zone`，而 query_raw 把它们读成带 +00:00 的字符串。
# 直接回写会报类型错；不带 timezone 强转则会按会话时区偏移（作息表那个日期错位 bug
# 就是这么来的）。显式走 timestamptz 再落回 UTC naive，与 Prisma 的存储约定一致。
_TS_COLUMNS = frozenset({"occur_time", "statement_time"})
_NOW_UTC = "(NOW() AT TIME ZONE 'UTC')"


def _placeholder(column: str, index: int) -> str:
    if column in _TS_COLUMNS:
        return f"${index}::timestamptz AT TIME ZONE 'UTC'"
    return f"${index}"


@dataclass
class Plan:
    table: str
    memory_id: str
    original: str
    pieces: list[str]
    row: dict

    @property
    def new_pieces(self) -> list[str]:
        return self.pieces[1:]


@dataclass
class Stats:
    scanned: int = 0
    oversized: int = 0
    planned: int = 0
    skipped_unsplittable: int = 0
    skipped_changed: int = 0
    rows_created: int = 0
    embed_failed: list[str] = field(default_factory=list)


async def load_oversized(table: str, workspace: str | None) -> list[dict]:
    cols = "id, content, mention_count, current_score, " + ", ".join(_INHERITED)
    sql = f"SELECT {cols} FROM {table} WHERE is_archived = false"
    if workspace:
        rows = await db.query_raw(sql + " AND workspace_id = $1", workspace)
    else:
        rows = await db.query_raw(sql)
    return rows


def build_plans(table: str, rows: list[dict], stats: Stats) -> list[Plan]:
    plans: list[Plan] = []
    for row in rows:
        stats.scanned += 1
        content = row.get("content") or ""
        if estimate_tokens(content) <= MAX_MEMORY_TOKENS_PER_ITEM:
            continue
        stats.oversized += 1
        pieces = split_multi_fact(content)
        if len(pieces) < 2:
            # 拆不动。硬切会切在句子中间, 那比超限更糟 —— 留给人工或后续 LLM 改写。
            stats.skipped_unsplittable += 1
            continue
        stats.planned += 1
        plans.append(Plan(table=table, memory_id=row["id"], original=content,
                          pieces=pieces, row=row))
    return plans


async def apply_plan(plan: Plan, new_ids: list[str], stats: Stats) -> bool:
    """执行一条拆分, 返回是否真的改了. new_ids 由调用方预先生成并已落 journal.

    id 预生成而不是在这里现取, 是为了让 journal 在写库**之前**就记全所有将要产生的
    行。否则崩在 INSERT 中途时, 已插入的行没人记着, 回滚会留下孤儿。

    UPDATE 带 `content = 原文` 条件: 这个脚本要跑好几分钟, 期间线上是活的, hygiene
    合并或新一轮抽取都可能改到同一条。无条件覆盖会把它们的结果冲掉, 而且不留痕迹。
    条件不满足就整条跳过 —— 别人改过的内容我们没重新拆过, 硬插分片会和新内容重复。
    """
    changed = await db.execute_raw(
        f"UPDATE {plan.table} SET content = $1, updated_at = {_NOW_UTC} "
        f"WHERE id = $2 AND content = $3",
        plan.pieces[0], plan.memory_id, plan.original,
    )
    if not changed:
        logger.info("  ~ 跳过 %s: 内容已被其他流程修改", plan.memory_id[:8])
        stats.skipped_changed += 1
        return False
    await _log_change(plan.row, plan.memory_id, "update", plan.original, plan.pieces[0])

    bound = ["id", "content", "mention_count", *_INHERITED]
    cols = [*bound, "created_at", "updated_at"]
    marks = [_placeholder(c, i + 1) for i, c in enumerate(bound)] + [_NOW_UTC, _NOW_UTC]
    sql = f'INSERT INTO {plan.table} ({", ".join(cols)}) VALUES ({", ".join(marks)})'

    for new_id, piece in zip(new_ids, plan.new_pieces):
        values = [new_id, piece, 0, *(plan.row[c] for c in _INHERITED)]
        await db.execute_raw(sql, *values)
        await _log_change(plan.row, new_id, "insert", None, piece)
        stats.rows_created += 1

    # 内容变了, 旧向量指向的是已经不存在的文本 —— 原行也必须重嵌, 不只是新行。
    for mem_id, text in [(plan.memory_id, plan.pieces[0]), *zip(new_ids, plan.new_pieces)]:
        try:
            await store_embedding(mem_id, await generate_embedding(text))
        except Exception as e:
            # 嵌入失败不回滚已写的行: 行本身是对的, 缺向量只是暂时检索不到,
            # 重跑 embedding 补齐即可; 回滚反而让数据处于半修状态。
            logger.warning("  ! embedding 失败 %s: %s", mem_id[:8], e)
            stats.embed_failed.append(mem_id)
    return True


async def _log_change(row: dict, memory_id: str, op: str,
                      old: str | None, new: str | None) -> None:
    try:
        await db.execute_raw(
            """INSERT INTO memory_changelogs
               (id, user_id, workspace_id, memory_id, operation, old_value, new_value)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            str(uuid.uuid4()), row["user_id"], row.get("workspace_id"),
            memory_id, op, old, new,
        )
    except Exception as e:
        # changelog 是审计副产物, 写失败不该让修复本身失败。
        logger.warning("  ! changelog 写入失败 %s: %s", memory_id[:8], e)


async def reembed(journal_path: Path) -> None:
    """按 journal 重新生成向量.

    拆分本身和嵌入是两件独立的事: 行写对了但嵌入服务临时不可用是常态 (实测本地
    Ollama 只监听 IPv4 而 httpx 把 localhost 解析成 ::1, 65 条全 503)。没有这个模式
    就只能整体回滚重来 —— 而行数据其实是好的, 只缺向量。

    原行也要重嵌: 它的内容已经换成第 1 片, 旧向量代表的是拆分前的整段文本, 留着会让
    检索按"整个故事"匹配却只拿到开头一段。
    """
    journal = json.loads(journal_path.read_text())
    targets: list[tuple[str, str]] = []
    for entry in journal["entries"]:
        ids = [entry["memory_id"], *entry["new_ids"]]
        rows = await db.query_raw(
            f'SELECT id, content FROM {entry["table"]} WHERE id = ANY($1::text[])', ids
        )
        targets.extend((r["id"], r["content"]) for r in rows)

    ok = failed = 0
    for i, (mem_id, text) in enumerate(targets, 1):
        try:
            await store_embedding(mem_id, await generate_embedding(text))
            ok += 1
        except Exception as e:
            logger.warning("  ! %s: %s", mem_id[:8], e)
            failed += 1
        if i % 50 == 0:
            logger.info("  … %d/%d", i, len(targets))
    logger.info("重嵌完成: 成功 %d, 失败 %d", ok, failed)


async def rollback(journal_path: Path) -> None:
    """按 journal 完整还原.

    journal 里的条目可能"记了但没执行"(崩在写 journal 之后、写库之前)。UPDATE 回原文
    对没改过的行是幂等的, DELETE 不存在的 id 也不报错 —— 所以这两种状态都能安全收敛。
    """
    journal = json.loads(journal_path.read_text())
    restored = deleted = 0
    for entry in journal["entries"]:
        await db.execute_raw(
            f'UPDATE {entry["table"]} SET content = $1 WHERE id = $2',
            entry["original"], entry["memory_id"],
        )
        try:
            await store_embedding(entry["memory_id"],
                                  await generate_embedding(entry["original"]))
        except Exception as e:
            logger.warning("  ! 回滚重嵌失败 %s: %s", entry["memory_id"][:8], e)
        restored += 1
        for new_id in entry["new_ids"]:
            await db.execute_raw("DELETE FROM memory_embeddings WHERE memory_id = $1", new_id)
            await db.execute_raw(f'DELETE FROM {entry["table"]} WHERE id = $1', new_id)
            deleted += 1
    logger.info("回滚完成: 还原 %d 条原文, 删除 %d 条新增行", restored, deleted)


def preview(plans: list[Plan], n: int) -> None:
    logger.info("\n--- 拆分预览 (前 %d 条) ---", min(n, len(plans)))
    for p in plans[:n]:
        logger.info("\n[%s %s %dtok] %s…",
                    p.table, p.memory_id[:8], estimate_tokens(p.original),
                    p.original[:60])
        for i, piece in enumerate(p.pieces, 1):
            logger.info("   (%d) [%dtok] %s", i, estimate_tokens(piece), piece)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="真正写库 (默认只 dry-run)")
    ap.add_argument("--workspace", help="只处理这一个 workspace")
    ap.add_argument("--limit", type=int, help="最多处理多少条 (灰度用)")
    ap.add_argument("--preview", type=int, default=5, help="预览多少条拆分结果")
    ap.add_argument("--journal", default="/tmp/split_oversized_journal.json")
    ap.add_argument("--rollback", help="按 journal 回滚")
    ap.add_argument("--reembed", help="按 journal 补跑向量 (嵌入服务中途挂掉时用)")
    args = ap.parse_args()

    await db.connect()
    try:
        if args.rollback:
            await rollback(Path(args.rollback))
            return 0
        if args.reembed:
            await reembed(Path(args.reembed))
            return 0

        stats = Stats()
        plans: list[Plan] = []
        for table in TABLES:
            rows = await load_oversized(table, args.workspace)
            plans.extend(build_plans(table, rows, stats))

        plans.sort(key=lambda p: -estimate_tokens(p.original))
        if args.limit:
            plans = plans[: args.limit]

        logger.info("扫描 %d 条, 超限 %d 条", stats.scanned, stats.oversized)
        logger.info("  可拆 %d 条 → 将产生 %d 条新行",
                    len(plans), sum(len(p.new_pieces) for p in plans))
        logger.info("  拆不动跳过 %d 条", stats.skipped_unsplittable)
        distinct = len({p.original for p in plans})
        logger.info("  (去重后 %d 段不同文本, 其余为模板克隆)", distinct)

        if args.preview:
            preview(plans, args.preview)

        if not args.apply:
            logger.info("\n[dry-run] 未写库。加 --apply 执行。")
            return 0

        journal_path = Path(args.journal)
        journal = {"created_at": datetime.now(timezone.utc).isoformat(), "entries": []}
        for i, plan in enumerate(plans, 1):
            # 先落 journal 再动库。反过来的话, 崩在 UPDATE 之后、写 journal 之前的那
            # 一行原文就永久丢了 —— 库里是拆过的, 而没有任何地方记着它原来是什么。
            new_ids = [str(uuid.uuid4()) for _ in plan.new_pieces]
            entry = {
                "table": plan.table, "memory_id": plan.memory_id,
                "original": plan.original, "new_ids": new_ids,
            }
            journal["entries"].append(entry)
            journal_path.write_text(json.dumps(journal, ensure_ascii=False))

            if not await apply_plan(plan, new_ids, stats):
                # 内容被别人改过, 什么都没写。留在 journal 里会让回滚把原文盖回去,
                # 把别人的修改冲掉 —— 正是这个条件想避免的事。
                journal["entries"].remove(entry)
                journal_path.write_text(json.dumps(journal, ensure_ascii=False))
            if i % 50 == 0:
                logger.info("  … 已处理 %d/%d", i, len(plans))

        applied = len(journal["entries"])
        logger.info("\n完成: 改写 %d 条, 新增 %d 条", applied, stats.rows_created)
        if stats.skipped_changed:
            logger.info("并发跳过 %d 条 (内容已被其他流程修改, 重跑即可)",
                        stats.skipped_changed)
        logger.info("journal: %s", args.journal)
        if stats.embed_failed:
            logger.warning("embedding 失败 %d 条 (行已写好, 补跑嵌入即可): %s",
                           len(stats.embed_failed), stats.embed_failed[:5])
        return 0
    finally:
        await db.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
