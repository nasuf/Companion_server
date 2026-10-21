"""存量 L1 清理: 把 L1 晋升闸门上线前错进 L1 的记忆按现行口径降级或归档.

2026-10 抽取把六成新记忆打在 importance 0.80, 生命周期一次注入就加到 0.92 升 L1。
闸门 (lifecycle/value.py:l1_promotion_allowed) 只拦以后的晋升; 库里已经混进 L1 的
「用户已经在工位了」「用户询问是否和对方在一个地方」仍在 prompt 里被当成核心事实。
本脚本按同一套口径回头处理一遍 —— 判定全部复用生产函数, 不另写一套。

## 处理什么

    用户侧 L1
      user_non_core_category            生活 / 情绪 / 各类"其他"      → 降 L2
      user_sustained_without_evidence   允许 L1 的偏好/思维子类, 靠晋升 (importance<0.85)
                                        进的 L1, 却没有跨 3 个自然日的注入记录 → 降 L2
      身份类                            保留
    AI 侧 (只碰聊天衍生: ai_authored / daily_summary / consolidated, 人设种子不动)
      ai_offline_copy_flip              任意层级。内容跟线下活动推荐/陪伴文案高度重合 ——
                                        那是对用户说的话, 被抽成了 AI 第一人称亲历 → 归档
      ai_chat_persona_claim             L1 的 身份/偏好/思维。现行写入规则本就拒绝这类
                                        聊天自述, 而 AI 的 L2/L3 不允许这三类 → 归档
      ai_chat_derived_l1                其余 L1 (生活/交互 关系里程碑除外)  → 降 L2

只动层级和归档标记, 不改内容、importance 和值。降到 L2 后值照常累积, 闸门保证它不会
再升回来。

## 流程: 只读演练 → 审核 → 按计划执行

    python -m scripts.repair_l1_core_memories --plan-out plan.json            # 只读演练
    python -m scripts.repair_l1_core_memories --apply --plan plan.json        # 执行
    python -m scripts.repair_l1_core_memories --apply --plan plan.json --reasons user_non_core_category
    python -m scripts.repair_l1_core_memories --revert [--plan plan.json]     # 预览还原
    python -m scripts.repair_l1_core_memories --revert --apply [--plan plan.json]

- 演练全程在一个 READ ONLY 事务里跑, 数据库层面拒绝任何写入; 计划文件只存内容
  哈希和 30 字预览。
- 执行只处理计划里列出的行, 不会顺手扩大范围: 每个 (用户, workspace) 一个事务,
  先拿生命周期同一把 advisory lock, 行锁后按当下数据重新判定 —— 内容变了、已经
  不在 L1、规则不再成立的一律跳过并计数。
- 每条改动写 changelog `repair_l1_core`, old/new 记 side/level/archived, --revert 按它
  还原; 还原只处理当前状态仍等于脚本写入状态的行。
- 执行与还原后按 scope 失效检索缓存 (要连生产 Redis, 所以执行必须在服务容器里跑)。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.db import db
from app.services.memory.config import L1_MIN_IMPORTANCE
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.lifecycle.value import (
    AI_CHAT_DERIVED_PROVENANCE,
    L1_PROMOTION_MIN_CONTRIBUTION_DAYS,
)
from app.services.memory.taxonomy import l1_promotion_requirement
from app.services.schedule_domain.sql_time import local_date_sql

PLAN_SCHEMA = "l1_core_repair/v1"
OPERATION = "repair_l1_core"
REVERT_OPERATION = "repair_l1_core_revert"
TABLES = {"user": "memories_user", "ai": "memories_ai"}

USER_NON_CORE = "user_non_core_category"
USER_UNSUSTAINED = "user_sustained_without_evidence"
AI_OFFLINE_COPY = "ai_offline_copy_flip"
AI_PERSONA_CLAIM = "ai_chat_persona_claim"
AI_CHAT_L1 = "ai_chat_derived_l1"
REASONS = (USER_NON_CORE, USER_UNSUSTAINED, AI_OFFLINE_COPY, AI_PERSONA_CLAIM, AI_CHAT_L1)

# 记忆文本的二元组有多少落在线下文案里才算"抄自文案"。生产样本: 真正翻转主语的
# 在 0.8 以上 (「我觉得大悦城里有不少独立书店…」对「大悦城里有不少独立书店…」),
# 同批抽取里别的话题 (「我和用户下五子棋…」) 在 0.2 以下。
OFFLINE_COPY_OVERLAP = 0.6
_AI_PERSONA_MAINS = frozenset({"身份", "偏好", "思维"})
_RELATIONSHIP_MILESTONE = ("生活", "交互")
_PUNCT_RE = re.compile(r"[\s，。！？!?~～…,.、:：；;\"'“”‘’（）()【】\[\]{}<>《》]+")
_AI_SUBJECT_RE = re.compile(r"^(我觉得|我认为|我)")
_PREVIEW_CHARS = 30
_ROW_COLUMNS = (
    "id, user_id, workspace_id, content, main_category, sub_category, importance, "
    "current_score, level, provenance, is_archived"
)


def content_hash(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def copy_overlap(memory_text: str | None, message_text: str | None) -> float:
    """AI 记忆有多大比例的内容直接来自这条消息。

    抽取把对用户说的"你"改写成"用户"、再在句首加"我/我觉得", 比较前先还原。
    """
    memory = _AI_SUBJECT_RE.sub("", _PUNCT_RE.sub("", memory_text or "")).replace("用户", "你")
    grams = _bigrams(memory)
    if not grams:
        return 0.0
    return len(grams & _bigrams(_PUNCT_RE.sub("", message_text or ""))) / len(grams)


def classify(
    side: str, row: dict, *, contribution_days: int = 0, offline_overlap: float = 0.0,
) -> tuple[str, str] | None:
    """(action, reason) 或 None (不处理)。演练与执行共用, 执行时用当下数据重算。"""
    if row["is_archived"]:
        return None
    main, sub = row["main_category"], row["sub_category"]
    if side == "ai":
        if row["provenance"] not in AI_CHAT_DERIVED_PROVENANCE:
            return None
        if offline_overlap >= OFFLINE_COPY_OVERLAP:
            return ("archive", AI_OFFLINE_COPY)
        if row["level"] != 1 or (main, sub) == _RELATIONSHIP_MILESTONE:
            return None
        if main in _AI_PERSONA_MAINS:
            return ("archive", AI_PERSONA_CLAIM)
        return ("demote", AI_CHAT_L1)
    if row["level"] != 1:
        return None
    requirement = l1_promotion_requirement(main, sub)
    if requirement == "never":
        return ("demote", USER_NON_CORE)
    if (requirement == "sustained"
            and float(row["importance"]) < L1_MIN_IMPORTANCE
            and contribution_days < L1_PROMOTION_MIN_CONTRIBUTION_DAYS):
        return ("demote", USER_UNSUSTAINED)
    return None


# ── 读取 ─────────────────────────────────────────────────────────────────


def _scope_sql(params: list, *, workspace: str | None, column: str = "workspace_id") -> str:
    if not workspace:
        return ""
    params.append(workspace)
    return f" AND {column} = ${len(params)}"


async def _contribution_days(conn, ids: list[str]) -> dict[str, int]:
    if not ids:
        return {}
    day = local_date_sql("created_at")
    rows = await conn.query_raw(
        f"""SELECT memory_id, COUNT(DISTINCT {day})::int AS days
            FROM memory_usage_receipts
            WHERE memory_side = 'user' AND memory_id = ANY($1::text[]) AND contributed
            GROUP BY memory_id""",
        ids,
    )
    return {r["memory_id"]: int(r["days"]) for r in rows}


async def _offline_copy_matches(conn, *, workspace: str | None = None,
                                memory_ids: list[str] | None = None) -> dict[str, tuple[float, str]]:
    """AI 记忆 → (最高重合度, 对应的线下文案消息 id)。"""
    params: list = [sorted(AI_CHAT_DERIVED_PROVENANCE)]
    where = ""
    if memory_ids is not None:
        params.append(memory_ids)
        where = f" AND m.id = ANY(${len(params)}::text[])"
    where += _scope_sql(params, workspace=workspace, column="m.workspace_id")
    rows = await conn.query_raw(
        f"""SELECT m.id, m.content, msg.id AS message_id, msg.content AS message_content
            FROM memories_ai m
            JOIN memory_evidence_links e ON e.ai_memory_id = m.id
            JOIN messages msg ON msg.id = e.source_message_id
            WHERE NOT m.is_archived AND m.provenance = ANY($1::text[])
              AND msg.role = 'assistant'
              AND msg.metadata->>'trigger_type' LIKE 'offline!_activity!_%' ESCAPE '!'
              {where}""",
        *params,
    )
    best: dict[str, tuple[float, str]] = {}
    for r in rows:
        overlap = copy_overlap(r["content"], r["message_content"])
        if overlap > best.get(r["id"], (-1.0, ""))[0]:
            best[r["id"]] = (overlap, r["message_id"])
    return best


def _item(side: str, row: dict, action: str, reason: str, **extra) -> dict:
    return {
        "side": side, "id": row["id"], "user_id": row["user_id"],
        "workspace_id": row["workspace_id"], "action": action, "reason": reason,
        "from_level": row["level"],
        "to_level": 2 if action == "demote" else row["level"],
        "main_category": row["main_category"], "sub_category": row["sub_category"],
        "importance": float(row["importance"]), "provenance": row["provenance"],
        "content_sha256": content_hash(row["content"]),
        "preview": (row["content"] or "")[:_PREVIEW_CHARS],
        **extra,
    }


async def build_plan(conn, *, workspace: str | None = None) -> tuple[list[dict], Counter]:
    """返回 (计划条目, 保留不动的 L1 计数)。只读。"""
    items: list[dict] = []
    kept: Counter = Counter()

    params: list = []
    user_rows = await conn.query_raw(
        f"SELECT {_ROW_COLUMNS} FROM memories_user WHERE level = 1 AND NOT is_archived"
        + _scope_sql(params, workspace=workspace) + " ORDER BY created_at, id",
        *params,
    )
    days = await _contribution_days(conn, [
        r["id"] for r in user_rows
        if l1_promotion_requirement(r["main_category"], r["sub_category"]) == "sustained"
    ])
    for r in user_rows:
        verdict = classify("user", r, contribution_days=days.get(r["id"], 0))
        if verdict:
            items.append(_item("user", r, *verdict, contribution_days=days.get(r["id"], 0)))
        else:
            kept[f"user {r['main_category']}/{r['sub_category']}"] += 1

    matches = await _offline_copy_matches(conn, workspace=workspace)
    params = [sorted(AI_CHAT_DERIVED_PROVENANCE)]
    copy_ids = [mid for mid, (overlap, _) in matches.items() if overlap >= OFFLINE_COPY_OVERLAP]
    params.append(copy_ids)
    ai_rows = await conn.query_raw(
        f"""SELECT {_ROW_COLUMNS} FROM memories_ai
            WHERE NOT is_archived AND provenance = ANY($1::text[])
              AND (level = 1 OR id = ANY($2::text[]))"""
        + _scope_sql(params, workspace=workspace) + " ORDER BY created_at, id",
        *params,
    )
    for r in ai_rows:
        overlap, message_id = matches.get(r["id"], (0.0, None))
        verdict = classify("ai", r, offline_overlap=overlap)
        if verdict:
            extra = ({"evidence_message_id": message_id, "overlap": round(overlap, 3)}
                     if verdict[1] == AI_OFFLINE_COPY else {})
            items.append(_item("ai", r, *verdict, **extra))
        elif r["level"] == 1:
            kept[f"ai {r['main_category']}/{r['sub_category']}"] += 1
    return items, kept


async def dry_run(*, workspace: str | None = None) -> tuple[list[dict], Counter]:
    async with db.tx(timeout=timedelta(minutes=2)) as tx:
        # 数据库层面保证演练不写: 这个事务里任何 INSERT/UPDATE 都会被拒绝。
        await tx.execute_raw("SET TRANSACTION READ ONLY")
        return await build_plan(tx, workspace=workspace)


# ── 执行 ─────────────────────────────────────────────────────────────────


def _state(side: str, level: int, archived: bool, reason: str | None = None) -> str:
    text = f"side={side} level={level} archived={'true' if archived else 'false'}"
    return f"{text} reason={reason}" if reason else text


def _parse_state(text: str | None) -> dict[str, str]:
    return dict(piece.split("=", 1) for piece in (text or "").split() if "=" in piece)


async def _apply_scope(side: str, user_id: str, workspace_id: str | None,
                       items: list[dict], stats: Counter) -> list[str]:
    table = TABLES[side]
    changed: list[str] = []
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key(side, user_id, workspace_id),
        )
        rows = {r["id"]: r for r in await tx.query_raw(
            f"""SELECT {_ROW_COLUMNS} FROM {table}
                WHERE id = ANY($1::text[]) AND user_id = $2
                  AND workspace_id IS NOT DISTINCT FROM $3::text
                ORDER BY id FOR UPDATE""",
            [i["id"] for i in items], user_id, workspace_id,
        )}
        days = await _contribution_days(tx, [
            i["id"] for i in items if i["reason"] == USER_UNSUSTAINED
        ]) if side == "user" else {}
        overlaps: dict[str, float] = {}
        evidence = {i["evidence_message_id"] for i in items if i.get("evidence_message_id")}
        if evidence:
            messages = {m["id"]: m["content"] for m in await tx.query_raw(
                "SELECT id, content FROM messages WHERE id = ANY($1::text[])", sorted(evidence),
            )}
            for i in items:
                if i.get("evidence_message_id") and i["id"] in rows:
                    overlaps[i["id"]] = copy_overlap(
                        rows[i["id"]]["content"], messages.get(i["evidence_message_id"]),
                    )
        demote, archive, logs = [], [], []
        for item in items:
            row = rows.get(item["id"])
            if row is None:
                stats["skipped:missing"] += 1
                continue
            if content_hash(row["content"]) != item["content_sha256"]:
                stats["skipped:content_changed"] += 1
                continue
            verdict = classify(
                side, row, contribution_days=days.get(row["id"], 0),
                offline_overlap=overlaps.get(row["id"], 0.0),
            )
            if verdict != (item["action"], item["reason"]):
                stats["skipped:rule_no_longer_applies"] += 1
                continue
            new_level = 2 if item["action"] == "demote" else row["level"]
            (demote if item["action"] == "demote" else archive).append(row["id"])
            logs.append((
                str(uuid.uuid4()), row["id"],
                _state(side, row["level"], False),
                _state(side, new_level, item["action"] == "archive", item["reason"]),
            ))
            stats[f"applied:{item['reason']}"] += 1
        if logs:
            await tx.execute_raw(
                f"""INSERT INTO memory_changelogs
                      (id, user_id, workspace_id, memory_id, operation, old_value, new_value)
                    SELECT unnest($1::text[]), $2, $3, unnest($4::text[]), '{OPERATION}',
                           unnest($5::text[]), unnest($6::text[])""",
                [x[0] for x in logs], user_id, workspace_id,
                [x[1] for x in logs], [x[2] for x in logs], [x[3] for x in logs],
            )
        if demote:
            await tx.execute_raw(
                f"UPDATE {table} SET level = 2 WHERE id = ANY($1::text[]) AND level = 1",
                demote,
            )
        if archive:
            await tx.execute_raw(
                f"UPDATE {table} SET is_archived = TRUE WHERE id = ANY($1::text[])", archive,
            )
        changed = demote + archive
    return changed


async def _after_commit(user_id: str, workspace_id: str | None, memory_ids: list[str]) -> None:
    """检索缓存按 scope 失效; 记忆详情页的质量状态顺带刷新。都是尽力而为。"""
    if not memory_ids:
        return
    from app.services.memory.lifecycle.quality_state import refresh_quality_state_for_changelog
    from app.services.memory.storage.repo import invalidate_scope

    await invalidate_scope(user_id, workspace_id)
    for memory_id in memory_ids:
        await refresh_quality_state_for_changelog(memory_id)


def _scopes(items: list[dict]) -> dict[tuple[str, str, str | None], list[dict]]:
    grouped: dict[tuple[str, str, str | None], list[dict]] = defaultdict(list)
    for item in items:
        grouped[(item["side"], item["user_id"], item["workspace_id"])].append(item)
    return grouped


async def apply_plan(items: list[dict], *, reasons: set[str] | None = None) -> Counter:
    stats: Counter = Counter()
    selected = [i for i in items if not reasons or i["reason"] in reasons]
    stats["not_selected"] = len(items) - len(selected)
    for (side, user_id, workspace_id), scope_items in _scopes(selected).items():
        changed = await _apply_scope(side, user_id, workspace_id, scope_items, stats)
        await _after_commit(user_id, workspace_id, changed)
    return stats


async def revert(*, apply: bool, memory_ids: list[str] | None = None) -> Counter:
    """按 changelog 还原; 只处理当前状态仍等于脚本写入状态、且还没还原过的行。"""
    stats: Counter = Counter()
    params: list = [OPERATION, REVERT_OPERATION]
    where = ""
    if memory_ids is not None:
        params.append(memory_ids)
        where = f" AND c.memory_id = ANY(${len(params)}::text[])"
    entries = await db.query_raw(
        f"""SELECT c.memory_id, c.user_id, c.workspace_id, c.old_value, c.new_value
            FROM memory_changelogs c
            WHERE c.operation = $1 {where}
              AND NOT EXISTS (
                SELECT 1 FROM memory_changelogs r
                WHERE r.memory_id = c.memory_id AND r.operation = $2
                  AND r.created_at >= c.created_at)
            ORDER BY c.created_at""",
        *params,
    )
    grouped: dict[tuple[str, str, str | None], list[dict]] = defaultdict(list)
    for entry in entries:
        side = _parse_state(entry["new_value"]).get("side")
        if side not in TABLES:
            stats["skipped:unparseable"] += 1
            continue
        grouped[(side, entry["user_id"], entry["workspace_id"])].append(entry)
    for (side, user_id, workspace_id), scope_entries in grouped.items():
        table = TABLES[side]
        restored: list[str] = []
        async with db.tx(timeout=timedelta(seconds=30)) as tx:
            if not apply:
                await tx.execute_raw("SET TRANSACTION READ ONLY")
            else:
                await tx.execute_raw(
                    "SELECT pg_advisory_xact_lock($1::bigint)",
                    scope_lock_key(side, user_id, workspace_id),
                )
            rows = {r["id"]: r for r in await tx.query_raw(
                f"SELECT id, level, is_archived FROM {table} WHERE id = ANY($1::text[])"
                + (" ORDER BY id FOR UPDATE" if apply else ""),
                [e["memory_id"] for e in scope_entries],
            )}
            logs = []
            for entry in scope_entries:
                old, new = _parse_state(entry["old_value"]), _parse_state(entry["new_value"])
                row = rows.get(entry["memory_id"])
                if row is None:
                    stats["skipped:missing"] += 1
                    continue
                if (str(row["level"]), "true" if row["is_archived"] else "false") != (
                        new.get("level"), new.get("archived")):
                    stats["skipped:changed_since_repair"] += 1
                    continue
                stats["restorable"] += 1
                if not apply:
                    continue
                await tx.execute_raw(
                    f"UPDATE {table} SET level = $1, is_archived = $2 WHERE id = $3",
                    int(old["level"]), old.get("archived") == "true", entry["memory_id"],
                )
                logs.append((str(uuid.uuid4()), entry["memory_id"], entry["new_value"],
                             entry["old_value"]))
                restored.append(entry["memory_id"])
            if logs:
                await tx.execute_raw(
                    f"""INSERT INTO memory_changelogs
                          (id, user_id, workspace_id, memory_id, operation, old_value, new_value)
                        SELECT unnest($1::text[]), $2, $3, unnest($4::text[]),
                               '{REVERT_OPERATION}', unnest($5::text[]), unnest($6::text[])""",
                    [x[0] for x in logs], user_id, workspace_id,
                    [x[1] for x in logs], [x[2] for x in logs], [x[3] for x in logs],
                )
        stats["reverted"] += len(restored)
        await _after_commit(user_id, workspace_id, restored)
    return stats


# ── 报告 ─────────────────────────────────────────────────────────────────


def report(items: list[dict], kept: Counter, *, samples: int) -> str:
    lines = [f"待处理 {len(items)} 条; 范围内保留的 L1 {sum(kept.values())} 条"]
    by_reason = Counter(i["reason"] for i in items)
    for reason in REASONS:
        group = [i for i in items if i["reason"] == reason]
        if not group:
            continue
        action = group[0]["action"]
        scopes = len({(i["user_id"], i["workspace_id"]) for i in group})
        lines.append(f"\n[{reason}] {by_reason[reason]} 条 ({action}, 涉及 {scopes} 个 workspace)")
        categories = Counter(f"{i['main_category']}/{i['sub_category']}" for i in group)
        lines.append("  类目: " + ", ".join(f"{k} {v}" for k, v in categories.most_common(8)))
        for i in group[:samples]:
            extra = f" 重合 {i['overlap']}" if "overlap" in i else (
                f" 注入天数 {i['contribution_days']}" if reason == USER_UNSUSTAINED else "")
            lines.append(
                f"  - {i['side']} {i['id'][:8]} L{i['from_level']}→L{i['to_level']} "
                f"imp={i['importance']:.2f}{extra} | {i['preview']}"
            )
    lines.append("\n保留的 L1 (用户侧全部 + AI 侧聊天衍生; 人设种子不在范围内), 前 10 类: " + ", ".join(
        f"{k} {v}" for k, v in kept.most_common(10)))
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="存量 L1 清理 (缺省只读演练)")
    parser.add_argument("--apply", action="store_true", help="真正写库 (缺省只读)")
    parser.add_argument("--plan", type=Path, help="执行/还原时读取的已审核计划文件")
    parser.add_argument("--plan-out", type=Path, default=Path("l1_core_repair_plan.json"),
                        help="演练产出的计划文件")
    parser.add_argument("--workspace", help="只处理某个 workspace")
    parser.add_argument("--reasons", help="只执行这些原因, 逗号分隔")
    parser.add_argument("--revert", action="store_true", help="按 changelog 还原")
    parser.add_argument("--samples", type=int, default=8, help="报告里每类的样例数")
    args = parser.parse_args()
    reasons = set(filter(None, (args.reasons or "").split(","))) or None
    if reasons and not reasons <= set(REASONS):
        parser.error(f"未知原因: {sorted(reasons - set(REASONS))}")
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if plan is not None and plan.get("schema") != PLAN_SCHEMA:
        parser.error(f"计划文件 schema 不是 {PLAN_SCHEMA}")
    if args.apply and not args.revert and plan is None:
        parser.error("--apply 必须配合审核过的 --plan; 先不带 --apply 跑一遍生成计划")

    await db.connect()
    try:
        if args.revert:
            ids = [i["id"] for i in plan["items"]] if plan else None
            stats = await revert(apply=args.apply, memory_ids=ids)
            print(("已还原" if args.apply else "只读预览还原") + f": {dict(stats)}")
        elif args.apply and plan is not None:
            stats = await apply_plan(plan["items"], reasons=reasons)
            print(f"执行完成: {dict(sorted(stats.items()))}")
        else:
            items, kept = await dry_run(workspace=args.workspace)
            args.plan_out.write_text(json.dumps({
                "schema": PLAN_SCHEMA,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "workspace": args.workspace,
                "counts": dict(Counter(i["reason"] for i in items)),
                "items": items,
            }, ensure_ascii=False, indent=1))
            print(report(items, kept, samples=args.samples))
            print(f"\n计划已写入 {args.plan_out} (只读演练, 数据库未改动)")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
