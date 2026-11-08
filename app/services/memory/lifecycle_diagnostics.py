"""记忆生命周期诊断 (O01.05.V03): 当前库存按来源/层级/归档分布, 和维护作业进度。

复用现有表, 不重复建设 (来源/版本证据、抽取回执、检索 Trace 已各有入口, 见
memory_repairs / runtime_jobs / traces)。这里补两块原来没有聚合口的:

- inventory: memories_user / memories_ai 的当前条数, 按 level、provenance、是否归档分布,
  可按 user/workspace/side 过滤; 每个计数都带分母 (该过滤范围内的总数) 和采样时刻。
- maintenance: memory_consolidation_runs 的近窗口进度 (hygiene / l3_compression),
  原来只有 ops 不变量在读, 后台看不到。

全部只读, 不返回记忆正文, 只返回计数与时间。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.db import db

_SIDES = {"user": "memories_user", "ai": "memories_ai"}
_MAX_WINDOW_DAYS = 90
_MAX_GROUP_ROWS = 200


def _scope_sql(user_id: str | None, workspace_id: str | None) -> tuple[str, list]:
    clauses, params = [], []
    if user_id:
        params.append(user_id)
        clauses.append(f"user_id = ${len(params)}")
    if workspace_id:
        params.append(workspace_id)
        clauses.append(f"workspace_id = ${len(params)}")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


async def _side_inventory(table: str, where: str, params: list) -> dict:
    rows = await db.query_raw(
        f"""SELECT level, COALESCE(provenance, 'unknown') AS provenance, is_archived, count(*) AS n
            FROM {table}{where}
            GROUP BY level, COALESCE(provenance, 'unknown'), is_archived
            ORDER BY n DESC LIMIT {_MAX_GROUP_ROWS}""",
        *params,
    )
    total = active = archived = 0
    by_level: dict[str, dict[str, int]] = {}
    by_provenance: dict[str, dict[str, int]] = {}
    for row in rows:
        n = int(row["n"])
        total += n
        archived_row = bool(row["is_archived"])
        if archived_row:
            archived += n
        else:
            active += n
        bucket = "archived" if archived_row else "active"
        lvl = by_level.setdefault(f"L{row['level']}", {"active": 0, "archived": 0})
        lvl[bucket] += n
        prov = by_provenance.setdefault(str(row["provenance"]), {"active": 0, "archived": 0})
        prov[bucket] += n
    return {
        "total": total, "active": active, "archived": archived,
        "by_level": _sorted(by_level), "by_provenance": _sorted(by_provenance),
    }


def _sorted(data: dict[str, dict[str, int]]) -> list[dict]:
    return [
        {"key": key, **counts}
        for key, counts in sorted(data.items(), key=lambda kv: (-(kv[1]["active"] + kv[1]["archived"]), kv[0]))
    ]


async def inventory(*, user_id: str | None = None, workspace_id: str | None = None,
                    side: str | None = None) -> dict:
    """当前记忆库存分布。side=user/ai 只看一侧; 不传看两侧。每侧计数的分母是该侧总数。"""
    where, params = _scope_sql(user_id, workspace_id)
    sides = [side] if side in _SIDES else list(_SIDES)
    out: dict[str, dict] = {}
    for name in sides:
        out[name] = await _side_inventory(_SIDES[name], where, params)
    return {
        "scope": {"user_id": user_id, "workspace_id": workspace_id, "side": side or "both"},
        "sides": out,
        # 每侧的 total 就是该侧这个过滤范围内的分母; 这里再给一个合计便于展示
        "denominator": sum(s["total"] for s in out.values()),
        "sampled_at": datetime.now(UTC).isoformat(),
    }


async def maintenance(*, user_id: str | None = None, workspace_id: str | None = None,
                      days: int = 7) -> dict:
    """维护作业 (hygiene / l3_compression) 近 days 天的运行进度。"""
    days = max(1, min(days, _MAX_WINDOW_DAYS))
    since = datetime.now(UTC) - timedelta(days=days)
    # created_at 是不带时区的 UTC 时间戳; 参数按文本传入, 显式转换
    clauses = ["created_at >= ($1::timestamptz AT TIME ZONE 'UTC')"]
    params: list = [since.isoformat()]
    if user_id:
        params.append(user_id)
        clauses.append(f"user_id = ${len(params)}")
    if workspace_id:
        params.append(workspace_id)
        clauses.append(f"workspace_id = ${len(params)}")
    where = " WHERE " + " AND ".join(clauses)
    rows = await db.query_raw(
        f"""SELECT job, status, count(*) AS runs,
                   COALESCE(sum(checked), 0) AS checked, COALESCE(sum(archived), 0) AS archived,
                   COALESCE(sum(merged), 0) AS merged, COALESCE(sum(updated), 0) AS updated,
                   COALESCE(sum(errors), 0) AS errors, max(created_at) AS last_run
            FROM memory_consolidation_runs{where}
            GROUP BY job, status ORDER BY job, runs DESC LIMIT {_MAX_GROUP_ROWS}""",
        *params,
    )
    jobs: dict[str, dict] = {}
    for row in rows:
        job = jobs.setdefault(str(row["job"]), {"runs": 0, "checked": 0, "archived": 0,
                                                "merged": 0, "updated": 0, "errors": 0,
                                                "last_run": None, "by_status": {}})
        for field in ("runs", "checked", "archived", "merged", "updated", "errors"):
            job[field] += int(row[field])
        job["by_status"][str(row["status"])] = int(row["runs"])
        last = row["last_run"]
        iso = last.isoformat() if hasattr(last, "isoformat") else (str(last) if last else None)
        if iso and (job["last_run"] is None or iso > job["last_run"]):
            job["last_run"] = iso
    return {
        "scope": {"user_id": user_id, "workspace_id": workspace_id},
        "window": {"days": days, "since": since.isoformat()},
        "jobs": [{"job": name, **data} for name, data in sorted(jobs.items())],
        "sampled_at": datetime.now(UTC).isoformat(),
    }
