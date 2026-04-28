"""Admin 后台 统计概览 — token 用量 + 费用聚合."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.llm.pricing import QWEN_PRICING_CNY_PER_1M, estimate_cost_cny

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/stats", tags=["admin-stats"])


def _window_start(days: int | None) -> datetime | None:
    if not days or days <= 0:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)


@router.get("/token-usage")
async def token_usage(
    days: int = Query(30, ge=0, le=365),
    agent_id: str | None = Query(None),
    _: dict = Depends(require_admin_jwt),
):
    """返回时间窗内的 token 用量与费用聚合.

    days=0 表示"全部历史"; 默认 30 天.
    agent_id 不传 = 跨 agent; 传了 = 单 agent drill-down.
    """
    start = _window_start(days)
    end = datetime.now(timezone.utc)

    # 构造 WHERE — 同时给两个版本: 无表别名的 (totals/by_model/daily 用) +
    # u. 前缀的 (by_agent JOIN 时避免列名跟 ai_agents 冲突).
    # ${N}::timestamp 显式 cast — prisma query_raw 不会自动推, PG 默认按 text 比较.
    base_filters: list[tuple[str, str]] = []  # (bare, prefixed)
    params: list = []
    if start is not None:
        idx = len(params) + 1
        base_filters.append(
            (f"created_at >= ${idx}::timestamp", f"u.created_at >= ${idx}::timestamp"),
        )
        params.append(start.replace(tzinfo=None).isoformat())
    if agent_id:
        idx = len(params) + 1
        base_filters.append((f"agent_id = ${idx}", f"u.agent_id = ${idx}"))
        params.append(agent_id)
    where_sql = " AND ".join(["1=1"] + [b for b, _ in base_filters])
    where_sql_u = " AND ".join(["1=1"] + [p for _, p in base_filters])

    # 1. totals
    totals_rows = await db.query_raw(
        f"""
        SELECT
            COUNT(*)::int AS request_count,
            COALESCE(SUM(input_tokens), 0)::int AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::int AS output_tokens,
            COALESCE(SUM(cost_cny), 0)::float AS cost_cny,
            COALESCE(SUM(call_count), 0)::int AS call_count
        FROM llm_usage
        WHERE {where_sql}
        """,
        *params,
    )
    totals = totals_rows[0] if totals_rows else {
        "request_count": 0, "input_tokens": 0, "output_tokens": 0,
        "cost_cny": 0.0, "call_count": 0,
    }

    # 2. by_model — jsonb_each 拆开 tokens_by_model
    by_model_rows = await db.query_raw(
        f"""
        SELECT
            kv.key AS model,
            SUM((kv.value->>'input')::int)::int AS input_tokens,
            SUM((kv.value->>'output')::int)::int AS output_tokens
        FROM llm_usage, jsonb_each(tokens_by_model) AS kv
        WHERE {where_sql}
        GROUP BY kv.key
        ORDER BY (SUM((kv.value->>'input')::int) + SUM((kv.value->>'output')::int)) DESC
        """,
        *params,
    )
    by_model = [
        {
            "model": r["model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_cny": round(estimate_cost_cny(
                r["model"], r["input_tokens"], r["output_tokens"],
            ), 6),
        }
        for r in by_model_rows
    ]

    # 3. by_agent — JOIN ai_agents 拿名字, 取 top 50
    by_agent_rows = await db.query_raw(
        f"""
        SELECT
            u.agent_id,
            COALESCE(a.name, '(已删除)') AS agent_name,
            COUNT(*)::int AS request_count,
            COALESCE(SUM(u.input_tokens), 0)::int AS input_tokens,
            COALESCE(SUM(u.output_tokens), 0)::int AS output_tokens,
            COALESCE(SUM(u.cost_cny), 0)::float AS cost_cny
        FROM llm_usage u
        LEFT JOIN ai_agents a ON a.id = u.agent_id
        WHERE {where_sql_u}
        GROUP BY u.agent_id, a.name
        ORDER BY cost_cny DESC
        LIMIT 50
        """,
        *params,
    )

    # 4. daily — 每天 bucket
    daily_rows = await db.query_raw(
        f"""
        SELECT
            DATE_TRUNC('day', created_at)::date AS bucket,
            COALESCE(SUM(input_tokens), 0)::int AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::int AS output_tokens,
            COALESCE(SUM(cost_cny), 0)::float AS cost_cny,
            COUNT(*)::int AS request_count
        FROM llm_usage
        WHERE {where_sql}
        GROUP BY bucket
        ORDER BY bucket ASC
        """,
        *params,
    )
    daily = [
        {
            "date": str(r["bucket"]),
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_cny": round(r["cost_cny"], 6),
            "request_count": r["request_count"],
        }
        for r in daily_rows
    ]

    return {
        "window": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat(),
            "days": days,
        },
        "pricing": {
            model: {"input": p["input"], "output": p["output"], "unit": "CNY per 1M tokens"}
            for model, p in QWEN_PRICING_CNY_PER_1M.items()
        },
        "totals": {**totals, "cost_cny": round(totals["cost_cny"], 6)},
        "by_model": by_model,
        "by_agent": [
            {
                "agent_id": r["agent_id"],
                "agent_name": r["agent_name"],
                "request_count": r["request_count"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_cny": round(r["cost_cny"], 6),
            }
            for r in by_agent_rows
        ],
        "daily": daily,
    }
