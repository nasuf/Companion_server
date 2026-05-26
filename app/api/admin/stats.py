"""Admin 后台 统计概览 — token 用量 + 费用聚合."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.redis_client import get_redis
from app.services.llm.pricing import estimate_cost_cny
from app.services.operations.metrics import summarize_crisis_events, summarize_visible_use
from app.services.proactive import triggers as proactive_triggers
from app.services.runtime import job_queue as runtime_job_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/stats", tags=["admin-stats"])


def _window_start(days: int | None) -> datetime | None:
    if not days or days <= 0:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)


def _build_ops_where(
    *,
    start: datetime | None,
    agent_id: str | None,
    user_id: str | None,
    created_expr: str,
    agent_expr: str | None,
    user_expr: str | None,
) -> tuple[str, list]:
    clauses = ["1=1"]
    params: list = []
    if start is not None:
        params.append(start.replace(tzinfo=None).isoformat())
        clauses.append(f"{created_expr} >= ${len(params)}::timestamp")
    if agent_id and agent_expr:
        params.append(agent_id)
        clauses.append(f"{agent_expr} = ${len(params)}")
    if user_id and user_expr:
        params.append(user_id)
        clauses.append(f"{user_expr} = ${len(params)}")
    return " AND ".join(clauses), params


def _count_by(rows: list[dict], key_field: str, count_field: str = "count") -> dict[str, int]:
    return {
        str(row.get(key_field) or "unknown"): int(row.get(count_field) or 0)
        for row in rows
    }


def _sum_by(rows: list[dict], key_field: str, count_field: str = "count") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(key_field) or "unknown")
        counts[key] = counts.get(key, 0) + int(row.get(count_field) or 0)
    return counts


def _classify_bug_category(error_type: str, reason: str | None) -> str:
    text = f"{error_type} {reason or ''}".lower()
    if any(k in text for k in ("crisis", "安全", "自伤", "轻生", "危险")):
        return "crisis_safety"
    if any(k in text for k in ("memory", "记忆", "失忆", "幻觉", "编造")):
        return "memory_safety"
    if any(k in text for k in ("reminder", "提醒", "日程")):
        return "reminder"
    if any(k in text for k in ("tone", "语气", "人设", "persona", "机器人")):
        return "tone"
    return "bug_report"


def _aggregate_bug_categories(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = _classify_bug_category(
            str(row.get("error_type") or ""),
            row.get("reason"),
        )
        counts[category] = counts.get(category, 0) + int(row.get("count") or 0)
    return counts


def _trace_risk_reasons(row: dict) -> tuple[list[str], int]:
    reasons: list[str] = []
    score = 0
    duration = int(row.get("total_duration_ms") or 0)
    llm_steps = int(row.get("llm_step_count") or 0)
    open_bugs = int(row.get("open_bug_count") or 0)
    share_status = str(row.get("share_status") or "")
    if duration >= 30_000:
        reasons.append("slow_trace_30s")
        score += 3
    elif duration >= 20_000:
        reasons.append("slow_trace_20s")
        score += 2
    if llm_steps >= 10:
        reasons.append("many_llm_steps_10")
        score += 2
    elif llm_steps >= 8:
        reasons.append("many_llm_steps_8")
        score += 1
    if share_status == "failed":
        reasons.append("trace_share_failed")
        score += 2
    if open_bugs > 0:
        reasons.append("open_bug_report")
        score += 4
    return reasons, score


def _serialize_high_risk_trace(row: dict) -> dict:
    reasons, score = _trace_risk_reasons(row)
    created_at = row.get("created_at")
    return {
        "trace_id": row.get("trace_id"),
        "message_id": row.get("message_id"),
        "conversation_id": row.get("conversation_id"),
        "agent_id": row.get("agent_id"),
        "user_id": row.get("user_id"),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
        "total_duration_ms": int(row.get("total_duration_ms") or 0),
        "llm_step_count": int(row.get("llm_step_count") or 0),
        "share_status": row.get("share_status"),
        "open_bug_count": int(row.get("open_bug_count") or 0),
        "risk_reasons": reasons,
        "risk_score": score,
        "root_message": row.get("root_message"),
    }


async def _redis_ops_counts() -> dict:
    """Best-effort Redis queue/DLQ counters for ops dashboards."""
    try:
        redis = await get_redis()
        reminder_dlq, runtime_dlq, runtime_ready, runtime_delayed, runtime_running, runtime_succeeded = await asyncio.gather(
            redis.zcard(proactive_triggers._DLQ_KEY),
            redis.llen(runtime_job_queue._DLQ_KEY),
            redis.llen(runtime_job_queue._READY_KEY),
            redis.zcard(runtime_job_queue._DELAYED_KEY),
            redis.zcard(runtime_job_queue._RUNNING_KEY),
            redis.llen(runtime_job_queue._SUCCEEDED_KEY),
        )
        return {
            "ok": True,
            "reminder_dlq_count": int(reminder_dlq or 0),
            "runtime_jobs": {
                "ready_count": int(runtime_ready or 0),
                "delayed_count": int(runtime_delayed or 0),
                "running_count": int(runtime_running or 0),
                "dlq_count": int(runtime_dlq or 0),
                "succeeded_count": int(runtime_succeeded or 0),
            },
        }
    except Exception as e:
        logger.warning("ops stats redis counters unavailable: %s", e)
        return {
            "ok": False,
            "error": type(e).__name__,
            "reminder_dlq_count": 0,
            "runtime_jobs": {
                "ready_count": 0,
                "delayed_count": 0,
                "running_count": 0,
                "dlq_count": 0,
                "succeeded_count": 0,
            },
        }


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
    # 价格表从 model_registry 取 (含 disabled — disabled 模型可能仍在历史 usage 行里).
    pricing_rows = await db.modelregistry.find_many()

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

    # 5 段聚合查询互相独立, asyncio.gather 并发跑省 dashboard 加载延迟.
    totals_rows, by_model_rows, by_agent_rows, by_scope_rows, daily_rows = await asyncio.gather(
        db.query_raw(
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
        ),
        db.query_raw(
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
        ),
        db.query_raw(
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
        ),
        db.query_raw(
            f"""
            SELECT
                scope,
                COUNT(*)::int AS request_count,
                COALESCE(SUM(input_tokens), 0)::int AS input_tokens,
                COALESCE(SUM(output_tokens), 0)::int AS output_tokens,
                COALESCE(SUM(cost_cny), 0)::float AS cost_cny
            FROM llm_usage
            WHERE {where_sql}
            GROUP BY scope
            ORDER BY cost_cny DESC
            """,
            *params,
        ),
        db.query_raw(
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
        ),
    )
    totals = totals_rows[0] if totals_rows else {
        "request_count": 0, "input_tokens": 0, "output_tokens": 0,
        "cost_cny": 0.0, "call_count": 0,
    }
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
    by_scope = [
        {
            "scope": r["scope"],
            "request_count": r["request_count"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_cny": round(r["cost_cny"], 6),
        }
        for r in by_scope_rows
    ]

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
            f"{r.provider}/{r.identifier}": {
                "input": r.inputCostPerMillion or 0.0,
                "output": r.outputCostPerMillion or 0.0,
                "unit": "CNY per 1M tokens",
            }
            for r in pricing_rows
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
        "by_scope": by_scope,
        "daily": daily,
    }


@router.get("/operations")
async def operations(
    days: int = Query(7, ge=0, le=365),
    agent_id: str | None = Query(None),
    user_id: str | None = Query(None),
    _: dict = Depends(require_admin_jwt),
):
    """返回 P1 运营健康指标聚合.

    days=0 表示"全部历史". Redis 队列/DLQ 是当前状态, 不受 days 限制.
    """
    start = _window_start(days)
    end = datetime.now(timezone.utc)
    now_naive = end.replace(tzinfo=None).isoformat()
    next_24h_naive = (end + timedelta(hours=24)).replace(tzinfo=None).isoformat()

    memory_where, memory_params = _build_ops_where(
        start=start,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="m.created_at",
        agent_expr="w.agent_id",
        user_expr="m.user_id",
    )
    llm_where, llm_params = _build_ops_where(
        start=start,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="l.created_at",
        agent_expr="l.agent_id",
        user_expr="l.user_id",
    )
    proactive_where, proactive_params = _build_ops_where(
        start=start,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="p.created_at",
        agent_expr="p.agent_id",
        user_expr="p.user_id",
    )
    proactive_state_where, proactive_state_params = _build_ops_where(
        start=None,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="s.created_at",
        agent_expr="s.agent_id",
        user_expr="s.user_id",
    )
    reminder_where, reminder_params = _build_ops_where(
        start=None,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="t.created_at",
        agent_expr="t.ai_agent_id",
        user_expr="t.user_id",
    )
    reminder_recent_where, reminder_recent_params = _build_ops_where(
        start=start,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="COALESCE(t.last_fired, t.created_at)",
        agent_expr="t.ai_agent_id",
        user_expr="t.user_id",
    )
    bug_where, bug_params = _build_ops_where(
        start=start,
        agent_id=agent_id,
        user_id=user_id,
        created_expr="b.created_at",
        agent_expr="c.agent_id",
        user_expr="c.user_id",
    )
    trace_where, trace_params = _build_ops_where(
        start=end - timedelta(hours=24),
        agent_id=agent_id,
        user_id=user_id,
        created_expr="mt.created_at",
        agent_expr="c.agent_id",
        user_expr="c.user_id",
    )
    reminder_now_idx = len(reminder_params) + 1
    reminder_next_idx = len(reminder_params) + 2

    (
        memory_ops_rows,
        llm_totals_rows,
        llm_scope_rows,
        proactive_event_rows,
        proactive_trigger_rows,
        proactive_state_rows,
        reminder_rows,
        reminder_recent_rows,
        bug_rows,
        bug_error_rows,
        high_risk_trace_rows,
        visible_use_summary,
        crisis_summary,
        redis_counts,
    ) = await asyncio.gather(
        db.query_raw(
            f"""
            SELECT m.operation, COUNT(*)::int AS count
            FROM memory_changelogs m
            LEFT JOIN chat_workspaces w ON w.id = m.workspace_id
            WHERE {memory_where}
            GROUP BY m.operation
            ORDER BY count DESC
            """,
            *memory_params,
        ),
        db.query_raw(
            f"""
            SELECT
                COUNT(*)::int AS request_count,
                COALESCE(SUM(l.call_count), 0)::int AS call_count,
                COALESCE(SUM(l.input_tokens), 0)::int AS input_tokens,
                COALESCE(SUM(l.output_tokens), 0)::int AS output_tokens,
                COALESCE(SUM(l.cost_cny), 0)::float AS cost_cny,
                COALESCE(SUM(l.latency_ms_total), 0)::int AS latency_ms_total,
                COALESCE(SUM(l.latency_count), 0)::int AS latency_count,
                COALESCE(SUM(l.failure_count), 0)::int AS failure_count,
                COALESCE(SUM(l.fallback_count), 0)::int AS fallback_count,
                COALESCE(SUM(l.circuit_open_count), 0)::int AS circuit_open_count
            FROM llm_usage l
            WHERE {llm_where}
            """,
            *llm_params,
        ),
        db.query_raw(
            f"""
            SELECT l.scope, COUNT(*)::int AS request_count
            FROM llm_usage l
            WHERE {llm_where}
            GROUP BY l.scope
            ORDER BY request_count DESC
            """,
            *llm_params,
        ),
        db.query_raw(
            f"""
            SELECT p.event_type, COUNT(*)::int AS count
            FROM proactive_event_logs p
            WHERE {proactive_where}
            GROUP BY p.event_type
            ORDER BY count DESC
            """,
            *proactive_params,
        ),
        db.query_raw(
            f"""
            SELECT COALESCE(p.trigger_type, 'unknown') AS trigger_type, COUNT(*)::int AS count
            FROM proactive_event_logs p
            WHERE {proactive_where}
            GROUP BY COALESCE(p.trigger_type, 'unknown')
            ORDER BY count DESC
            """,
            *proactive_params,
        ),
        db.query_raw(
            f"""
            SELECT s.status, COUNT(*)::int AS count
            FROM proactive_states s
            WHERE {proactive_state_where}
            GROUP BY s.status
            ORDER BY count DESC
            """,
            *proactive_state_params,
        ),
        db.query_raw(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE t.action_type = 'reminder')::int AS total_count,
                COUNT(*) FILTER (
                    WHERE t.action_type = 'reminder' AND t.is_active = true
                )::int AS active_count,
                COUNT(*) FILTER (
                    WHERE t.action_type = 'reminder'
                      AND t.is_active = true
                      AND t.trigger_time < ${reminder_now_idx}::timestamp
                )::int AS overdue_active_count,
                COUNT(*) FILTER (
                    WHERE t.action_type = 'reminder'
                      AND t.is_active = true
                      AND t.trigger_time >= ${reminder_now_idx}::timestamp
                      AND t.trigger_time < ${reminder_next_idx}::timestamp
                )::int AS due_next_24h_count
            FROM time_triggers t
            WHERE {reminder_where}
            """,
            *reminder_params,
            now_naive,
            next_24h_naive,
        ),
        db.query_raw(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE t.action_type = 'reminder' AND t.last_fired IS NOT NULL
                )::int AS fired_count
            FROM time_triggers t
            WHERE {reminder_recent_where}
            """,
            *reminder_recent_params,
        ),
        db.query_raw(
            f"""
            SELECT b.status, COUNT(*)::int AS count
            FROM bug_reports b
            JOIN messages m ON m.id = b.message_id
            JOIN conversations c ON c.id = m.conversation_id
            WHERE {bug_where}
            GROUP BY b.status
            ORDER BY count DESC
            """,
            *bug_params,
        ),
        db.query_raw(
            f"""
            SELECT
                COALESCE(NULLIF(TRIM(et.error_type), ''), 'uncategorized') AS error_type,
                b.reason,
                COUNT(*)::int AS count
            FROM bug_reports b
            JOIN messages m ON m.id = b.message_id
            JOIN conversations c ON c.id = m.conversation_id
            LEFT JOIN LATERAL unnest(
                CASE
                    WHEN array_length(b.error_types, 1) IS NULL
                    THEN ARRAY['uncategorized']::text[]
                    ELSE b.error_types
                END
            ) AS et(error_type) ON true
            WHERE {bug_where}
            GROUP BY COALESCE(NULLIF(TRIM(et.error_type), ''), 'uncategorized'), b.reason
            ORDER BY count DESC
            LIMIT 50
            """,
            *bug_params,
        ),
        db.query_raw(
            f"""
            SELECT
                mt.trace_id,
                mt.message_id,
                mt.conversation_id,
                c.agent_id,
                c.user_id,
                mt.root_message,
                mt.total_duration_ms,
                mt.llm_step_count,
                mt.share_status,
                mt.created_at,
                COALESCE(open_bugs.open_count, 0)::int AS open_bug_count
            FROM message_traces mt
            LEFT JOIN conversations c ON c.id = mt.conversation_id
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::int AS open_count
                FROM bug_reports br
                WHERE br.message_id = mt.message_id
                  AND br.status = 'open'
            ) open_bugs ON true
            WHERE {trace_where}
              AND (
                COALESCE(mt.total_duration_ms, 0) >= 20000
                OR COALESCE(mt.llm_step_count, 0) >= 8
                OR mt.share_status = 'failed'
                OR COALESCE(open_bugs.open_count, 0) > 0
              )
            ORDER BY
                COALESCE(open_bugs.open_count, 0) DESC,
                COALESCE(mt.total_duration_ms, 0) DESC,
                COALESCE(mt.llm_step_count, 0) DESC,
                mt.created_at DESC
            LIMIT 20
            """,
            *trace_params,
        ),
        summarize_visible_use(start=start, agent_id=agent_id, user_id=user_id, db_client=db),
        summarize_crisis_events(start=start, agent_id=agent_id, user_id=user_id, db_client=db),
        _redis_ops_counts(),
    )

    memory_ops = _count_by(memory_ops_rows, "operation")
    proactive_events = _count_by(proactive_event_rows, "event_type")
    proactive_triggers = _count_by(proactive_trigger_rows, "trigger_type")
    proactive_states = _count_by(proactive_state_rows, "status")
    bug_statuses = _count_by(bug_rows, "status")
    bug_error_types = _sum_by(bug_error_rows, "error_type")
    bug_categories = _aggregate_bug_categories(bug_error_rows)
    high_risk_traces = [_serialize_high_risk_trace(row) for row in high_risk_trace_rows]

    llm_totals = llm_totals_rows[0] if llm_totals_rows else {
        "request_count": 0,
        "call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_cny": 0.0,
        "latency_ms_total": 0,
        "latency_count": 0,
        "failure_count": 0,
        "fallback_count": 0,
        "circuit_open_count": 0,
    }
    reminder = reminder_rows[0] if reminder_rows else {
        "total_count": 0,
        "active_count": 0,
        "overdue_active_count": 0,
        "due_next_24h_count": 0,
    }
    reminder_recent = reminder_recent_rows[0] if reminder_recent_rows else {"fired_count": 0}

    correction_ops = {
        "user_edit",
        "contradiction_archived",
        "contradiction_new",
        "retrieval_feedback_confirmed",
    }
    deletion_ops = {"delete", "user_bulk_delete", "workspace_wipe"}
    llm_latency_count = int(llm_totals["latency_count"] or 0)
    llm_call_count = int(llm_totals["call_count"] or 0)
    llm_event_denominator = llm_latency_count or llm_call_count
    llm_fallback_count = int(llm_totals["fallback_count"] or 0)
    llm_failure_count = int(llm_totals["failure_count"] or 0)

    return {
        "window": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat(),
            "days": days,
        },
        "filters": {
            "agent_id": agent_id,
            "user_id": user_id,
        },
        "llm": {
            "request_count": int(llm_totals["request_count"] or 0),
            "call_count": int(llm_totals["call_count"] or 0),
            "input_tokens": int(llm_totals["input_tokens"] or 0),
            "output_tokens": int(llm_totals["output_tokens"] or 0),
            "cost_cny": round(float(llm_totals["cost_cny"] or 0.0), 6),
            "latency_ms_total": int(llm_totals["latency_ms_total"] or 0),
            "latency_count": llm_latency_count,
            "avg_latency_ms": round(
                int(llm_totals["latency_ms_total"] or 0) / llm_latency_count,
                2,
            ) if llm_latency_count else 0.0,
            "failure_count": llm_failure_count,
            "fallback_count": llm_fallback_count,
            "fallback_rate": round(llm_fallback_count / llm_event_denominator, 4)
            if llm_event_denominator else 0.0,
            "circuit_open_count": int(llm_totals["circuit_open_count"] or 0),
            "by_scope": [
                {
                    "scope": row["scope"],
                    "request_count": int(row["request_count"] or 0),
                }
                for row in llm_scope_rows
            ],
        },
        "memory": {
            "by_operation": memory_ops,
            "stored_count": memory_ops.get("insert", 0),
            "retrieval_access_count": memory_ops.get("access", 0),
            "evidence_link_count": memory_ops.get("evidence_linked", 0),
            "correction_count": sum(memory_ops.get(op, 0) for op in correction_ops),
            "deletion_count": sum(memory_ops.get(op, 0) for op in deletion_ops),
            "contradiction_count": sum(
                count for op, count in memory_ops.items()
                if op.startswith("contradiction_")
            ),
            "visible_use": visible_use_summary,
        },
        "proactive": {
            "by_event_type": proactive_events,
            "by_trigger_type": proactive_triggers,
            "state_counts": proactive_states,
            "sent_count": proactive_events.get("message_sent", 0),
            "skipped_count": proactive_events.get("send_skipped", 0)
            + proactive_events.get("window_missed", 0)
            + proactive_events.get("window_deferred", 0),
            "waiting_user_count": proactive_states.get("waiting_user", 0),
        },
        "reminders": {
            "total_count": int(reminder["total_count"] or 0),
            "active_count": int(reminder["active_count"] or 0),
            "overdue_active_count": int(reminder["overdue_active_count"] or 0),
            "due_next_24h_count": int(reminder["due_next_24h_count"] or 0),
            "fired_count": int(reminder_recent["fired_count"] or 0),
            "dlq_count": redis_counts["reminder_dlq_count"],
        },
        "runtime_jobs": redis_counts["runtime_jobs"],
        "bug_reports": {
            "by_status": bug_statuses,
            "by_error_type": bug_error_types,
            "by_eval_category": bug_categories,
            "created_count": sum(bug_statuses.values()),
            "open_count": bug_statuses.get("open", 0),
            "resolved_count": bug_statuses.get("resolved", 0),
        },
        "crisis_events": crisis_summary,
        "high_risk_traces": {
            "window_hours": 24,
            "items": high_risk_traces,
            "count": len(high_risk_traces),
        },
        "data_quality": {
            "redis_available": redis_counts["ok"],
            "llm_latency_available": True,
            "llm_fallback_available": True,
            "notes": [
                "LLM latency/fallback/circuit metrics are session-level aggregates from llm_usage rows.",
                "Redis queue/DLQ counters are point-in-time health signals and ignore the days window.",
                "High-risk traces always use the latest 24h window, independent of the dashboard days filter.",
            ],
        },
    }
