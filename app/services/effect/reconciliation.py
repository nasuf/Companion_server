"""Exact Trace/usage reconciliation, materialized by local day.

Never reconstruct missing supplier usage. Historical traces expire after 30
calendar days; a retained derived row is not evidence that its source survives.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json

from app.services.effect import definitions as defs
from app.services.effect.costs import model_key

ACCOUNTING_VERSION = "trace-linked-v2"
# Official first price bands, verified 2026-10-29. Unknown routes stay unknown.
# k is conservatively represented as 1000 tokens, matching the billing guard.
FIRST_BANDS = {
    "doubao-seed-character-260628": 32000,
    "doubao-seed-2-0-mini-260428": 32000,
    "qwen3.5-flash": 128000,
    "qwen3.5-plus": 128000,
}
EXPECTED_UNLINKED = ("chat_media", "schedule_cron", "agent_creation")


def classify(data: dict) -> str:
    if data["usage_rows"] == 0 and data["trace_input"] + data["trace_output"] > 0:
        return "missing_usage"
    if data["trace_calls"] == 0:
        return "missing_trace"
    if data["unmetered_calls"] or data["unknown_trace_calls"]:
        return "unmetered"
    if data["usage_rows"] == 0:
        return "no_billed_usage"
    if (data["trace_input"], data["trace_output"]) != (
        data["usage_input"],
        data["usage_output"],
    ):
        return "mismatch"
    return "matched"


async def compute(tx, day: date, now: datetime) -> list[dict]:
    from app.services.effect.daily_facts import _row, is_settled

    start, end = defs.local_day_bounds(day)
    prices = {
        r["key"]: r
        for r in await tx.query_raw("""SELECT provider||'/'||identifier AS key
      FROM model_registry WHERE input_cost_per_million IS NOT NULL AND output_cost_per_million IS NOT NULL""")
    }
    # Full trace totals include background work crossing midnight. Anchor once
    # at its earliest run (or earliest usage when no trace was retained).
    records = await tx.query_raw(
        """WITH ids AS (
      SELECT trace_id FROM trace_runs WHERE started_at >= $1::timestamp AND started_at < $2::timestamp
      UNION SELECT trace_id FROM llm_usage WHERE trace_id IS NOT NULL
        AND created_at >= $1::timestamp AND created_at < $2::timestamp),
      anchor AS (
        SELECT i.trace_id,COALESCE((SELECT min(started_at) FROM trace_runs t WHERE t.trace_id=i.trace_id),
          (SELECT min(created_at) FROM llm_usage l WHERE l.trace_id=i.trace_id)) AS started_at FROM ids i)
      SELECT a.trace_id,root.extra_json->'metadata'->>'usage_accounting_version' AS accounting_version,
        COALESCE(c.user_id,lu.user_id) AS user_id,COALESCE(c.agent_id,lu.agent_id) AS agent_id,
        c.id AS conversation_id,COALESCE(lu.scope,CASE WHEN root.name='chat_request' THEN 'chat' ELSE 'offline' END) AS scope,
        (SELECT COALESCE(jsonb_agg(jsonb_build_object('model',model_name,'input',prompt_tokens,
          'output',completion_tokens,'status',status)), '[]'::jsonb) FROM trace_runs t
          WHERE t.trace_id=a.trace_id AND t.run_type='llm') AS calls,
        (SELECT COALESCE(jsonb_agg(jsonb_build_object('input',input_tokens,'output',output_tokens,
          'models',tokens_by_model)), '[]'::jsonb) FROM llm_usage l WHERE l.trace_id=a.trace_id) AS usage
      FROM anchor a LEFT JOIN LATERAL (SELECT name,inputs_json,extra_json FROM trace_runs t
        WHERE t.trace_id=a.trace_id AND t.parent_id IS NULL ORDER BY started_at,run_id LIMIT 1) root ON TRUE
      LEFT JOIN message_traces mt ON mt.trace_id=a.trace_id
      LEFT JOIN conversations c ON c.id=COALESCE(mt.conversation_id,root.inputs_json->>'conversation_id')
      LEFT JOIN LATERAL (SELECT user_id,agent_id,scope FROM llm_usage l WHERE l.trace_id=a.trace_id
        ORDER BY (scope='chat') DESC,created_at,id LIMIT 1) lu ON TRUE
      LEFT JOIN users u ON u.id=COALESCE(c.user_id,lu.user_id)
      WHERE a.started_at >= $1::timestamp AND a.started_at < $2::timestamp
        AND (COALESCE(c.user_id,lu.user_id) IS NULL OR u.status<>'deleting')""",
        start.isoformat(),
        end.isoformat(),
    )
    output = []
    for record in records:
        calls, usage = record.pop("calls"), record.pop("usage")
        model_max = {}
        for call in calls:
            name = call["model"] or "unknown"
            model_max[name] = max(model_max.get(name, 0), int(call["input"] or 0))
        keys = {key for row in usage for key in (row["models"] or {})}
        data = dict(
            trace_calls=len(calls),
            usage_rows=len(usage),
            trace_input=sum(int(c["input"] or 0) for c in calls),
            trace_output=sum(int(c["output"] or 0) for c in calls),
            usage_input=sum(int(u["input"]) for u in usage),
            usage_output=sum(int(u["output"]) for u in usage),
            unknown_trace_calls=sum(
                c["input"] is None and c["output"] is None and c["status"] != "error"
                for c in calls
            ),
            unmetered_calls=sum(
                int(b.get("unmetered_calls") or 0)
                for u in usage
                for b in (u["models"] or {}).values()
            ),
            unpriced_models=sorted(k for k in keys if model_key(k, prices) is None),
            max_input_by_model=model_max,
            band_exceeded=sorted(
                k
                for k, v in model_max.items()
                if k.split("/", 1)[-1] in FIRST_BANDS
                and v > FIRST_BANDS[k.split("/", 1)[-1]]
            ),
        )
        output.append(
            {
                **record,
                "release_qualified": record.pop("accounting_version")
                == ACCOUNTING_VERSION,
                "status": classify(data),
                "data": data,
            }
        )
    await tx.execute_raw(
        "DELETE FROM stats_token_reconciliation WHERE local_date=$1::date", str(day)
    )
    if output:
        await tx.execute_raw(
            """INSERT INTO stats_token_reconciliation
          (trace_id,local_date,user_id,conversation_id,agent_id,scope,status,release_qualified,data,definition_version,computed_at)
          SELECT r.trace_id,$1::date,r.user_id,r.conversation_id,r.agent_id,r.scope,r.status,r.release_qualified,
            r.data,$2,$3::timestamptz FROM jsonb_to_recordset($4::jsonb) r(trace_id text,user_id text,
            conversation_id text,agent_id text,scope text,status text,release_qualified boolean,data jsonb)""",
            str(day),
            defs.DEFINITION_VERSION,
            now.isoformat(),
            json.dumps(output),
        )
    usage_rows = await tx.query_raw(
        """SELECT scope,trace_id,tokens_by_model FROM llm_usage l
      LEFT JOIN users u ON u.id=l.user_id WHERE l.created_at >= $1::timestamp AND l.created_at<$2::timestamp
        AND (l.user_id IS NULL OR u.status<>'deleting')""",
        start.isoformat(),
        end.isoformat(),
    )
    rows = []
    scopes = sorted({r["scope"] for r in output} | {r["scope"] for r in usage_rows})
    for scope in [None, *scopes]:
        traces = [r for r in output if scope is None or r["scope"] == scope]
        usages = [r for r in usage_rows if scope is None or r["scope"] == scope]
        comparable = [
            r for r in traces if r["status"] in ("matched", "mismatch", "missing_usage")
        ]
        values = {
            "token_exact_match": dict(
                numerator=sum(r["status"] == "matched" for r in comparable),
                denominator=len(comparable),
            ),
            "token_missing_usage": dict(
                value=sum(r["status"] == "missing_usage" for r in traces)
            ),
            "token_unmetered_calls": dict(
                value=sum(
                    int(b.get("unmetered_calls") or 0)
                    for r in usages
                    for b in (r["tokens_by_model"] or {}).values()
                )
            ),
            "token_unpriced_models": dict(
                value=len(
                    {
                        k
                        for r in usages
                        for k in (r["tokens_by_model"] or {})
                        if model_key(k, prices) is None
                    }
                )
            ),
            "token_unlinked_rows": dict(value=sum(not r["trace_id"] for r in usages)),
            "token_max_input": dict(
                value=max(
                    (
                        v
                        for r in traces
                        for v in r["data"]["max_input_by_model"].values()
                    ),
                    default=0,
                )
            ),
        }
        for metric, args in values.items():
            rows.append(
                _row(
                    metric,
                    "all",
                    settled=is_settled(metric, day, now),
                    **(
                        {}
                        if scope is None
                        else dict(dimension="scope", dimension_value=scope)
                    ),
                    **args,
                )
            )
    maxima = defaultdict(int)
    for r in output:
        for key, value in r["data"]["max_input_by_model"].items():
            maxima[key] = max(maxima[key], value)
    rows.extend(
        _row(
            "token_max_input",
            "all",
            settled=is_settled("token_max_input", day, now),
            value=value,
            dimension="model",
            dimension_value=key,
        )
        for key, value in maxima.items()
    )
    return rows


async def read(
    start: date,
    end: date,
    *,
    status: str = "",
    agent: str = "",
    page: int = 1,
    page_size: int = 50,
    qualified_only: bool = False,
    now: datetime | None = None,
) -> dict:
    from app.db import db
    from app.services.effect.dashboard import fact_coverage

    allowed = {
        "",
        "missing_usage",
        "missing_trace",
        "unmetered",
        "mismatch",
        "matched",
        "unpriced",
        "band_exceeded",
    }
    if (
        status not in allowed
        or not 1 <= page_size <= 100
        or not 1 <= page <= 100000
        or len(agent) > 128
    ):
        raise ValueError("无效对账筛选")
    today = defs.local_today(now)
    cutoff = today - timedelta(days=29)
    coverage = await fact_coverage(start, end)
    where = """local_date BETWEEN $1::date AND $2::date AND definition_version=$3
      AND ($4='' OR agent_id=$4) AND (NOT $5 OR release_qualified)
      AND (user_id IS NULL OR EXISTS(SELECT 1 FROM users owner WHERE owner.id=user_id AND owner.status<>'deleting'))"""
    args = [
        str(max(start, cutoff)),
        str(end),
        defs.DEFINITION_VERSION,
        agent,
        qualified_only,
    ]
    rows = await db.query_raw(
        f"""SELECT status,data FROM stats_token_reconciliation WHERE {where}""", *args
    )
    counts = {
        k: sum(r["status"] == k for r in rows)
        for k in allowed
        if k and k not in ("unpriced", "band_exceeded")
    }
    counts.update(
        unpriced=sum(bool(r["data"]["unpriced_models"]) for r in rows),
        band_exceeded=sum(bool(r["data"]["band_exceeded"]) for r in rows),
    )
    unpriced = sorted({m for r in rows for m in r["data"]["unpriced_models"]})
    maxima = defaultdict(int)
    for r in rows:
        for key, value in r["data"]["max_input_by_model"].items():
            maxima[key] = max(maxima[key], value)
    predicate = (
        "jsonb_array_length(data->'unpriced_models')>0"
        if status == "unpriced"
        else "jsonb_array_length(data->'band_exceeded')>0"
        if status == "band_exceeded"
        else "status=$6"
        if status
        else "TRUE"
    )
    detail_args = (
        [*args, status]
        if status and status not in ("unpriced", "band_exceeded")
        else args
    )
    n = len(detail_args)
    items = await db.query_raw(
        f"""SELECT trace_id,conversation_id,user_id,local_date::text AS day,scope,status,
      release_qualified,data FROM stats_token_reconciliation WHERE {where} AND {predicate}
      ORDER BY local_date DESC,trace_id LIMIT ${n + 1} OFFSET ${n + 2}""",
        *detail_args,
        page_size,
        (page - 1) * page_size,
    )
    comparable = counts["matched"] + counts["mismatch"] + counts["missing_usage"]
    state = "expired" if end < cutoff else coverage["state"]
    return dict(
        start=str(start),
        end=str(end),
        cohort="all",
        state=state,
        coverage=coverage,
        partial_expiry=start < cutoff <= end,
        valid_from=str(max(start, cutoff)),
        total=len(rows) if not status else counts[status],
        counts=counts,
        comparable_traces=comparable,
        match=defs.ratio(counts["matched"], comparable),
        unmetered_calls=sum(r["data"]["unmetered_calls"] for r in rows),
        unpriced_models=unpriced,
        max_inputs=[
            dict(
                model=k,
                tokens=v,
                first_band=FIRST_BANDS.get(k.split("/", 1)[-1]),
                remaining=None
                if k.split("/", 1)[-1] not in FIRST_BANDS
                else FIRST_BANDS[k.split("/", 1)[-1]] - v,
            )
            for k, v in sorted(maxima.items())
        ],
        items=items,
        page=page,
        page_size=page_size,
        definition_version=defs.DEFINITION_VERSION,
        notes=[
            "同一 Trace 合并聊天和后台用量；无 Trace 的 scope 在成本用途表单列。",
            "未返回用量不推算 token，不混入完全一致比例；失败且未计费的调用不算漏记。",
            "只对账保留期内来源；历史 offline 差异不假定为重复计费。",
        ],
    )
