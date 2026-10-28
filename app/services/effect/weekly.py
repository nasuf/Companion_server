"""Count-only weekly review and release annotations; no model calls or chat text."""

from datetime import date, timedelta
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode

from app.services.effect import definitions as defs

OVERVIEW = (
    "chat_active_users",
    "next_day_return",
    "turns_per_active_user",
    "proactive_response_3h",
    "feedback_per_100_turns",
    "reply_first_bubble_ms_p90",
    "no_reply_10m",
    "cost_per_active_user",
)
WEIGHTS = {
    "no_reply": 5,
    "reply_failed": 5,
    "resend_60s": 2,
    "meta_comment": 2,
    "slow_first_bubble": 1,
}


@lru_cache(maxsize=1)
def release_markers() -> list[dict]:
    return json.loads(Path(__file__).with_name("release_markers.json").read_text())


def markers(start: date, end: date) -> list[dict]:
    return [row for row in release_markers() if str(start) <= row["date"] <= str(end)]


async def worst(start: date, end: date, cohort: str) -> list[dict]:
    from app.db import db

    rows = await db.query_raw(
        f"""WITH counts AS (
      SELECT r.user_id,r.conversation_id,r.kind,COUNT(*)::int AS n,
        (array_agg(r.trace_id ORDER BY r.occurred_at DESC) FILTER(WHERE r.trace_id IS NOT NULL))[1] AS trace_id
      FROM stats_review_items r JOIN users u ON u.id=r.user_id
      JOIN conversations c ON c.id=r.conversation_id AND c.user_id=r.user_id
      WHERE r.local_date BETWEEN $1::date AND $2::date AND r.definition_version=$3
        AND u.status<>'deleting' AND {defs.user_row_population_sql("u", cohort)} AND r.kind=ANY($4::text[])
      GROUP BY r.user_id,r.conversation_id,r.kind)
      SELECT user_id,conversation_id,jsonb_object_agg(kind,n) AS counts,
        SUM(n*CASE WHEN kind IN ('no_reply','reply_failed') THEN 5 WHEN kind IN ('resend_60s','meta_comment') THEN 2 ELSE 1 END)::int AS score,
        MAX(trace_id) AS trace_id FROM counts GROUP BY user_id,conversation_id
      ORDER BY score DESC,conversation_id LIMIT 5""",
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
        list(WEIGHTS),
    )
    for row in rows:
        row["trace_link"] = "/admin?" + urlencode(
            dict(
                section="users",
                user=row["user_id"],
                conversation=row["conversation_id"],
                **({"trace": row["trace_id"]} if row["trace_id"] else {}),
            )
        )
    return rows


async def report(end: date, cohort: str) -> dict:
    from app.services.effect import daily_facts, dashboard

    start = end - timedelta(days=6)
    current = await daily_facts.read_series(
        start, end, cohort=cohort, metrics=list(OVERVIEW)
    )
    previous = await daily_facts.read_series(
        start - timedelta(days=7),
        start - timedelta(days=1),
        cohort=cohort,
        metrics=list(OVERVIEW),
    )
    current["summary"] = await dashboard.summarize(
        start, end, cohort, current["series"]
    )
    previous["summary"] = await dashboard.summarize(
        start - timedelta(days=7), start - timedelta(days=1), cohort, previous["series"]
    )
    feedback = await daily_facts.read_series(
        start, end, cohort=cohort, metrics=["feedback_count"]
    )
    labels = await daily_facts.read_series(
        start, end, cohort=cohort, metrics=["admin_bug_labels"], dimension="label"
    )
    coverage = await dashboard.fact_coverage(start, end)
    data = dict(
        start=str(start),
        end=str(end),
        cohort=cohort,
        coverage=coverage,
        current=current,
        previous=previous,
        feedback=feedback,
        labels=labels,
        worst=await worst(start, end, cohort) if coverage["state"] == "ok" else [],
        weights=WEIGHTS,
        release_markers=markers(start, end),
        definition_version=defs.DEFINITION_VERSION,
        definitions=defs.describe(OVERVIEW),
        notes=[
            "排序只用于人工复核优先级，不代表已验证的用户不满。",
            "历史日期基于提交或台账记录时会明确标注，不假装是准确切流时间。",
        ],
    )
    data["markdown"] = to_markdown(data)
    return data


def to_markdown(data: dict) -> str:
    def cell(item):
        if item.get("value") is None:
            return (
                f"{item['state']} ({item.get('numerator')}/{item.get('denominator')})"
            )
        return f"{item['value']:.4g} [{item['state']}]"

    lines = [
        f"# 统计周报 {data['start']}—{data['end']}",
        f"人群：{data['cohort']}；北京时间；口径：{data['definition_version']}",
        f"更新时间：{data['coverage'].get('computed_at') or '未运行'}",
        "",
        "| 指标 | 本周 | 上周 |",
        "|---|---:|---:|",
    ]
    for metric in OVERVIEW:
        lines.append(
            f"| {defs.METRICS[metric]['label']} | {cell(data['current']['summary'][metric])} | {cell(data['previous']['summary'][metric])} |"
        )
    lines += ["", "## 待人工复核的会话（不是负面评价）"]
    lines.extend(
        f"- [{r['conversation_id']}]({r['trace_link']}): 分值 {r['score']}；计数 {json.dumps(r['counts'], ensure_ascii=False)}"
        for r in data["worst"]
    )
    lines += [
        "",
        "## 意见反馈与管理员标注",
        "意见反馈来自 App 反馈表；管理员标注单列，不能互换。",
    ]
    for key, series in [
        ("意见反馈", data["feedback"]["series"]),
        ("管理员标注", data["labels"]["series"]),
    ]:
        points = next(iter(series.values()))
        if any(p["state"] in ("not_computed", "stale", "failed") for p in points):
            lines.append(f"- {key}：未完成采集或重算")
        else:
            grouped = {}
            for point in points:
                label = point.get("dimension_value") or "总计"
                grouped[label] = grouped.get(label, 0) + (point.get("value") or 0)
            lines.extend(
                f"- {key} / {label}: {count:g}"
                for label, count in sorted(grouped.items(), key=lambda x: -x[1])[:10]
            )
    return "\n".join(lines) + "\n"


async def enqueue(start: date, end: date, cohort: str, admin_id: str) -> dict:
    from app.db import db
    from app.services.effect.dashboard import fact_coverage

    if (await fact_coverage(start, end))["state"] != "ok":
        raise ValueError("统计尚未完整，暂不能提交复核")
    entries = await worst(start, end, cohort)
    added = 0
    for row in entries:
        key = sha256(
            f"weekly:{start}:{end}:{row['conversation_id']}:{defs.DEFINITION_VERSION}".encode()
        ).hexdigest()
        added += await db.execute_raw(
            """INSERT INTO stats_review_queue
          (id,user_id,conversation_id,start_date,end_date,source,counts,definition_version,created_by)
          SELECT $1,c.user_id,c.id,$3::date,$4::date,'weekly',$5::jsonb,$6,$7
          FROM conversations c JOIN users u ON u.id=c.user_id
          WHERE c.id=$2 AND u.status<>'deleting' ON CONFLICT(id) DO NOTHING""",
            key,
            row["conversation_id"],
            str(start),
            str(end),
            json.dumps(row["counts"]),
            defs.DEFINITION_VERSION,
            admin_id,
        )
    return dict(
        selected=len(entries),
        added=added,
        source="weekly",
        status="pending",
        note="已登记到 B1 人工复核待办；不会自动调用模型或修改聊天内容。",
    )
