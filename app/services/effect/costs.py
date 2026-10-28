"""Daily, count-only cost facts. UI reads never scan the usage/message ledgers.

Stored model/TTS amounts remain authoritative. Search is attributed to the
actual calls crossing the shared monthly quota, before cohort/agent filtering.
This is an estimate from configured prices, not an invoice reconciliation.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from hashlib import sha256
import json

from app.config import settings
from app.services.effect import definitions as defs

CATEGORIES = ("model", "search", "tts", "asr")
NUMBERS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "call_count",
    "request_count",
    "unmetered_calls",
    "latency_ms_total",
    "latency_count",
    "failure_count",
    "fallback_count",
    "circuit_open_count",
    "unlinked_rows",
    "units",
    "cost_cny",
    "bytes",
    "duration_ms",
)


def blank() -> dict:
    return {**dict.fromkeys(NUMBERS, 0), "unpriced": False, "models": {}}


def model_key(key: str, prices: dict) -> str | None:
    if key in prices:
        return key
    matches = [name for name in prices if name.split("/", 1)[-1] == key]
    return matches[0] if len(matches) == 1 else None


def accumulate(target: dict, data: dict) -> None:
    for key in NUMBERS:
        target[key] += data.get(key) or 0
    target["unpriced"] |= bool(data.get("unpriced"))
    for model, bucket in data.get("models", {}).items():
        out = target["models"].setdefault(
            model,
            dict.fromkeys(
                ("input", "output", "cached_input", "unmetered_calls", "cost_cny"), 0
            ),
        )
        for field in out:
            out[field] += bucket.get(field) or 0


async def compute(tx, day: date, now: datetime) -> None:
    start, end = defs.local_day_bounds(day)
    pricing = await tx.query_raw(
        "SELECT provider,identifier,input_cost_per_million,output_cost_per_million,cached_input_cost_per_million "
        "FROM model_registry"
    )
    prices = {
        r["provider"] + "/" + r["identifier"]: r
        for r in pricing
        if r["input_cost_per_million"] is not None
        and r["output_cost_per_million"] is not None
    }
    groups: dict[tuple, dict] = defaultdict(blank)
    # Owners marked deleting are excluded. Nullable ownership represents real
    # system work, not a guessed user attribution.
    usage = await tx.query_raw(
        """SELECT l.user_id,l.agent_id,l.scope,l.input_tokens,l.output_tokens,
      l.cost_cny,l.call_count,l.tokens_by_model,l.trace_id,l.latency_ms_total,l.latency_count,
      l.failure_count,l.fallback_count,l.circuit_open_count
      FROM llm_usage l LEFT JOIN users u ON u.id=l.user_id
      WHERE l.created_at >= $1::timestamp AND l.created_at<$2::timestamp
        AND (l.user_id IS NULL OR u.status<>'deleting')""",
        start.isoformat(),
        end.isoformat(),
    )
    for row in usage:
        out = groups[(row["user_id"], row["agent_id"], "model", row["scope"])]
        item = {k: row[k] for k in NUMBERS if k in row}
        models = {k: dict(v) for k, v in (row["tokens_by_model"] or {}).items()}
        # Allocate the persisted row amount in the same proportions as the old
        # cost card; never count a multi-model row's entire fee for every model.
        weights = {}
        for model, bucket in models.items():
            price = prices.get(model_key(model, prices))
            if price is not None:
                cached = min(
                    int(bucket.get("input") or 0),
                    max(0, int(bucket.get("cached_input") or 0)),
                )
                cached_price = price["cached_input_cost_per_million"]
                weights[model] = (
                    (int(bucket.get("input") or 0) - cached)
                    * float(price["input_cost_per_million"])
                    + cached
                    * float(
                        cached_price
                        if cached_price is not None
                        else price["input_cost_per_million"]
                    )
                    + int(bucket.get("output") or 0)
                    * float(price["output_cost_per_million"])
                )
        weight = sum(weights.values())
        for model, bucket in models.items():
            bucket["cost_cny"] = (
                float(row["cost_cny"] or 0) * weights.get(model, 0) / weight
                if weight
                else 0
            )

        unpriced = any(model_key(m, prices) is None for m in models)
        # Nonempty billed input with no breakdown cannot be certified as priced.
        unpriced |= not models and (row["input_tokens"] > 0 or row["output_tokens"] > 0)
        item.update(
            request_count=1,
            models=models,
            unpriced=unpriced,
            unmetered_calls=sum(
                int(b.get("unmetered_calls") or 0) for b in models.values()
            ),
            cached_input_tokens=sum(
                int(b.get("cached_input") or 0) for b in models.values()
            ),
            unlinked_rows=int(not row["trace_id"]),
        )
        accumulate(out, item)
    month_start = defs.local_day_bounds(day.replace(day=1))[0]
    # Allocate quota in a deterministic ledger order, including internal/system
    # calls. Applying filters before this step would grant the quota repeatedly.
    search = await tx.query_raw(
        """WITH ordered AS (
      SELECT l.user_id,l.agent_id,l.scope,l.created_at,l.web_search_calls,
        SUM(l.web_search_calls) OVER(ORDER BY l.created_at,l.id) AS cumulative
      FROM llm_usage l WHERE l.web_search_calls>0 AND l.created_at>=$1::timestamp
        AND l.created_at<$3::timestamp)
      SELECT o.*,GREATEST(0,cumulative-$4::bigint)-GREATEST(0,cumulative-web_search_calls-$4::bigint) AS billed
      FROM ordered o LEFT JOIN users u ON u.id=o.user_id
      WHERE o.created_at>=$2::timestamp AND (o.user_id IS NULL OR u.status<>'deleting')""",
        month_start.isoformat(),
        start.isoformat(),
        end.isoformat(),
        max(0, int(settings.web_search_free_calls_monthly)),
    )
    for row in search:
        price = float(settings.web_search_price_cny_per_k)
        out = groups[(row["user_id"], row["agent_id"], "search", row["scope"])]
        accumulate(
            out,
            dict(
                units=row["web_search_calls"],
                cost_cny=float(row["billed"]) * price / 1000,
                unpriced=price <= 0,
                request_count=1,
            ),
        )
    tts = await tx.query_raw(
        """SELECT t.user_id,t.agent_id,COUNT(*)::int AS request_count,
      SUM(t.duration_milliseconds)::float8 AS duration_ms,SUM(t.audio_bytes)::bigint AS bytes,
      SUM(t.billable_characters)::bigint AS units,SUM(t.cost_cny)::float8 AS cost_cny,
      bool_or(t.unit_price_cny<=0) AS unpriced FROM tts_usage t JOIN users u ON u.id=t.user_id
      WHERE t.created_at>=$1::timestamp AND t.created_at<$2::timestamp AND u.status<>'deleting'
      GROUP BY t.user_id,t.agent_id""",
        start.isoformat(),
        end.isoformat(),
    )
    for row in tts:
        out = groups[(row["user_id"], row["agent_id"], "tts", "speech_output")]
        out.update(row)
        out.pop("user_id")
        out.pop("agent_id")
    asr = await tx.query_raw(
        """SELECT s.user_id,c.agent_id,s.display_mode,COUNT(*)::int AS request_count,
      SUM(s.duration_seconds)::bigint AS units FROM speech_usage s
      JOIN users u ON u.id=s.user_id LEFT JOIN conversations c ON c.id=s.conversation_id
      WHERE s.created_at>=$1::timestamp AND s.created_at<$2::timestamp AND u.status<>'deleting'
      GROUP BY s.user_id,c.agent_id,s.display_mode""",
        start.isoformat(),
        end.isoformat(),
    )
    for row in asr:
        price = float(settings.asr_price_cny_per_second)
        accumulate(
            groups[(row["user_id"], row["agent_id"], "asr", row["display_mode"])],
            dict(
                request_count=row["request_count"],
                units=row["units"],
                cost_cny=row["units"] * price,
                unpriced=price <= 0,
            ),
        )
    # The original media cards count uploads as well as billable transcriptions.
    media = await tx.query_raw(
        """SELECT a.user_id,c.agent_id,a.kind,COUNT(*)::int AS n,
      COALESCE(SUM(a.duration_seconds),0)::float8 AS seconds,COALESCE(SUM(a.size),0)::bigint AS bytes
      FROM chat_message_attachments a JOIN messages m ON m.id=a.message_id
      JOIN conversations c ON c.id=m.conversation_id JOIN users u ON u.id=a.user_id
      WHERE a.created_at>=$1::timestamp AND a.created_at<$2::timestamp AND m.role='user'
        AND a.kind IN ('audio','image') AND u.status<>'deleting'
      GROUP BY a.user_id,c.agent_id,a.kind""",
        start.isoformat(),
        end.isoformat(),
    )
    for row in media:
        groups[(row["user_id"], row["agent_id"], "media", row["kind"])].update(
            request_count=row["n"], units=row["seconds"], bytes=row["bytes"]
        )
    records = [
        dict(
            id=sha256(
                json.dumps([str(day), *key], ensure_ascii=False).encode()
            ).hexdigest(),
            user_id=key[0],
            agent_id=key[1],
            category=key[2],
            scope=key[3],
            data=data,
        )
        for key, data in groups.items()
    ]
    await tx.execute_raw(
        "DELETE FROM stats_cost_daily WHERE local_date=$1::date", str(day)
    )
    if records:
        await tx.execute_raw(
            """INSERT INTO stats_cost_daily
          (id,local_date,user_id,agent_id,category,scope,data,definition_version,computed_at)
          SELECT r.id,$1::date,r.user_id,r.agent_id,r.category,r.scope,r.data,$2,$3::timestamptz
          FROM jsonb_to_recordset($4::jsonb) r(id text,user_id text,agent_id text,category text,scope text,data jsonb)""",
            str(day),
            defs.DEFINITION_VERSION,
            now.isoformat(),
            json.dumps(records),
        )


async def read_costs(
    start: date,
    end: date,
    *,
    cohort: str,
    agent: str = "",
    page: int = 1,
    page_size: int = 10,
) -> dict:
    from app.db import db
    from app.services.effect.dashboard import fact_coverage

    if len(agent) > 128:
        raise ValueError("Agent 筛选过长")
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError("分页超出范围")
    coverage = await fact_coverage(start, end)
    rows = await db.query_raw(
        f"""SELECT f.local_date::text AS day,f.user_id,f.agent_id,f.category,f.scope,f.data,
       COALESCE(u.display_name,u.username,'系统任务') AS username,COALESCE(a.name,'未关联 Agent') AS agent_name
       FROM stats_cost_daily f LEFT JOIN users u ON u.id=f.user_id LEFT JOIN ai_agents a ON a.id=f.agent_id
       WHERE f.local_date BETWEEN $1::date AND $2::date AND f.definition_version=$3
         AND ($4='' OR f.agent_id=$4) AND (f.user_id IS NULL OR u.status<>'deleting')
         AND {defs.population_sql("f.user_id", cohort)}""",
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
        agent,
    )
    categories = {c: blank() for c in CATEGORIES}
    evaluation = blank()
    daily = defaultdict(blank)
    scopes = defaultdict(blank)
    users = defaultdict(blank)
    media = defaultdict(blank)
    identities = {}
    media_users = defaultdict(lambda: defaultdict(blank))
    for row in rows:
        data = row["data"]
        category = row["category"]
        scope = row["scope"]
        if row["user_id"] and category in ("media", "asr", "tts"):
            accumulate(
                media_users[row["user_id"]][scope if category != "tts" else "tts"], data
            )
            identities[row["user_id"]] = row["username"]
        if category == "media":
            accumulate(media[scope], data)
            continue
        if scope == "quality_eval":
            accumulate(evaluation, data)
            continue
        accumulate(categories[category], data)
        accumulate(daily[row["day"]], data)
        if category == "model":
            accumulate(scopes[scope], data)
            if row["user_id"]:
                accumulate(users[row["user_id"]], data)
                identities[row["user_id"]] = row["username"]

    def public(data):
        return {
            **data,
            "state": coverage["state"]
            if coverage["state"] != "ok"
            else "unpriced"
            if data["unpriced"]
            else "ok",
            "cost_cny": None
            if data["unpriced"] or coverage["state"] != "ok"
            else round(data["cost_cny"], 6),
        }

    # Empty activity does not prove that a category has a configured price.
    categories["search"]["unpriced"] |= float(settings.web_search_price_cny_per_k) <= 0
    categories["asr"]["unpriced"] |= float(settings.asr_price_cny_per_second) <= 0
    if not categories["tts"]["request_count"]:
        categories["tts"]["unpriced"] |= (
            float(settings.tts_price_cny_per_10k_chars) <= 0
        )
    for offset in range((end - start).days + 1):
        daily[str(start + timedelta(days=offset))]
    total = blank()
    for data in categories.values():
        accumulate(total, data)
    return {
        "start": str(start),
        "end": str(end),
        "cohort": cohort,
        "agent": agent,
        "coverage": coverage,
        "definition_version": defs.DEFINITION_VERSION,
        "categories": {c: public(d) for c, d in categories.items()},
        "media_users": [
            {
                "id": user,
                "username": identities[user],
                "usage": {k: public(v) for k, v in buckets.items()},
            }
            for user, buckets in sorted(media_users.items())[
                page_size * (page - 1) : page_size * page
            ]
        ],
        "media_user_total": len(media_users),
        "page": page,
        "page_size": page_size,
        "total": public(total),
        "evaluation": public(evaluation),
        "media": {k: public(v) for k, v in media.items()},
        "daily": [{"date": day, **public(v)} for day, v in sorted(daily.items())],
        "scopes": [{"scope": k, **public(v)} for k, v in sorted(scopes.items())],
        "top_users": [
            {"id": k, "username": identities[k], **public(v)}
            for k, v in sorted(
                users.items(), key=lambda item: (-item[1]["cost_cny"], item[0])
            )[:10]
        ],
        "notes": [
            "模型与 TTS 使用入账金额；缺定价不显示成免费。",
            "搜索按全账户本地自然月先扣免费额度，按调用时间归属到本窗口；属于配置价估算。",
            "quality_eval 单列，不计入用户成本；无 user_id 的系统任务仍计入全站费用。",
        ],
    }
