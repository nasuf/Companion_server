"""Cluster-wide conservative reservations. No provider call precedes reserve().

Lock order is executor -> month -> job. Settlement only takes month -> charge.
Failed/ambiguous requests keep their reservation;
settlement and llm_usage are one transaction and idempotent by charge ID.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4
import json

from app.config import settings
from app.db import db
from app.services.quality.policy import local_day, money, run_limit


class BudgetExhausted(RuntimeError):
    pass


async def reserve(
    job_id: str, model: str, ceiling, *, lease_token: str, now=None, request=None
) -> str:
    now = now or datetime.now(UTC)
    day = local_day(now)
    month = day.replace(day=1)
    ceiling = money(ceiling)
    if ceiling <= 0:
        raise ValueError("positive_priced_reservation_required")
    charge_id = uuid4().hex
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await tx.execute_raw("SET LOCAL lock_timeout='3s'")
        from app.services.quality.jobs import require_lease

        await require_lease(tx, job_id, lease_token)
        await tx.execute_raw(
            "INSERT INTO quality_budget_months(month) VALUES($1::date) ON CONFLICT DO NOTHING",
            str(month),
        )
        row = (
            await tx.query_raw(
                "SELECT committed_cny FROM quality_budget_months WHERE month=$1::date FOR UPDATE",
                str(month),
            )
        )[0]
        jobs = await tx.query_raw(
            "SELECT kind,status FROM quality_jobs WHERE id=$1 FOR UPDATE", job_id
        )
        if not jobs or jobs[0]["status"] != "running":
            raise ValueError("evaluation_job_not_running")
        kind = jobs[0]["kind"]
        spent = (
            await tx.query_raw(
                """SELECT
          COALESCE(SUM(COALESCE(c.cost_cny,c.reserved_cny)) FILTER(WHERE c.job_id=$1),0) AS run,
          COALESCE(SUM(COALESCE(c.cost_cny,c.reserved_cny)) FILTER(WHERE c.local_date=$2::date AND j.kind IN ('daily','calibration')),0) AS daily,
          COALESCE(SUM(COALESCE(c.cost_cny,c.reserved_cny)) FILTER(WHERE c.local_date=$2::date AND j.kind='prompt'),0) AS prompts
          FROM quality_charges c JOIN quality_jobs j ON j.id=c.job_id
          WHERE c.month=$3::date OR c.job_id=$1""",
                job_id,
                str(day),
                str(month),
            )
        )[0]
        limits = [
            (
                money(row["committed_cny"]),
                money(settings.quality_monthly_cny),
                "monthly",
            ),
            (money(spent["run"]), run_limit(kind), "run"),
        ]
        if kind in ("daily", "calibration"):
            limits.append(
                (money(spent["daily"]), money(settings.quality_daily_cny), "daily")
            )
        if kind == "prompt":
            limits.append(
                (
                    money(spent["prompts"]),
                    money(settings.quality_prompt_daily_cny),
                    "prompt_daily",
                )
            )
        for used, limit, reason in limits:
            if used + ceiling > limit:
                raise BudgetExhausted(reason)
        await tx.execute_raw(
            "UPDATE quality_budget_months SET committed_cny=committed_cny+$2::numeric WHERE month=$1::date",
            str(month),
            str(ceiling),
        )
        await tx.execute_raw(
            """INSERT INTO quality_charges(id,job_id,month,local_date,model,reserved_cny,request_policy)
          VALUES($1,$2,$3::date,$4::date,$5,$6::numeric,$7::jsonb)""",
            charge_id,
            job_id,
            str(month),
            str(day),
            model,
            str(ceiling),
            json.dumps(
                {
                    k: v
                    for k, v in (request or {}).items()
                    if k in {"operation", "max_tokens"}
                }
            ),
        )
    return charge_id


async def settle(charge_id: str, cost, usage: dict) -> None:
    cost = money(cost)
    # Read only the immutable month before taking locks in reserve() order.
    rows = await db.query_raw(
        "SELECT month FROM quality_charges WHERE id=$1", charge_id
    )
    if not rows:
        raise ValueError("charge_not_found")
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await tx.query_raw(
            "SELECT month FROM quality_budget_months WHERE month=$1::date FOR UPDATE",
            str(rows[0]["month"])[:10],
        )
        charge = (
            await tx.query_raw(
                "SELECT * FROM quality_charges WHERE id=$1 FOR UPDATE", charge_id
            )
        )[0]
        if charge["status"] == "settled":
            if money(charge["cost_cny"]) != cost or charge["usage"] != usage:
                raise ValueError("settlement_conflict")
            return
        if cost < 0 or cost > money(charge["reserved_cny"]):
            raise ValueError("provider_exceeded_reserved_ceiling")
        await tx.execute_raw(
            "UPDATE quality_budget_months SET committed_cny=committed_cny-$2::numeric WHERE month=$1::date",
            str(charge["month"])[:10],
            str(money(charge["reserved_cny"]) - cost),
        )
        await tx.execute_raw(
            "UPDATE quality_charges SET cost_cny=$2::numeric,usage=$3::jsonb,status='settled' WHERE id=$1",
            charge_id,
            str(cost),
            json.dumps(usage),
        )
        bucket = {
            charge["model"]: {
                "input": usage["input"],
                "output": usage["output"],
                "cached_input": usage.get("cached_input", 0),
            }
        }
        await tx.execute_raw(
            """INSERT INTO llm_usage(id,scope,input_tokens,output_tokens,tokens_by_model,cost_cny,call_count,created_at)
          VALUES($1,'quality_eval',$2,$3,$4::jsonb,$5::double precision,1,CURRENT_TIMESTAMP)""",
            "quality-" + charge_id,
            usage["input"],
            usage["output"],
            json.dumps(bucket),
            float(cost),
        )


async def uncertain(charge_id: str):
    await db.execute_raw(
        "UPDATE quality_charges SET status='unknown' WHERE id=$1 AND status='reserved'",
        charge_id,
    )
