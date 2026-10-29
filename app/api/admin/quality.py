"""Administrator-only quality workspace. Lists are bounded and omit dialogue text."""

from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal

from app.api.jwt_auth import require_admin_jwt
from app.config import settings
from app.db import db
from app.services.quality import daily, review, sampling
from app.services.quality.policy import local_day, public_policy

router = APIRouter(
    prefix="/admin-api/quality",
    tags=["admin-quality"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("")
async def overview(day: date | None = None):
    day = day or local_day(datetime.now(UTC))
    if (
        not local_day(datetime.now(UTC)) - timedelta(days=90)
        <= day
        <= local_day(datetime.now(UTC))
    ):
        raise HTTPException(400, "quality_date_out_of_range")
    jobs = await db.query_raw(
        """SELECT j.id,j.kind,j.source_key,j.source_version,j.local_date,j.status,j.report,j.created_at,j.updated_at,
        COALESCE(c.settled_cny,0)::float8 AS settled_cny,
        COALESCE(c.reserved_cny,0)::float8 AS reserved_or_unknown_cny
      FROM (SELECT * FROM quality_jobs WHERE local_date=$1::date
        ORDER BY created_at DESC,id DESC LIMIT 101) j
      LEFT JOIN LATERAL (SELECT sum(cost_cny) AS settled_cny,
        sum(reserved_cny) FILTER(WHERE cost_cny IS NULL) AS reserved_cny
        FROM quality_charges WHERE job_id=j.id) c ON true
      ORDER BY j.created_at DESC,j.id DESC""",
        str(day),
    )
    money = await db.query_raw(
        """SELECT COALESCE(sum(cost_cny),0)::float8 AS settled_cny,
      COALESCE(sum(reserved_cny) FILTER(WHERE cost_cny IS NULL),0)::float8 AS reserved_or_unknown_cny,
      count(*) FILTER(WHERE status='unknown')::int AS unknown_calls
      FROM quality_charges WHERE month=$1::date""",
        str(day.replace(day=1)),
    )
    for job in jobs[:100]:
        if job["kind"] == "daily":
            job["report"] = {**job["report"], **await daily.counts(job["id"])}
    return dict(
        sampled_at=datetime.now(UTC).isoformat(),
        day=str(day),
        policy=public_policy(),
        enabled=settings.quality_enabled,
        jobs=jobs[:100],
        truncated=len(jobs) > 100,
        budget=money[0],
    )


@router.get("/jobs/{job_id}/samples")
async def samples(
    job_id: str,
    queue_only: bool = True,
    after: str = "",
    limit: int = Query(20, ge=1, le=50),
):
    rows = await db.query_raw(
        """SELECT s.id,s.message_id,s.reply_id,s.trace_id,s.status,s.internal,
      s.verdict,s.review_queue,s.review_status,s.review_category,s.review_note,s.revision,s.reviewed_at,
      reviewer.username AS reviewer_name FROM quality_samples s
      JOIN users owner ON owner.id=s.user_id AND owner.status<>'deleting'
      LEFT JOIN users reviewer ON reviewer.id=s.reviewed_by
      WHERE s.job_id=$1 AND (NOT $2::boolean OR (s.review_queue AND s.review_status='unreviewed')) AND s.id>$3
      ORDER BY s.id LIMIT $4""",
        job_id,
        queue_only,
        after,
        limit + 1,
    )
    return dict(
        sampled_at=datetime.now(UTC).isoformat(),
        items=rows[:limit],
        next_cursor=rows[limit - 1]["id"] if len(rows) > limit else None,
    )


@router.get("/samples/{sample_id}")
async def detail(sample_id: str):
    result = await sampling.context(sample_id)
    if result is None:
        raise HTTPException(404, "sample_unavailable")
    return result


class HumanReview(BaseModel):
    revision: int = Field(ge=0)
    decision: Literal["confirmed", "false_positive", "uncertain"]
    category: (
        Literal[
            "hallucination", "irrelevance", "temporal", "persona_tone", "repetition"
        ]
        | None
    ) = None
    note: str = Field(default="", max_length=1000)


@router.put("/samples/{sample_id}/review")
async def save_review(
    sample_id: str, body: HumanReview, admin=Depends(require_admin_jwt)
):
    try:
        return await review.save(
            sample_id,
            admin["sub"],
            body.revision,
            body.decision,
            body.category,
            body.note,
        )
    except review.ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/jobs/{job_id}/results")
async def results(job_id: str, after: str = "", limit: int = Query(20, ge=1, le=50)):
    rows = await db.query_raw(
        """SELECT case_id,repetition,result FROM quality_results
      WHERE job_id=$1 AND case_id||':'||repetition::text>$2
      ORDER BY case_id||':'||repetition::text LIMIT $3""",
        job_id,
        after,
        limit + 1,
    )
    return dict(
        items=rows[:limit],
        next_cursor=f"{rows[limit - 1]['case_id']}:{rows[limit - 1]['repetition']}"
        if len(rows) > limit
        else None,
    )


@router.get("/routing-history")
async def routing_history():
    from app.services.quality.public_results import summaries

    return dict(
        sampled_at=datetime.now(UTC).isoformat(),
        enabled=settings.quality_enabled,
        items=(await summaries("routing", "global_model_route"))[:20],
    )
