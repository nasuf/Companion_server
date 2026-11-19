"""Administrator-only quality workspace. Lists are bounded and omit dialogue text."""

from datetime import UTC, date, datetime, timedelta
from math import sqrt

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
async def results(
    job_id: str,
    after: str = "",
    limit: int = Query(20, ge=1, le=50),
    suite: str | None = Query(None, max_length=64),
    critical: bool | None = None,
    failed_only: bool = False,
):
    # failed_only also returns unjudged rows: neither counts as a pass.
    rows = await db.query_raw(
        """SELECT case_id,repetition,result FROM quality_results
      WHERE job_id=$1 AND case_id||':'||repetition::text>$2
        AND ($4::text IS NULL OR result->>'suite'=$4)
        AND ($5::boolean IS NULL OR COALESCE((result->>'critical')::boolean,false)=$5)
        AND (NOT $6::boolean OR (result->>'pass') IS DISTINCT FROM 'true')
      ORDER BY case_id||':'||repetition::text LIMIT $3""",
        job_id,
        after,
        limit + 1,
        suite,
        critical,
        failed_only,
    )
    return dict(
        items=rows[:limit],
        next_cursor=f"{rows[limit - 1]['case_id']}:{rows[limit - 1]['repetition']}"
        if len(rows) > limit
        else None,
    )


REGRESSION_KINDS = ("prompt", "nightly", "full")
# Below this many judged samples an interval is too wide to read as a rate.
SMALL_SAMPLE = 30
_GROUPS_SQL = """SELECT result->>'suite' AS suite,
  COALESCE((result->>'critical')::boolean,false) AS critical,
  count(*)::int AS samples, count(DISTINCT case_id)::int AS cases,
  count(*) FILTER(WHERE result->>'pass'='true')::int AS passed,
  count(*) FILTER(WHERE result->>'pass'='false')::int AS failed,
  count(*) FILTER(WHERE (result->>'pass') IS NULL)::int AS unjudged
  FROM quality_results WHERE job_id=$1 GROUP BY 1,2 ORDER BY 1,2"""


def wilson(passed, judged, z=1.96):
    if not judged:
        return None
    p = passed / judged
    centre = (p + z * z / (2 * judged)) / (1 + z * z / judged)
    half = z * sqrt(p * (1 - p) / judged + z * z / (4 * judged * judged)) / (1 + z * z / judged)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _group(row, expected):
    judged = row["passed"] + row["failed"]
    key = f"{row['suite']}|{'critical' if row['critical'] else 'other'}"
    planned = expected.get(key) if isinstance(expected, dict) else None
    return dict(
        row,
        key=key,
        origin="production_confirmed" if row["suite"] == "human_confirmed" else "synthetic",
        judged=judged,
        pass_rate=round(row["passed"] / judged, 4) if judged else None,
        ci95=wilson(row["passed"], judged),
        small_sample=judged < SMALL_SAMPLE,
        expected=planned if isinstance(planned, int) else None,
        not_run=max(planned - row["samples"], 0) if isinstance(planned, int) else None,
    )


def _versions(job):
    snapshot = job["snapshot"] if isinstance(job["snapshot"], dict) else {}
    report = job["report"] if isinstance(job["report"], dict) else {}
    prompts = snapshot.get("prompts")
    route = snapshot.get("route")
    # Digests and counts only: prompt bodies and prices stay server-side.
    return dict(
        # A prompt job's source_version is the saved prompt version, not a commit.
        code=job["source_version"] if job["kind"] != "prompt" else None,
        prompt_key=job["source_key"] if job["kind"] == "prompt" else None,
        prompt_version=job["source_version"] if job["kind"] == "prompt" else None,
        prompt_snapshot=snapshot.get("hash"),
        prompt_count=len(prompts) if isinstance(prompts, (list, dict)) else None,
        route=[str(part) for part in route][:2] if isinstance(route, list) else None,
        corpus_version=report.get("corpus_version"),
        corpus_hash=report.get("corpus_hash"),
        repeats=report.get("repeats"),
    )


def report_state(job):
    status = job["status"]
    if status == "running":
        return "stale" if job["lease_expired"] else "running"
    if status == "pending":
        return "not_run"
    if status == "completed":
        return "completed"
    return status if status in {"skipped", "not_configured"} else "failed"


async def _regression_job(job_id):
    rows = await db.query_raw(
        """SELECT id,kind,status,source_key,source_version,local_date,snapshot,report,created_at,updated_at,
        (lease_until IS NULL OR lease_until<NOW()) AS lease_expired
      FROM quality_jobs WHERE id=$1 AND kind=ANY($2::text[])""",
        job_id,
        list(REGRESSION_KINDS),
    )
    return rows[0] if rows else None


@router.get("/jobs/{job_id}/report")
async def regression_report(job_id: str):
    job = await _regression_job(job_id)
    if job is None:
        raise HTTPException(404, "regression_report_unavailable")
    now = datetime.now(UTC)
    report = job["report"] if isinstance(job["report"], dict) else {}
    expected = report.get("expected_by_group")
    groups = [_group(r, expected) for r in await db.query_raw(_GROUPS_SQL, job_id)]
    seen = {g["key"] for g in groups}
    if isinstance(expected, dict):
        # Planned groups with no stored sample at all: never ran.
        for key, planned in sorted(expected.items()):
            if key in seen or not isinstance(planned, int) or "|" not in key:
                continue
            suite, kind = key.split("|", 1)
            groups.append(_group(
                dict(suite=suite, critical=kind == "critical", samples=0, cases=0,
                     passed=0, failed=0, unjudged=0),
                expected,
            ))
    baseline = None
    baseline_id = report.get("baseline_job_id")
    if isinstance(baseline_id, str) and baseline_id != job_id:
        base_job = await _regression_job(baseline_id)
        if base_job is not None:
            base_report = base_job["report"] if isinstance(base_job["report"], dict) else {}
            baseline = dict(
                id=base_job["id"],
                kind=base_job["kind"],
                created_at=base_job["created_at"],
                versions=_versions(base_job),
                corpus_matches=base_report.get("corpus_hash") == report.get("corpus_hash"),
                groups={
                    g["key"]: dict(passed=g["passed"], failed=g["failed"], judged=g["judged"],
                                   pass_rate=g["pass_rate"], ci95=g["ci95"])
                    for g in (_group(r, None) for r in await db.query_raw(_GROUPS_SQL, baseline_id))
                },
            )
    evaluated, planned = report.get("evaluated"), report.get("expected")
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else None
    return dict(
        sampled_at=now.isoformat(),
        job=dict(id=job["id"], kind=job["kind"], status=job["status"], local_date=job["local_date"],
                 created_at=job["created_at"], updated_at=job["updated_at"]),
        state=report_state(job),
        versions=_versions(job),
        gate=gate,
        thresholds=dict(
            noise_rate=gate["noise_rate"]
            if gate and isinstance(gate.get("noise_rate"), (int, float))
            else settings.quality_noise_rate,
            noise_min_cases=settings.quality_noise_min_cases,
            # Reports before the majority gate carry no critical_regressed_cases.
            critical_rule=(
                f"关键用例按 {settings.quality_critical_repeats} 次重复多数票比较，"
                f"由通过变为不通过的用例超过 {gate['critical_noise_cases']} 个即判失败"
                if gate and isinstance(gate.get("critical_noise_cases"), int)
                else "关键用例失败数比基线多 1 条即判失败"
            ),
            small_sample=SMALL_SAMPLE,
            warning_only=report.get("warning_only"),
        ),
        totals=dict(
            expected=planned if isinstance(planned, int) else None,
            evaluated=evaluated if isinstance(evaluated, int) else sum(g["samples"] for g in groups),
            not_run=max(planned - evaluated, 0) if isinstance(planned, int) and isinstance(evaluated, int) else None,
            unjudged=sum(g["unjudged"] for g in groups),
            errored=len(report["errored_scenarios"])
            if isinstance(report.get("errored_scenarios"), list) else 0,
            expected_by_group="recorded" if isinstance(expected, dict) else "not_collected",
        ),
        error=dict(type=report.get("error_type"), case_id=report.get("error_case_id"))
        if report.get("error_type") else None,
        groups=groups,
        baseline=baseline,
    )


@router.get("/routing-history")
async def routing_history():
    from app.services.quality.public_results import summaries

    return dict(
        sampled_at=datetime.now(UTC).isoformat(),
        enabled=settings.quality_enabled,
        items=(await summaries("routing", "global_model_route"))[:20],
    )
