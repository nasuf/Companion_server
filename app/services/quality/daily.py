"""One judgement per sampled turn; no writes to conversation or memory tables."""

from datetime import UTC, datetime, timedelta
import json

from app.config import settings
from app.db import db
from app.services.quality import budget, judge, jobs, sampling, transport


def bounded_context(value: dict, limit=24000) -> dict:
    result = dict(value)
    # Replay input duplicates prompt_evidence and is not extra judge evidence.
    result.pop("recorded_messages", None)
    result["history"] = list(value["history"])
    result["prompt_evidence"] = list(value["prompt_evidence"])

    def size():
        return len(json.dumps(result, ensure_ascii=False).encode())

    # Preserve complete newest turns; do not fabricate a mid-message boundary.
    while size() > limit and result["prompt_evidence"]:
        result["prompt_evidence"].pop()
        result["truncated"] = True
    while size() > limit and len(result["history"]) > 1:
        result["history"].pop(0)
        result["truncated"] = True
    if not result["prompt_evidence"]:
        result["evidence_status"] = "not_collected"
    if size() > limit:
        raise ValueError("evaluation_context_exceeds_bound")
    return result


async def counts(job_id: str) -> dict:
    row = (
        await db.query_raw(
            """SELECT count(*)::int AS selected_count,
      count(*) FILTER(WHERE status='judged')::int AS judged_count,
      count(*) FILTER(WHERE status='unknown' OR verdict->'categories' @> '{"hallucination":"unknown"}'::jsonb
        OR EXISTS(SELECT 1 FROM jsonb_each_text(COALESCE(verdict->'categories','{}'::jsonb)) v WHERE v.value='unknown'))::int AS unknown_count,
      count(*) FILTER(WHERE status='budget_exhausted')::int AS budget_unjudged_count,
      count(*) FILTER(WHERE review_status<>'unreviewed')::int AS reviewed_count,
      count(*) FILTER(WHERE review_queue AND review_status='unreviewed')::int AS review_pending_count
      FROM quality_samples WHERE job_id=$1""",
            job_id,
        )
    )[0]
    return row


async def run(job):
    async def write(query, *args):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            await jobs.require_lease(tx, job["id"], job["lease_token"])
            await tx.execute_raw(query, *args)

    if not (job.get("report") or {}).get("definition_version"):
        await sampling.initialize_samples(job, datetime.now(UTC))
    samples = await db.query_raw(
        "SELECT id FROM quality_samples WHERE job_id=$1 AND status='pending' ORDER BY internal,id LIMIT $2",
        job["id"],
        settings.quality_daily_turns,
    )
    status = "completed"
    for sample in samples:
        await jobs.renew(job["id"], job["lease_token"])
        try:
            context = await sampling.context(sample["id"])
            if context is None:
                # Archived conversations can retain the sample row. A deleted
                # row makes this a no-op; a retained row must count as unknown.
                await write(
                    "UPDATE quality_samples SET status='unknown',verdict='{\"error_type\":\"source_unavailable\"}'::jsonb WHERE id=$1 AND status='pending'",
                    sample["id"],
                )
                continue
            text, _ = await transport.complete(
                job,
                "dashscope",
                settings.quality_judge_model,
                judge.request(bounded_context(context)),
                max_tokens=900,
                json_mode=True,
            )
            verdict = judge.parse(text)
            await write(
                "UPDATE quality_samples SET status='judged',verdict=$2::jsonb WHERE id=$1 AND status='pending'",
                sample["id"],
                json.dumps(verdict),
            )
        except budget.BudgetExhausted:
            await write(
                "UPDATE quality_samples SET status='budget_exhausted' WHERE job_id=$1 AND status='pending'",
                job["id"],
            )
            status = "budget_exhausted"
            break
        except transport.NotConfigured:
            await write(
                "UPDATE quality_samples SET status='unknown',verdict='{\"error_type\":\"not_configured\"}'::jsonb WHERE job_id=$1 AND status='pending'",
                job["id"],
            )
            status = "not_configured"
            break
        except jobs.LostLease:
            raise
        except Exception as exc:
            await write(
                "UPDATE quality_samples SET status='unknown',verdict=$2::jsonb WHERE id=$1",
                sample["id"],
                json.dumps({"error_type": type(exc).__name__}),
            )
    # The queue is a capped view; all original judgements stay available.
    await write(
        """UPDATE quality_samples SET review_queue=true WHERE id IN (
      SELECT id FROM quality_samples WHERE job_id=$1 AND status='judged'
      AND EXISTS(SELECT 1 FROM jsonb_each_text(verdict->'categories') v WHERE v.value='fail')
      ORDER BY (jsonb_array_length(verdict->'critical_kinds')>0) DESC,
        (verdict->>'confidence')::float8 DESC,id LIMIT $2)""",
        job["id"],
        settings.quality_review_limit,
    )
    await jobs.finish(job, status, await counts(job["id"]))
