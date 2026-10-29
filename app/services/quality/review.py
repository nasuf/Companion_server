"""Optimistic human review; confirmed failures become owned regression cases."""

from datetime import timedelta
import json

from app.db import db
from app.services.quality import sampling
from app.services.quality.policy import CATEGORIES


class ReviewConflict(ValueError):
    pass


async def save(sample_id, admin_id, revision, decision, category, note):
    if decision not in {"confirmed", "false_positive", "uncertain"}:
        raise ValueError("invalid_review_decision")
    if decision == "confirmed" and (
        category not in CATEGORIES or len(note.strip()) < 5
    ):
        raise ValueError("confirmed_review_requires_category_and_evidence")
    async with db.tx(timeout=timedelta(seconds=12)) as tx:
        rows = await tx.query_raw(
            """SELECT s.* FROM quality_samples s JOIN users u ON u.id=s.user_id
          WHERE s.id=$1 AND u.status<>'deleting' FOR UPDATE OF s""",
            sample_id,
        )
        if not rows:
            raise LookupError("sample_unavailable")
        row = rows[0]
        if row["revision"] != revision:
            raise ReviewConflict("review_changed_refresh_first")
        context = await sampling.context(sample_id, tx)
        if context is None:
            raise LookupError("sample_unavailable")
        await tx.execute_raw(
            """UPDATE quality_samples SET review_status=$2,review_category=$3,review_note=$4,
          reviewed_by=$5,reviewed_at=NOW(),revision=revision+1 WHERE id=$1""",
            sample_id,
            decision,
            category,
            note.strip(),
            admin_id,
        )
        if decision == "confirmed":
            from app.api.admin.bug_reports import _build_eval_case

            case = _build_eval_case(
                report_id=sample_id,
                error_types=[category],
                reason=note,
                assistant_reply=context["reply"],
                user_turns=[
                    {"user": t["content"]}
                    for t in context["history"]
                    if t["role"] == "user"
                ],
                category_override=category,
                priority_override="P0" if category == "hallucination" else "P1",
                description_override=note,
                assertions_override=None,
            )
            case.update(
                id="quality_" + sample_id,
                history=context["history"],
                criterion=note,
                prompt_evidence=context["prompt_evidence"],
                recorded_messages=context.get("recorded_messages"),
                replay_scope="recorded_input_model_only",
                source={"type": "quality_review", "sample_id": sample_id},
            )
            await tx.execute_raw(
                """INSERT INTO quality_cases(id,sample_id,case_data) VALUES($1,$2,$3::jsonb)
              ON CONFLICT(sample_id) DO UPDATE SET case_data=EXCLUDED.case_data,enabled=true,updated_at=NOW()""",
                case["id"],
                sample_id,
                json.dumps(case, ensure_ascii=False),
            )
        else:
            await tx.execute_raw(
                "UPDATE quality_cases SET enabled=false,updated_at=NOW() WHERE sample_id=$1",
                sample_id,
            )
    return {"id": sample_id, "revision": revision + 1, "review_status": decision}
