"""Daily-facts bursts, deterministic external-first sampling, bounded live detail."""

from datetime import date, datetime, timedelta
import json
from uuid import uuid4

from app.db import db
from app.config import settings
from app.services.effect.daily_facts import burst_details_cte, burst_query_params
from app.services.effect.definitions import internal_user_sql
from app.services.quality.policy import DEFINITION_VERSION
from app.services.quality import jobs


async def select_day(day: date, now: datetime) -> tuple[int, list[dict]]:
    # The same burst eligibility and reply interval as daily_facts. The last
    # user bubble is only the identity; the entire burst is used for judging.
    sql = (
        burst_details_cte()
        + f""", candidates AS (
      SELECT b.user_id,b.conversation_id,b.last_at,u.id AS message_id,r.id AS reply_id,
        r.metadata->>'trace_id' AS trace_id,{internal_user_sql("owner")} AS internal
      FROM day_bursts b CROSS JOIN bind_types p
      JOIN users owner ON owner.id=b.user_id AND owner.status <> 'deleting'
      JOIN conversations conv ON conv.id=b.conversation_id AND NOT conv.is_deleted AND conv.archived_at IS NULL
      JOIN LATERAL (SELECT id FROM messages WHERE conversation_id=b.conversation_id
        AND role='user' AND created_at=b.last_at ORDER BY id DESC LIMIT 1) u ON true
      JOIN LATERAL (SELECT id,metadata FROM messages WHERE conversation_id=b.conversation_id
        AND role='assistant' AND created_at>b.last_at AND created_at<=b.last_at+$7::interval
        AND COALESCE(metadata->>'proactive','false') <> 'true'
        AND NOT (COALESCE(metadata,'{{}}'::jsonb) ? 'human_takeover') ORDER BY created_at,id LIMIT 1) r ON true
      WHERE b.replied AND EXISTS (
        SELECT 1 FROM numbered n JOIN messages m ON m.id=n.id
        WHERE n.conversation_id=b.conversation_id AND n.burst_no=b.burst_no
          AND m.content ~ '[[:alnum:]一-鿿]'
          AND COALESCE(m.metadata->>'type','text') IN ('text','message')
          AND NOT (COALESCE(m.metadata,'{{}}'::jsonb) ?| ARRAY['offering_received','system','auto_card'])
      )
    ), chosen AS (
      SELECT * FROM candidates ORDER BY internal,md5(message_id || $3::text),message_id LIMIT $12
    ) SELECT (SELECT count(*)::int FROM candidates) AS candidate_count,
        COALESCE(jsonb_agg(to_jsonb(chosen)) FILTER(WHERE chosen.message_id IS NOT NULL),'[]'::jsonb) AS selected
      FROM chosen"""
    )
    async with db.tx(timeout=timedelta(seconds=12)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout='10s'")
        row = (
            await tx.query_raw(
                sql, *burst_query_params(day, now), settings.quality_daily_turns
            )
        )[0]
    return row["candidate_count"], row["selected"]


async def initialize_samples(job: dict, now: datetime):
    count, selected = await select_day(
        date.fromisoformat(str(job["local_date"])[:10]), now
    )
    async with db.tx(timeout=timedelta(seconds=10)) as tx:
        await jobs.require_lease(tx, job["id"], job["lease_token"])
        for row in selected:
            # Deletion wins; no retained content or dangling identifiers.
            await tx.execute_raw(
                """INSERT INTO quality_samples(id,job_id,user_id,conversation_id,message_id,reply_id,trace_id,internal)
              SELECT $1,$2,$3,$4,$5,$6,$7,$8 FROM users u JOIN messages m ON m.id=$5
              JOIN messages r ON r.id=$6 WHERE u.id=$3 AND u.status<>'deleting'
              ON CONFLICT(job_id,message_id) DO NOTHING""",
                uuid4().hex,
                job["id"],
                row["user_id"],
                row["conversation_id"],
                row["message_id"],
                row["reply_id"],
                row["trace_id"],
                row["internal"],
            )
        await tx.execute_raw(
            "UPDATE quality_jobs SET report=$2::jsonb,updated_at=NOW() WHERE id=$1",
            job["id"],
            json.dumps(
                dict(
                    candidate_count=count,
                    selected_count=len(selected),
                    definition_version=DEFINITION_VERSION,
                )
            ),
        )


async def context(sample_id: str, database=None) -> dict | None:
    database = database or db
    rows = await database.query_raw(
        """SELECT s.id,s.message_id,s.reply_id,s.trace_id,s.conversation_id,
      m.created_at AS user_at,r.created_at AS reply_at FROM quality_samples s
      JOIN users u ON u.id=s.user_id AND u.status<>'deleting'
      JOIN messages m ON m.id=s.message_id JOIN messages r ON r.id=s.reply_id
      JOIN conversations c ON c.id=s.conversation_id AND NOT c.is_deleted AND c.archived_at IS NULL
      WHERE s.id=$1""",
        sample_id,
    )
    if not rows:
        return None
    row = rows[0]
    # Only messages up to this reply; future messages must never help the judge.
    history = await database.query_raw(
        """SELECT id,role,left(content,3000) AS content,length(content)>3000 AS truncated,metadata,created_at AS "createdAt"
      FROM messages WHERE conversation_id=$1 AND role IN ('user','assistant') AND (created_at < $2::timestamp OR (created_at=$2::timestamp AND id <= $3))
      ORDER BY created_at DESC,id DESC LIMIT 61""",
        row["conversation_id"],
        str(row["user_at"]),
        row["message_id"],
    )
    reply = await database.query_raw(
        """SELECT id,role,left(content,3000) AS content,length(content)>3000 AS truncated,metadata,created_at AS "createdAt"
      FROM messages WHERE conversation_id=$1 AND role IN ('user','assistant') AND (created_at > $2::timestamp OR (created_at=$2::timestamp AND id >= $3))
        AND created_at <= $2::timestamp+INTERVAL '10 minutes' ORDER BY created_at,id LIMIT 31""",
        row["conversation_id"],
        str(row["reply_at"]),
        row["reply_id"],
    )
    from app.services.chat.message_utils import group_dialogue_turns

    groups = group_dialogue_turns(list(reversed(history[:60])))
    if not reply or reply[0]["id"] != row["reply_id"]:
        return None
    answer = group_dialogue_turns(reply[:30])[0]
    # A Trace must be rooted in this conversation's exact message trace.
    # We do not attach an arbitrary trace ID supplied by an administrator.
    traces = (
        await database.query_raw(
            """SELECT inputs_json,outputs_json FROM trace_runs
      WHERE trace_id=$1 AND run_type='llm' AND status='success'
        AND started_at <= $2::timestamp AND octet_length(inputs_json::text)<=40000
        AND octet_length(outputs_json::text)<=40000
      ORDER BY started_at DESC LIMIT 3""",
            row["trace_id"],
            str(row["reply_at"]),
        )
        if row["trace_id"]
        else []
    )
    from app.services.quality.recorded import reply_messages

    return dict(
        history=[
            dict(
                role=g[0]["role"],
                content="\n".join(m["content"] for m in g),
                at=str(g[-1]["createdAt"]),
            )
            for g in groups[-10:]
        ],
        reply="\n".join(m["content"] for m in answer),
        prompt_evidence=[t["inputs_json"] for t in traces],
        recorded_messages=reply_messages(traces, answer[0]["content"]),
        evidence_status="available" if traces else "not_collected",
        truncated=len(history) > 60
        or len(reply) > 30
        or len(groups) > 10
        or any(m["truncated"] for m in history + reply),
        occurred_at=str(row["user_at"]),
    )
