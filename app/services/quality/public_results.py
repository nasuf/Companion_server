"""Bounded administrator summaries; no prompt content or private answers."""

from datetime import timedelta
from app.db import db


async def summaries(kind, source_key, versions=None):
    async with db.tx(timeout=timedelta(seconds=5)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout='3s'")
        return await tx.query_raw(
            """SELECT j.id,j.source_key,j.source_version,j.status,j.report,j.created_at,j.updated_at,
          j.snapshot->'before_route' AS before_route,j.snapshot->'after_route' AS after_route,
          COALESCE(c.settled_cny,0)::float8 AS settled_cny,
          COALESCE(c.reserved_cny,0)::float8 AS reserved_or_unknown_cny,
          COALESCE(f.failures,'[]'::jsonb) AS failures
          FROM quality_jobs j
          LEFT JOIN LATERAL (SELECT sum(cost_cny) AS settled_cny,
            sum(reserved_cny) FILTER(WHERE cost_cny IS NULL) AS reserved_cny
            FROM quality_charges WHERE job_id=j.id) c ON true
          LEFT JOIN LATERAL (SELECT jsonb_agg(result) AS failures FROM (
            SELECT result FROM quality_results WHERE job_id=j.id AND result->>'pass'='false'
            ORDER BY case_id,repetition LIMIT 20) r) f ON true
          WHERE j.kind=$1 AND j.source_key=$2 AND ($3::text[] IS NULL OR j.source_version=ANY($3::text[]))
          ORDER BY j.created_at DESC,j.id DESC LIMIT 200""",
            kind,
            source_key,
            versions,
        )
