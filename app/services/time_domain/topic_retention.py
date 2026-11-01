"""Bounded recent observations and audit history, independent of durable opt-outs.

Pruning never deletes a live stop/pause. Source-free watermarks retain replay
protection after inactive projections are removed. They contain no topic label
or chat text and are erased with the workspace.
"""

OBSERVATION_DAYS = 30
MAX_OBSERVATIONS = 64
MAX_EVENTS = 128
INACTIVE_DAYS = 90
# Durable identity/fence slots are never silently evicted. A full workspace
# remains fail-closed until its explicit lifecycle erasure; reads are bounded.
MAX_TOPIC_IDENTITIES = 4096

# Readers use a current window rather than trusting a cached lifetime counter.
# The table is pruned on writes; LIMIT also bounds reads of a delayed cleanup.
OBSERVATION_JOIN = """
LEFT JOIN LATERAL (
 SELECT count(*) FILTER (WHERE o.kind='question')::int AS window_asked_count,
  max(o.statement_at) FILTER (WHERE o.kind='question') AS window_last_question,
  max(o.statement_at) FILTER (WHERE o.kind='mention') AS window_last_mention
 FROM (SELECT kind,statement_at FROM conversation_topic_observations
       WHERE topic_id=t.id AND statement_at>{reference}-interval '30 days'
       ORDER BY statement_at DESC,source_message_id DESC LIMIT 64) o
) observed ON TRUE
"""


def apply_observation_window(row: dict) -> dict:
    """Projection state remains distinct from evidence in the recent window."""
    result = dict(row)
    result["asked_count"] = row["window_asked_count"]
    result["last_agent_question"] = row["window_last_question"]
    result["last_user_mention"] = row["window_last_mention"]
    result["answered_at"] = (
        row["window_last_mention"]
        if row["window_last_question"]
        and row["window_last_mention"]
        and row["window_last_mention"] > row["window_last_question"]
        else None
    )
    return result


async def prune_topic_history(tx, topic_ids: list[str]) -> None:
    """At most eight newly interpreted commands or 128 observed topics per call."""
    if not topic_ids:
        return
    if len(topic_ids) > 128:
        raise ValueError("topic_retention_scope_over_limit")
    # Only rows outside the policy projection are discarded. The projection's
    # decision clock (or its deletion fence) remains the replay authority.
    await tx.execute_raw(
        """DELETE FROM conversation_topic_events e USING (
      SELECT selected.id AS topic_id, h.id FROM unnest($1::text[]) selected(id)
      CROSS JOIN LATERAL (SELECT id FROM conversation_topic_events
       WHERE topic_id=selected.id ORDER BY recorded_at DESC,id DESC OFFSET 128 LIMIT 4096) h
      ) old WHERE e.topic_id=old.topic_id AND e.id=old.id""",
        sorted(set(topic_ids)),
    )
    await tx.execute_raw(
        """DELETE FROM conversation_topic_observations o USING (
      SELECT topic_id,source_message_id,statement_at,
       row_number() OVER (PARTITION BY topic_id ORDER BY statement_at DESC,source_message_id DESC) AS rank
      FROM conversation_topic_observations WHERE topic_id=ANY($1::text[])
      ) old WHERE o.topic_id=old.topic_id AND o.source_message_id=old.source_message_id
      AND (old.rank>64 OR old.statement_at<=clock_timestamp()-interval '30 days')""",
        sorted(set(topic_ids)),
    )


async def prune_inactive_topics(tx, workspace_id: str) -> None:
    # Lazy bounded cleanup does not depend on a nightly task being alive.
    # No unexpired pause or indefinite stop is ever evicted for capacity.
    await tx.execute_raw(
        """DELETE FROM conversation_topics t USING (
      SELECT id FROM conversation_topics WHERE workspace_id=$1
       AND ((action='resume' AND decision_at<clock_timestamp()-interval '90 days')
        OR (expires_at IS NOT NULL AND expires_at<clock_timestamp()-interval '90 days'))
      ORDER BY updated_at,id LIMIT 64 FOR UPDATE
      ) expired WHERE t.id=expired.id""",
        workspace_id,
    )


async def identity_capacity(database, *, user_id: str, workspace_id: str) -> dict:
    rows = await database.query_raw(
        """SELECT count(*)::int AS used FROM (
         SELECT f.topic_id FROM conversation_topic_fences f
         JOIN chat_workspaces w ON w.id=f.workspace_id
         WHERE f.workspace_id=$1 AND w.user_id=$2
         ORDER BY f.topic_id LIMIT $3
        ) bounded""",
        workspace_id,
        user_id,
        MAX_TOPIC_IDENTITIES,
    )
    used = rows[0]["used"]
    return {
        "used": used,
        "limit": MAX_TOPIC_IDENTITIES,
        "exhausted": used >= MAX_TOPIC_IDENTITIES,
        "retention": "workspace_lifetime",
        "count_is_capped": True,
    }
