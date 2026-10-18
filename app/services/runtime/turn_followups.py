"""Explicit parent-turn completion and replayable, source-bound statistics."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import logging

from app.db import db
from app.services.runtime.followup_queue import (
    TURN, FollowupRetryConflict, dispatch_followups, inspect_followups,
    prune_followups, retry_followup,
)
from app.services.runtime.job_queue import enqueue_runtime_job, inspect_runtime_job

logger = logging.getLogger(__name__)
JOB_TYPE = TURN.job_type


class TurnSourceUnavailable(ValueError):
    pass


class TurnFollowupFailed(RuntimeError):
    pass


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 200


async def _snapshot(connection, message_id, conversation_id, turn_id, user_ids):
    """Bounded reads; a transaction caller keeps source rows stable until commit."""
    roots = await connection.query_raw("""SELECT m.created_at,
        jsonb_build_object('user_id',c.user_id,'agent_id',c.agent_id,
          'workspace_id',c.workspace_id,'conversation_id',c.id) AS scope
        FROM messages m JOIN conversations c ON c.id=m.conversation_id
        JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
        LEFT JOIN chat_workspaces w ON w.id=c.workspace_id
        WHERE m.id=$1 AND c.id=$2 AND m.role='assistant' AND NOT c.is_deleted
          AND m.metadata->>'achievement_turn_id'=$3
          AND COALESCE(m.metadata->'human_takeover','false'::jsonb)<>'true'::jsonb
          AND COALESCE(m.metadata->'proactive','false'::jsonb)<>'true'::jsonb
          AND (c.workspace_id IS NULL OR (w.status='active' AND w.user_id=c.user_id AND w.agent_id=c.agent_id))
        FOR SHARE OF m,c,a""", message_id, conversation_id, turn_id)
    if not roots:
        raise TurnSourceUnavailable('missing_or_inactive_turn')
    root = roots[0]
    if root['scope']['workspace_id']:
        workspaces = await connection.query_raw("""SELECT id FROM chat_workspaces
            WHERE id=$1 AND user_id=$2 AND agent_id=$3 AND status='active' FOR SHARE""",
            root['scope']['workspace_id'], root['scope']['user_id'], root['scope']['agent_id'])
        if not workspaces:
            raise TurnSourceUnavailable('inactive_workspace')
    users = await connection.query_raw("""SELECT id,
        message_followup_version(content,role,metadata,created_at) AS version
        FROM messages WHERE conversation_id=$1 AND role='user' AND id=ANY($2::text[])
          AND created_at<=$3::timestamp
          AND COALESCE(metadata->'human_takeover','false'::jsonb)<>'true'::jsonb
        ORDER BY id LIMIT 101 FOR SHARE""", conversation_id, user_ids, root['created_at'])
    if {row['id'] for row in users} != set(user_ids):
        raise TurnSourceUnavailable('incomplete_user_sources')
    assistants = await connection.query_raw("""SELECT id,
        message_followup_version(content,role,metadata,created_at) AS version,
        COALESCE(metadata->'human_takeover','false'::jsonb)='true'::jsonb
          OR COALESCE(metadata->'proactive','false'::jsonb)='true'::jsonb AS excluded
        FROM messages WHERE conversation_id=$1 AND role='assistant'
          AND metadata->>'achievement_turn_id'=$2
        ORDER BY id LIMIT 101 FOR SHARE""", conversation_id, turn_id)
    if (not assistants or len(assistants) > 100 or any(row['excluded'] for row in assistants)
            or message_id not in {row['id'] for row in assistants}):
        raise TurnSourceUnavailable('incomplete_assistant_sources')
    version = hashlib.sha256(json.dumps(
        {'scope': root['scope'], 'root': message_id, 'users': users, 'assistants': assistants},
        sort_keys=True, separators=(',', ':'),
    ).encode()).hexdigest()
    return root['scope'], version


async def register_completed_turn(*, conversation_id, message_id, turn_id, user_message_ids):
    """Await after all fragments succeed, before acknowledging parent completion.

    Missing legacy identity is ineligible, not a guessed latest-message turn.
    Database failures propagate so completion is never falsely acknowledged.
    """
    if (not all(_identity(v) for v in (conversation_id, message_id, turn_id))
            or not isinstance(user_message_ids, list) or not 0 < len(user_message_ids) <= 100
            or not all(_identity(v) for v in user_message_ids)):
        return False
    ids = sorted(set(user_message_ids))
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        try:
            scope, version = await _snapshot(tx, message_id, conversation_id, turn_id, ids)
        except TurnSourceUnavailable:
            return False
        await tx.execute_raw("""INSERT INTO turn_followup_events
            (turn_id,message_id,user_message_ids,scope,source_version)
            VALUES ($1,$2,$3::jsonb,$4::jsonb,$5) ON CONFLICT (turn_id) DO NOTHING""",
            turn_id, message_id, json.dumps(ids), json.dumps(scope), version)
        rows = await tx.query_raw('SELECT source_version FROM turn_followup_events WHERE turn_id=$1', turn_id)
        if not rows or rows[0]['source_version'] != version:
            raise TurnSourceUnavailable('completion_identity_conflict')
    return True


async def _finish(turn_id, generation, status, outcome):
    await db.execute_raw("""UPDATE turn_followup_events SET status=$3,outcome=$4,
        last_error=NULL,updated_at=NOW() WHERE turn_id=$1 AND generation=$2
        AND status NOT IN ('completed','cancelled')""", turn_id, generation, status, outcome)


async def run_turn_followup(payload):
    turn_id, generation = payload.get('turn_id'), payload.get('generation')
    if not _identity(turn_id) or type(generation) is not int or generation < 0:
        raise ValueError('invalid_turn_followup_payload')
    try:
        rows = await db.query_raw('SELECT * FROM turn_followup_events WHERE turn_id=$1 FOR SHARE', turn_id)
        if not rows:
            return
        row = rows[0]
        if row['status'] in {'completed','cancelled'} or row['generation'] > generation:
            return
        if row['generation'] < generation:
            raise RuntimeError('dispatch_not_committed')
        try:
            scope, version = await _snapshot(db, row['message_id'], row['scope']['conversation_id'],
                turn_id, row['user_message_ids'])
        except TurnSourceUnavailable:
            await _finish(turn_id, generation, 'cancelled', 'source_changed_or_unavailable')
            return
        if scope != row['scope'] or version != row['source_version']:
            await _finish(turn_id, generation, 'cancelled', 'source_changed_or_unavailable')
            return
        from app.services.achievements.engine import handle_achievement_event
        from app.services.achievements.events import AssistantTurnAchievementEvent
        from app.services.achievements.mode import achievement_evaluation_enabled
        from app.services.achievements.source import AchievementSourceUnavailable, resolve_message_event
        event = AssistantTurnAchievementEvent(scope['conversation_id'],row['message_id'],[],row['user_message_ids'],turn_id)
        try:
            await resolve_message_event(event)
        except AchievementSourceUnavailable:
            outcome = 'achievement_source_unavailable'
        else:
            if await achievement_evaluation_enabled():
                await handle_achievement_event(event)
                outcome = 'statistics_evaluated'
            else:
                outcome = 'achievement_disabled'
        await _finish(turn_id, generation, 'completed', outcome)
    except Exception as exc:
        try:
            await db.execute_raw("""UPDATE turn_followup_events SET last_error=$3,updated_at=NOW()
                WHERE turn_id=$1 AND generation=$2 AND status NOT IN ('completed','cancelled')""",
                turn_id, generation, type(exc).__name__[:80])
        except Exception:
            logger.warning('Turn followup error receipt unavailable')
        raise TurnFollowupFailed(type(exc).__name__) from None


async def dispatch_turn_followups(*, limit=10, turn_id=None):
    return await dispatch_followups(db, TURN, limit=limit, identity=turn_id,
        enqueue=enqueue_runtime_job, inspect=inspect_runtime_job)


async def prune_turn_followups():
    return await prune_followups(db, TURN)


async def inspect_turn_followups(*, limit=20, status='all'):
    return await inspect_followups(db, TURN, limit=limit, status=status)


async def retry_turn_followup(turn_id):
    return await retry_followup(db, TURN, turn_id, inspect=inspect_runtime_job)
