"""Durable arrival messages. SQL owns the receipt; Redis is retryable transport.

Messages use stable IDs and a generation/scope fence at their SQL commit. A lost
worker acknowledgement may replay delivery, but cannot insert a second card or
companion message. Existing message IDs also deduplicate client/push delivery.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from app.db import db
from app.services.runtime.followup_queue import (
    ARRIVAL, FollowupRetryConflict, dispatch_followups, inspect_followups,
    prune_followups, retry_followup,
)
from app.services.runtime.job_queue import enqueue_runtime_job, inspect_runtime_job

logger = logging.getLogger(__name__)
JOB_TYPE = ARRIVAL.job_type
_TERMINAL = {'completed', 'cancelled'}


class ArrivalSuperseded(RuntimeError):
    pass


class ArrivalVoiceBindingFailed(LookupError):
    pass


class ArrivalFollowupFailed(RuntimeError):
    """Only an error class reaches the administrative queue, never location/text."""


def delivery_id(activity_id: str, role: str) -> str:
    if role not in {'user', 'assistant'}:
        raise ValueError('invalid_arrival_role')
    return str(uuid5(NAMESPACE_URL, f'companion:offline-arrival:{activity_id}:{role}'))


async def _event(client, activity_id: str, generation: int, *, lock=False):
    rows = await client.query_raw('SELECT * FROM offline_arrival_followups WHERE activity_id=$1'
        + (' FOR UPDATE' if lock else ''), activity_id)
    if not rows or rows[0]['status'] in _TERMINAL or rows[0]['generation'] > generation:
        raise ArrivalSuperseded()
    event = rows[0]
    if event['generation'] < generation:
        raise RuntimeError('dispatch_not_committed')
    scope = event['scope']
    live = await client.query_raw("""SELECT a.id FROM offline_activity_recommendations a
        JOIN users u ON u.id=a.user_id AND u.status='active' AND u.archived_at IS NULL
        JOIN chat_workspaces w ON w.id=$3 AND w.user_id=a.user_id AND w.status='active'
        JOIN ai_agents agent ON agent.id=w.agent_id AND agent.user_id=w.user_id
        JOIN conversations c ON c.id=$4 AND c.workspace_id=w.id AND c.user_id=w.user_id
          AND c.agent_id=w.agent_id AND NOT c.is_deleted
        WHERE a.id=$1 AND a.user_id=$2 AND w.agent_id=$5
          AND a.agent_id=w.agent_id AND a.workspace_id IS NOT DISTINCT FROM $6
          AND (a.conversation_id IS NULL OR a.conversation_id=c.id)
          AND a.reached AND a.status='accepted'
          AND (a.auto_archive_at IS NULL OR a.auto_archive_at>NOW())
          AND a.arrival_confirmed_at=($7::timestamptz AT TIME ZONE 'UTC')
        """ + (' FOR SHARE OF a,w,c,agent,u' if lock else ''),
        activity_id, scope.get('user_id'), scope.get('workspace_id'), scope.get('conversation_id'),
        scope.get('agent_id'), event['source_workspace_id'], event['arrived_at'])
    if not live:
        await client.execute_raw("""UPDATE offline_arrival_followups SET status='cancelled',
            outcome='source_changed_or_unavailable',last_error=NULL,updated_at=NOW()
            WHERE activity_id=$1 AND generation=$2 AND status NOT IN ('completed','cancelled')""",
            activity_id, generation)
        return None
    return event


async def load_delivery(activity_id: str, role: str):
    rows = await db.query_raw('SELECT id,conversation_id,role,content,metadata,created_at FROM messages WHERE id=$1',
        delivery_id(activity_id, role))
    return rows[0] if rows else None


async def persist_delivery(*, activity_id: str, generation: int, role: str,
                           conversation_id: str, content: str, metadata: dict):
    """A short transaction; no model, TTS or network work under database locks."""
    async with db.tx(timeout=timedelta(seconds=5)) as tx:
        event = await _event(tx, activity_id, generation, lock=True)
        if event is None:
            # Return outside the transaction, so cancellation is committed.
            return None
        if event['scope']['conversation_id'] != conversation_id:
            raise RuntimeError('arrival_scope_mismatch')
        mid = delivery_id(activity_id, role)
        rows = await tx.query_raw("""INSERT INTO messages(id,conversation_id,role,content,metadata,created_at)
            VALUES($1,$2,$3,$4,$5::jsonb,CURRENT_TIMESTAMP) ON CONFLICT(id) DO NOTHING
            RETURNING id,conversation_id,role,content,metadata,created_at""",
            mid, conversation_id, role, content, json.dumps(metadata, ensure_ascii=False, default=str))
        inserted = bool(rows)
        if not rows:
            rows = await tx.query_raw('SELECT id,conversation_id,role,content,metadata,created_at FROM messages WHERE id=$1', mid)
        row = rows[0]
        if row['conversation_id'] != conversation_id or row['role'] != role:
            raise RuntimeError('arrival_delivery_scope_mismatch')
        # Bind new voice assets in the same commit as their message. A crash
        # cannot leave a persisted voice bubble pointing at an unbound asset.
        if inserted and role == 'assistant' and metadata.get('display_mode') == 'voice':
            attachments = metadata.get('attachments') or []
            ids = [a['id'] for a in attachments if isinstance(a, dict) and isinstance(a.get('id'), str)]
            if not ids or len(ids) != len(attachments) or len(ids) != len(set(ids)):
                raise ArrivalVoiceBindingFailed('invalid_arrival_voice_assets')
            bound = await tx.execute_raw("""UPDATE chat_message_attachments SET message_id=$1,updated_at=NOW()
                WHERE id=ANY($2::text[]) AND user_id=$3 AND conversation_id=$4 AND message_id IS NULL""",
                mid, ids, event['scope']['user_id'], conversation_id)
            if bound != len(ids):
                raise ArrivalVoiceBindingFailed('arrival_voice_binding_failed')
        return {**row, 'inserted': inserted}


async def run_arrival_followup(payload: dict) -> None:
    aid, generation = payload.get('activity_id'), payload.get('generation')
    if not isinstance(aid, str) or not aid or len(aid) > 200 or type(generation) is not int or generation < 0:
        raise ValueError('invalid_arrival_followup_payload')
    try:
        event = await _event(db, aid, generation)
        if event is None:
            return
        scope = event['scope']
        from app.services.offline import repository as repo
        from app.services.offline.activity_service import _arrival_guide_text
        from app.services.offline.chat_emit import insert_user_activity_card, emit_assistant
        from app.services.offline.activity_companion import start_opening_segment
        activity = await repo.get_activity(aid, scope['user_id'], reveal_task=True)
        ctx = await repo.resolve_user_context(scope['user_id'], scope['workspace_id'])
        if not activity or not ctx or any(ctx.get(key) != scope.get(key)
                for key in ('user_id', 'workspace_id', 'agent_id', 'conversation_id')):
            raise RuntimeError('arrival_context_unavailable')
        receipt = (aid, generation)
        card_id = await insert_user_activity_card(conversation_id=scope['conversation_id'],
            workspace_id=scope['workspace_id'], activity=activity,
            trigger_type='offline_activity_arrived_card', status_label='我到了', arrival_delivery=receipt)
        if not card_id:
            return
        existing = await load_delivery(aid, 'assistant')
        guide = existing['content'] if existing else await asyncio.wait_for(_arrival_guide_text(activity, ctx), 45)
        guide_id = await emit_assistant(conversation_id=scope['conversation_id'], user_id=scope['user_id'],
            agent_id=scope['agent_id'], workspace_id=scope['workspace_id'], message=guide,
            real_world_type='activity', source_id=aid, trigger_type='offline_activity_arrival_guide',
            arrival_delivery=receipt)
        if not guide_id:
            return
        # The companion helper already ignores a segment that was started or
        # superseded by a user's reply. Retrying cannot reset its clock.
        await start_opening_segment(aid, scope['user_id'])
        await db.execute_raw("""UPDATE offline_arrival_followups SET status='completed',outcome='delivered',
            last_error=NULL,updated_at=NOW() WHERE activity_id=$1 AND generation=$2
            AND status NOT IN ('completed','cancelled')""", aid, generation)
    except ArrivalSuperseded:
        return
    except Exception as exc:
        try:
            await db.execute_raw("""UPDATE offline_arrival_followups SET last_error=$3,updated_at=NOW()
                WHERE activity_id=$1 AND generation=$2 AND status NOT IN ('completed','cancelled')""",
                aid, generation, type(exc).__name__[:80])
        except Exception:
            logger.warning('Arrival followup error receipt unavailable')
        raise ArrivalFollowupFailed(type(exc).__name__) from None


async def dispatch_arrival_followups(*, limit=10, activity_id=None):
    return await dispatch_followups(db, ARRIVAL, limit=limit, identity=activity_id,
        enqueue=enqueue_runtime_job, inspect=inspect_runtime_job)


async def inspect_arrival_followups(*, limit=20, status='all'):
    return await inspect_followups(db, ARRIVAL, limit=limit, status=status)


async def retry_arrival_followup(activity_id: str):
    return await retry_followup(db, ARRIVAL, activity_id, inspect=inspect_runtime_job)


async def prune_arrival_followups():
    return await prune_followups(db, ARRIVAL)
