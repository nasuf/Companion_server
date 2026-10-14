"""Complete private candidates, explicit publication and read-only takeover details."""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.db import db
from app.models.offline import OfflineActivityItem, OfflineActivityReviewResponse
from app.redis_client import get_redis
from app.services.admin_takeover.service import _cell
from app.services.offline import activity_service
from app.services.offline import repository as repo
from app.services.offline.activity_generation import generate_activity_preview_cards
from app.services.offline.activity_service import _location_for_activity
from app.services.offline.chat_emit import offline_trace

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 5
_PREVIEW_TTL_SECONDS = 30 * 60


def _preview_key(takeover_id: str, admin_id: str) -> str:
    return f"takeover:activity_preview:{takeover_id}:{admin_id}"


def select_preview_cards(candidates: list[dict], *, limit: int = _PREVIEW_LIMIT) -> list[dict]:
    """Shuffle place candidates and keep one card per location."""
    pool = [item for item in candidates if isinstance(item, dict)]
    random.shuffle(pool)
    chosen: list[dict] = []
    seen: set[str] = set()
    for item in pool:
        name = str(item.get("location_name") or item.get("title") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def public_preview(preview_id: str, card: dict) -> dict:
    # The same public DTO as a real invitation; no raw personalization evidence.
    now = datetime.now(UTC).isoformat()
    # Use the repository's exact event/date/coordinate projection too.
    public = repo.activity_from_row({**card, 'id': preview_id, 'status': 'pending',
        'created_at': now, 'updated_at': now})
    public = OfflineActivityItem(**public).model_dump(mode='json')
    return {**public, 'preview_id': preview_id,
        'image_url': (public['image_urls'] or [None])[0]}


async def _owned_subject(takeover_id: str, admin_id: str) -> dict | None:
    rows = await db.query_raw(
        """
        SELECT user_id AS "userId", workspace_id AS "workspaceId",
               conversation_id AS "conversationId", agent_id AS "agentId"
        FROM human_takeovers
        WHERE id=$1 AND owner_admin_id=$2 AND status='active'
        LIMIT 1
        """,
        takeover_id,
        admin_id,
    )
    if not rows:
        return None
    return rows[0]


async def _read_owned_activity(takeover_id: str, admin_id: str, activity_id: str) -> dict:
    # One bounded read binds the resource to the active claim, user and chat.
    # Do not use repo.get_activity: its event-expiry refresh writes user state.
    rows = await db.query_raw(
        """
        SELECT a.* FROM offline_activity_recommendations a
        JOIN human_takeovers t ON t.user_id = a.user_id
          AND t.conversation_id = a.conversation_id
          AND t.workspace_id IS NOT DISTINCT FROM a.workspace_id
        WHERE t.id=$1 AND t.owner_admin_id=$2 AND t.status='active'
          AND a.id=$3
        LIMIT 1
        """,
        takeover_id, admin_id, activity_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="活动不存在或已无权查看")
    return repo.activity_from_row(rows[0], reveal_task=True)


async def read_takeover_activity(
    takeover_id: str, admin_id: str, activity_id: str,
) -> OfflineActivityItem:
    activity = await _read_owned_activity(takeover_id, admin_id, activity_id)
    return OfflineActivityItem(**(await activity_service._with_completion_feedback(activity)))


async def read_takeover_activity_review(
    takeover_id: str, admin_id: str, activity_id: str,
) -> OfflineActivityReviewResponse:
    activity = await _read_owned_activity(takeover_id, admin_id, activity_id)
    return await activity_service.build_activity_review(activity, generate_note=False)


async def recommend_takeover_activity(
    takeover_id: str, admin_id: str,
) -> dict:
    """Run the same complete generation/publication used by the activity test page."""
    subject = await _owned_subject(takeover_id, admin_id)
    if subject is None:
        raise HTTPException(status_code=403, detail="Not the owning admin")
    user_id = str(_cell(subject, "userId", "user_id"))
    workspace_id = _cell(subject, "workspaceId", "workspace_id")
    conversation_id = _cell(subject, "conversationId", "conversation_id")
    agent_id = _cell(subject, "agentId", "agent_id")
    if not conversation_id or not agent_id:
        raise HTTPException(status_code=422, detail="这个用户还没有可用的会话")

    async def guard(ctx: dict) -> None:
        current = await _owned_subject(takeover_id, admin_id)
        if current is None or (
            str(_cell(current, "userId", "user_id")) != user_id
            or _cell(current, "workspaceId", "workspace_id") != workspace_id
            or _cell(current, "conversationId", "conversation_id") != conversation_id
            or _cell(current, "agentId", "agent_id") != agent_id
        ):
            raise HTTPException(status_code=403, detail="接管状态已变化，无法发送活动")
        if (ctx.get("conversation_id") != conversation_id
                or ctx.get("workspace_id") != workspace_id
                or ctx.get("agent_id") != agent_id):
            raise HTTPException(status_code=409, detail="用户会话已变化，请重新进入接管聊天")

    from app.services.offline.module_settings import is_activity_enabled
    if not await is_activity_enabled():
        raise HTTPException(status_code=403, detail={
            "reason": "activity_disabled", "message": "活动推荐暂未开放。",
        })
    activity = await activity_service.create_recommendation_for_user(
        user_id=user_id, workspace_id=workspace_id, source="manual",
        publication_guard=guard,
    )
    if activity is None:
        raise HTTPException(status_code=503, detail="暂时没找到合适的新去处，请稍后再试")
    return activity


async def list_activity_previews(takeover_id: str, admin_id: str) -> list[dict] | None:
    """Generate up to five complete candidates without persisting or sending."""
    subject = await _owned_subject(takeover_id, admin_id)
    if subject is None:
        return None
    from app.services.offline.module_settings import is_activity_enabled
    if not await is_activity_enabled():
        raise HTTPException(403, detail="活动推荐暂未开放")
    user_id = str(_cell(subject, "userId", "user_id"))
    workspace = _cell(subject, "workspaceId", "workspace_id")
    ctx = await repo.resolve_user_context(user_id, workspace)
    await _guard_preview_subject(takeover_id, admin_id, subject, ctx)
    city, anchor, terms = _location_for_activity(ctx)
    if not city:
        raise ValueError("location_required")
    redis = await get_redis()
    lock_key = f'{_preview_key(takeover_id, admin_id)}:generating'
    token = uuid.uuid4().hex
    if not await redis.set(lock_key, token, nx=True, ex=180):
        raise HTTPException(409, detail='已有一批候选正在生成，请稍后刷新')
    try:
        async with offline_trace('activity_preview', conversation_id=ctx['conversation_id'],
                agent_id=ctx['agent_id'], user_id=user_id):
            cards = await generate_activity_preview_cards(
                user_id=user_id, workspace_id=workspace, city=city, source='manual',
                conversation_id=ctx['conversation_id'], search_location=anchor,
                location_terms=list(terms),
                center=(float(ctx['user_location_latitude']), float(ctx['user_location_longitude']))
                    if ctx.get('user_location_latitude') is not None and ctx.get('user_location_longitude') is not None else None,
            )
        await _guard_preview_subject(takeover_id, admin_id, subject, ctx)
        if not cards:
            raise ValueError('no_suitable_activity')
        stored = [{'preview_id': uuid.uuid4().hex, 'card': card} for card in select_preview_cards(cards)]
        public = [public_preview(item['preview_id'], item['card']) for item in stored]
        if await redis.get(lock_key) != token:
            raise HTTPException(409, detail='这批候选已失效，请刷新后重试')
        await redis.set(_preview_key(takeover_id, admin_id),
            json.dumps({'subject': subject, 'items': stored}, ensure_ascii=False, default=str),
            ex=_PREVIEW_TTL_SECONDS)
        return public
    finally:
        try:
            await redis.eval("if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) end return 0",
                1, lock_key, token)
        except Exception:
            logger.warning('takeover preview lock release failed', exc_info=True)


async def _guard_preview_subject(takeover_id: str, admin_id: str, subject: dict, ctx: dict | None) -> None:
    current = await _owned_subject(takeover_id, admin_id)
    if current != subject:
        raise HTTPException(403, detail='接管状态已变化，请重新进入聊天')
    if not ctx or not ctx.get('conversation_id') or not ctx.get('agent_id'):
        raise ValueError('conversation_required')
    if str(ctx.get('user_id')) != str(_cell(subject, 'userId', 'user_id')):
        raise HTTPException(409, detail='用户会话已变化，请重新进入接管聊天')
    if any(ctx.get(key) != _cell(subject, camel, key) for key, camel in (
            ('conversation_id', 'conversationId'), ('agent_id', 'agentId'), ('workspace_id', 'workspaceId'))):
        raise HTTPException(409, detail='用户会话已变化，请重新进入接管聊天')


async def publish_activity_preview(takeover_id: str, admin_id: str, preview_id: str) -> dict | None:
    """Publish only the chosen fully generated candidate through the shared flow."""
    subject = await _owned_subject(takeover_id, admin_id)
    if subject is None:
        return None
    from app.services.offline.module_settings import is_activity_enabled
    if not await is_activity_enabled():
        raise HTTPException(403, detail='活动推荐暂未开放')
    redis = await get_redis()
    key = _preview_key(takeover_id, admin_id)
    raw = await redis.get(key)
    batch = json.loads(raw) if raw else {}
    if not isinstance(batch, dict) or batch.get('subject') != subject:
        raise ValueError('preview_expired')
    chosen = next((x for x in batch.get('items', []) if x.get('preview_id') == preview_id), None)
    if not chosen:
        raise ValueError('preview_expired')
    # A result survives a lost HTTP response. Concurrent publish attempts cannot send twice.
    result_key = f'{key}:published:{preview_id}'
    result = await redis.get(result_key)
    if result:
        return json.loads(result)
    deadline = chosen['card'].get('expires_at') or chosen['card'].get('ends_at')
    if deadline:
        expires = datetime.fromisoformat(str(deadline).replace('Z', '+00:00'))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires <= datetime.now(UTC):
            raise HTTPException(409, detail='这项活动已过期，请刷新候选列表')
    lock_key = f'{key}:publishing:{preview_id}'
    if not await redis.set(lock_key, '1', nx=True, ex=_PREVIEW_TTL_SECONDS):
        raise HTTPException(409, detail='这条活动正在发送或等待确认，请先查看聊天')
    user_id = str(_cell(subject, 'userId', 'user_id'))
    commit_started = False
    async def guard(ctx):
        await _guard_preview_subject(takeover_id, admin_id, subject, ctx)
    def started():
        nonlocal commit_started
        commit_started = True
    try:
        activity = await activity_service.create_recommendation_for_user(
            user_id=user_id, workspace_id=_cell(subject, 'workspaceId', 'workspace_id'),
            source='manual', publication_guard=guard, prepared_card=chosen['card'],
            message_metadata={'human_takeover': True, 'admin_id': admin_id, 'takeover_id': takeover_id},
            publication_started=started,
        )
    except Exception:
        if not commit_started:
            await redis.delete(lock_key)
            raise
        logger.exception('takeover activity publication outcome uncertain', extra={
            'takeover_id': takeover_id, 'preview_id': preview_id,
            'conversation_id': _cell(subject, 'conversationId', 'conversation_id'),
        })
        raise HTTPException(503, detail='活动发送状态待确认，请先查看聊天，避免重复发送') from None
    if activity is None:
        if not commit_started:
            await redis.delete(lock_key)
        raise HTTPException(503, detail='暂时无法发送这项活动，请先查看聊天')
    result = {'activity_id': activity['id'], 'title': activity.get('title') or ''}
    try:
        await redis.set(result_key, json.dumps(result, ensure_ascii=False), ex=_PREVIEW_TTL_SECONDS)
    except Exception:
        logger.exception('takeover activity publication receipt unavailable', extra={
            'takeover_id': takeover_id, 'preview_id': preview_id, 'activity_id': activity['id'],
        })
        raise HTTPException(503, detail='活动发送状态待确认，请先查看聊天，避免重复发送') from None
    return result


async def recognize_takeover_activity_photos(
    *,
    user_id: str,
    conversation_id: str,
    message_id: str,
    user_text: str,
) -> None:
    """Run the same vision match used in agent chat, without a model reply."""
    try:
        from app.services.chat_media.repo import find_attachments_for_message
        from app.services.chat_media.vision import ensure_vision_summaries
        from app.services.offline.chat_capture import on_user_chat_media

        attachments = await find_attachments_for_message(message_id)
        images = [item for item in attachments if item.kind == "image"]
        if not images:
            return
        await ensure_vision_summaries(images, user_text=user_text)
        rows = await db.query_raw(
            'SELECT workspace_id AS "workspaceId" FROM conversations WHERE id=$1 LIMIT 1',
            conversation_id,
        )
        workspace_id = _cell(rows[0], "workspaceId", "workspace_id") if rows else None
        await on_user_chat_media(
            user_id=user_id,
            workspace_id=workspace_id,
            message_id=message_id,
            attachments=images,
        )
    except Exception as exc:
        logger.warning("takeover activity photo recognition failed: %s", type(exc).__name__)
