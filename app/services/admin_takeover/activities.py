"""Admin-picked activity previews for a taken-over user.

Previews stay off the user's chat until the admin publishes one. Publishing
uses the same invite and card the agent would send. Later photo recognition
still runs through the vision model.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from typing import Any

from fastapi import HTTPException

from app.config import settings
from app.db import db
from app.models.offline import OfflineActivityItem, OfflineActivityReviewResponse
from app.redis_client import get_redis
from app.services.admin_takeover.service import _cell
from app.services.offline import activity_service
from app.services.offline import repository as repo
from app.services.offline.activity_generation import (
    _fallback_card,
    _search_activity_candidates,
    generate_activity_invite_message,
)
from app.services.offline.activity_service import _location_for_activity
from app.services.offline.chat_emit import emit_activity_card, emit_assistant, offline_trace
from app.services.offline.geocode import geocode_address, make_place_key

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 5
_PREVIEW_TTL_SECONDS = 30 * 60


def _preview_key(takeover_id: str, admin_id: str) -> str:
    return f"takeover:activity_preview:{takeover_id}:{admin_id}"


def _http_image(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if isinstance(value, dict):
        for key in ("url", "image", "src"):
            found = _http_image(value.get(key))
            if found:
                return found
    return None


def _image_urls(candidate: dict) -> list[str]:
    urls: list[str] = []
    for item in candidate.get("native_images") or candidate.get("image_urls") or []:
        found = _http_image(item)
        if found and found not in urls:
            urls.append(found)
        if len(urls) >= 3:
            break
    return urls


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


def _card_from_candidate(candidate: dict, city: str) -> dict:
    images = _image_urls(candidate)
    title = str(candidate.get("title") or candidate.get("location_name") or "附近的去处")
    location = str(candidate.get("location_name") or title)
    address = str(candidate.get("address") or "")
    summary = str(candidate.get("summary") or f"可以去{location}看看")
    description = str(candidate.get("description") or f"{location}，{address}".strip("，"))
    metadata = candidate.get("discovery_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "title": title,
        "summary": summary,
        "description": description,
        "category": candidate.get("category"),
        "city": candidate.get("city") or city,
        "location_name": location,
        "address": address,
        "official_url": candidate.get("official_url"),
        "image_urls": images,
        "place_lat": candidate.get("place_lat"),
        "place_lng": candidate.get("place_lng"),
        "vibe": candidate.get("vibe") or candidate.get("suitable") or "",
        "suitable": candidate.get("suitable") or "",
        "discovery_metadata": metadata,
        "source": "takeover_admin",
    }


def public_preview(preview_id: str, card: dict) -> dict:
    images = card.get("image_urls") or []
    return {
        "preview_id": preview_id,
        "title": card.get("title") or "",
        "summary": card.get("summary") or "",
        "description": card.get("description") or "",
        "location_name": card.get("location_name") or "",
        "address": card.get("address") or "",
        "city": card.get("city") or "",
        "suitable": card.get("suitable") or "",
        "category": card.get("category") or "",
        "image_url": images[0] if images else None,
    }


async def _owned_subject(takeover_id: str, admin_id: str) -> dict | None:
    rows = await db.query_raw(
        """
        SELECT user_id AS "userId", workspace_id AS "workspaceId",
               conversation_id AS "conversationId"
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


async def _candidate_pool(user_id: str, ctx: dict, city: str, anchor: str) -> list[dict]:
    tags = await repo.list_user_tags(user_id, ctx.get("workspace_id"), limit=9)
    recent = await repo.list_recent_activity_fingerprints(
        user_id, ctx.get("workspace_id"), limit=20,
    )
    center = None
    if ctx.get("user_location_latitude") is not None and ctx.get("user_location_longitude") is not None:
        center = (float(ctx["user_location_latitude"]), float(ctx["user_location_longitude"]))
    if settings.offline_search_provider == "cleversee":
        from app.services.offline.cleversee_discovery import discover

        found = await discover(
            city=city, search_anchor=anchor or city, tags=tags, recent=recent, center=center,
        )
        return [_card_from_candidate(item, city) for item in found if isinstance(item, dict)]
    _filtered, usable, _query = await _search_activity_candidates(
        anchor or city, tags, recent, city=city,
    )
    cards = []
    for result in usable:
        card = _fallback_card(city, tags, [result])
        if card:
            cards.append(card)
    return cards


async def list_activity_previews(takeover_id: str, admin_id: str) -> list[dict] | None:
    """Build five recommendations for the subject. None means the admin does not own it."""
    subject = await _owned_subject(takeover_id, admin_id)
    if subject is None:
        return None
    user_id = str(_cell(subject, "userId", "user_id"))
    ctx = await repo.resolve_user_context(user_id, _cell(subject, "workspaceId", "workspace_id"))
    if not ctx or not ctx.get("conversation_id"):
        raise ValueError("conversation_required")
    city, anchor, _terms = _location_for_activity(ctx)
    if not city:
        raise ValueError("location_required")
    pool = select_preview_cards(await _candidate_pool(user_id, ctx, city, anchor))
    if not pool:
        raise ValueError("no_suitable_activity")
    stored = [{"preview_id": uuid.uuid4().hex[:12], "card": card} for card in pool]
    redis = await get_redis()
    await redis.set(
        _preview_key(takeover_id, admin_id),
        json.dumps(stored, ensure_ascii=False, default=str),
        ex=_PREVIEW_TTL_SECONDS,
    )
    return [public_preview(item["preview_id"], item["card"]) for item in stored]


async def publish_activity_preview(takeover_id: str, admin_id: str, preview_id: str) -> dict | None:
    """Persist the chosen preview and send the invite the way an agent would."""
    subject = await _owned_subject(takeover_id, admin_id)
    if subject is None:
        return None
    redis = await get_redis()
    raw = await redis.get(_preview_key(takeover_id, admin_id))
    stored = json.loads(raw) if raw else []
    chosen = next((item for item in stored if item.get("preview_id") == preview_id), None)
    if not chosen:
        raise ValueError("preview_expired")
    user_id = str(_cell(subject, "userId", "user_id"))
    ctx = await repo.resolve_user_context(user_id, _cell(subject, "workspaceId", "workspace_id"))
    if not ctx or not ctx.get("conversation_id"):
        raise ValueError("conversation_required")
    card = dict(chosen["card"])
    city = str(card.get("city") or "")
    metadata = card.get("discovery_metadata") or {}
    if metadata.get("coordinate_source") == "native_poi" and card.get("place_lat") is not None:
        place_lat, place_lng = card.get("place_lat"), card.get("place_lng")
        place_key = metadata.get("session_key") or ("poi:" + str(metadata.get("poi_id") or ""))
    else:
        coords = await geocode_address(card.get("address"), city)
        place_lat, place_lng = coords if coords else (card.get("place_lat"), card.get("place_lng"))
        place_key = make_place_key(card.get("location_name"), card.get("address"), city)
    async with offline_trace(
        "activity_invite",
        conversation_id=ctx["conversation_id"],
        agent_id=ctx.get("agent_id"),
        user_id=user_id,
    ) as tracer:
        activity = await repo.create_activity({
            **card,
            "user_id": user_id,
            "agent_id": ctx["agent_id"],
            "workspace_id": ctx["workspace_id"],
            "conversation_id": ctx["conversation_id"],
            "status": "pending",
            "place_lat": place_lat,
            "place_lng": place_lng,
            "place_key": place_key,
        })
        message = await generate_activity_invite_message(
            activity=activity, user_id=user_id, workspace_id=ctx["workspace_id"],
        )
        await emit_assistant(
            conversation_id=ctx["conversation_id"],
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            message=message,
            real_world_type="activity",
            source_id=activity["id"],
            trigger_type="offline_activity_recommendation",
            trace_id=tracer.safe_trace_id,
        )
        await emit_activity_card(
            conversation_id=ctx["conversation_id"],
            user_id=user_id,
            agent_id=ctx["agent_id"],
            workspace_id=ctx["workspace_id"],
            activity=activity,
            trigger_type="offline_activity_recommendation_card",
            status_label="待确定",
        )
    remaining = [item for item in stored if item.get("preview_id") != preview_id]
    await redis.set(
        _preview_key(takeover_id, admin_id),
        json.dumps(remaining, ensure_ascii=False, default=str),
        ex=_PREVIEW_TTL_SECONDS,
    )
    return {"activity_id": activity["id"], "title": activity.get("title") or card.get("title")}


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
