from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.db import db
from app.services import profile_tags
from app.services.offline.guidance import safe_guidance
from app.services.offline.user_tags import derive_user_tags

logger = logging.getLogger(__name__)


def new_id() -> str:
    return uuid4().hex


def now_utc() -> datetime:
    return datetime.now(UTC)


def _field(row: Any, snake: str, camel: str | None = None, default: Any = None) -> Any:
    if isinstance(row, dict):
        if snake in row:
            return row[snake]
        if camel and camel in row:
            return row[camel]
        return default
    if hasattr(row, snake):
        return getattr(row, snake)
    if camel and hasattr(row, camel):
        return getattr(row, camel)
    return default


def _json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, (dict, list)):
        return data
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(default)) else default
        except Exception:
            return default
    return default


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    return text or None


def _timestamp_or_none(value: Any) -> datetime | str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(normalized)
        return normalized
    except ValueError:
        logger.warning("Dropping invalid offline activity timestamp: %r", text[:120])
        return None


def activity_from_row(row: Any, *, reveal_task: bool = False) -> dict[str, Any]:
    status = str(_field(row, "status") or "pending")
    return {
        "id": str(_field(row, "id")),
        "user_id": _field(row, "user_id", "userId"),
        "agent_id": _field(row, "agent_id", "agentId"),
        "workspace_id": _field(row, "workspace_id", "workspaceId"),
        "conversation_id": _field(row, "conversation_id", "conversationId"),
        "status": status,
        "title": str(_field(row, "title") or ""),
        "summary": str(_field(row, "summary") or ""),
        "description": str(_field(row, "description") or ""),
        "category": _field(row, "category"),
        "city": _field(row, "city"),
        "location_name": _field(row, "location_name", "locationName"),
        "address": _field(row, "address"),
        "vibe": _field(row, "vibe"),
        "suitable": _field(row, "suitable"),
        "starts_at": _iso(_field(row, "starts_at", "startsAt")),
        "ends_at": _iso(_field(row, "ends_at", "endsAt")),
        "official_url": _field(row, "official_url", "officialUrl"),
        "image_urls": list(_json(_field(row, "image_urls", "imageUrls"), [])),
        "task_hint": _field(row, "task_hint", "taskHint"),
        "easter_egg_task": (
            _json(_field(row, "easter_egg_task", "easterEggTask"), None)
            if reveal_task or status in {"accepted", "completed"}
            else None
        ),
        "search_sources": list(_json(_field(row, "search_sources", "searchSources"), [])),
        # place_* 为内部字段（到达校验/同地点复用用），OfflineActivityItem 未声明，
        # 序列化到前端时 pydantic 自动忽略，不外泄经纬度。
        "place_lat": _field(row, "place_lat", "placeLat"),
        "place_lng": _field(row, "place_lng", "placeLng"),
        "place_key": _field(row, "place_key", "placeKey"),
        "reached": bool(_field(row, "reached", default=False)),
        "arrival_confirmed_at": _iso(_field(row, "arrival_confirmed_at", "arrivalConfirmedAt")),
        "conditions_ready_at": _iso(
            _field(row, "conditions_ready_at", "conditionsReadyAt")
        ),
        "prophecy_text": _field(row, "prophecy_text", "prophecyText"),
        "auto_archive_at": _iso(_field(row, "auto_archive_at", "autoArchiveAt")),
        "travel_note": _field(row, "travel_note", "travelNote"),
        "miss_count": int(_field(row, "miss_count", "missCount", 0) or 0),
        "hint_count": int(_field(row, "hint_count", "hintCount", 0) or 0),
        # Internal-only companion fields. OfflineActivityItem ignores them.
        "focus_condition_id": _field(
            row, "focus_condition_id", "focusConditionId"
        ),
        "next_companion_at": _iso(
            _field(row, "next_companion_at", "nextCompanionAt")
        ),
        "last_companion_at": _iso(
            _field(row, "last_companion_at", "lastCompanionAt")
        ),
        "companion_claim_token": _field(
            row, "companion_claim_token", "companionClaimToken"
        ),
        "companion_claimed_at": _iso(
            _field(row, "companion_claimed_at", "companionClaimedAt")
        ),
        "companion_state": _json(
            _field(row, "companion_state", "companionState"), {}
        ),
        "fragment_count": int(_field(row, "fragment_count", "fragmentCount", 0) or 0),
        "accepted_at": _iso(_field(row, "accepted_at", "acceptedAt")),
        "ignored_at": _iso(_field(row, "ignored_at", "ignoredAt")),
        "completed_at": _iso(_field(row, "completed_at", "completedAt")),
        "expires_at": _iso(_field(row, "expires_at", "expiresAt")),
        "completion_feedback": None,
        "created_at": _iso(_field(row, "created_at", "createdAt")) or "",
        "updated_at": _iso(_field(row, "updated_at", "updatedAt")) or "",
    }


async def resolve_user_context(user_id: str, workspace_id: str | None = None) -> dict[str, Any] | None:
    if workspace_id:
        rows = await db.query_raw(
            """
            SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
                   a.city AS agent_city, a.occupation AS agent_occupation,
                   COALESCE(a.current_mbti, a.mbti) AS agent_mbti,
                   u.display_name AS user_name,
                   c.id AS conversation_id, u.created_at AS user_created_at,
                   u.location_latitude AS user_location_latitude,
                   u.location_longitude AS user_location_longitude,
                   u.location_city AS user_location_city,
                   u.location_region AS user_location_region,
                   u.location_country AS user_location_country,
                   u.location_permission_status AS user_location_permission_status,
                   u.location_updated_at AS user_location_updated_at
            FROM chat_workspaces w
            JOIN ai_agents a ON a.id = w.agent_id
            JOIN users u ON u.id = w.user_id
            LEFT JOIN LATERAL (
                SELECT id FROM conversations
                WHERE workspace_id = w.id AND is_deleted = FALSE
                ORDER BY updated_at DESC LIMIT 1
            ) c ON TRUE
            WHERE w.id = $1 AND w.user_id = $2 AND w.status = 'active'
            LIMIT 1
            """,
            workspace_id,
            user_id,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
                   a.city AS agent_city, a.occupation AS agent_occupation,
                   COALESCE(a.current_mbti, a.mbti) AS agent_mbti,
                   u.display_name AS user_name,
                   c.id AS conversation_id, u.created_at AS user_created_at,
                   u.location_latitude AS user_location_latitude,
                   u.location_longitude AS user_location_longitude,
                   u.location_city AS user_location_city,
                   u.location_region AS user_location_region,
                   u.location_country AS user_location_country,
                   u.location_permission_status AS user_location_permission_status,
                   u.location_updated_at AS user_location_updated_at
            FROM chat_workspaces w
            JOIN ai_agents a ON a.id = w.agent_id
            JOIN users u ON u.id = w.user_id
            LEFT JOIN LATERAL (
                SELECT id FROM conversations
                WHERE workspace_id = w.id AND is_deleted = FALSE
                ORDER BY updated_at DESC LIMIT 1
            ) c ON TRUE
            WHERE w.user_id = $1 AND w.status = 'active'
            ORDER BY w.created_at DESC
            LIMIT 1
            """,
            user_id,
        )
    if not rows:
        return None
    row = rows[0]
    latitude = _field(row, "user_location_latitude", "userLocationLatitude")
    longitude = _field(row, "user_location_longitude", "userLocationLongitude")
    city = _field(row, "user_location_city", "userLocationCity")
    region = _field(row, "user_location_region", "userLocationRegion")
    permission_status = _field(
        row, "user_location_permission_status", "userLocationPermissionStatus"
    )
    return {
        "workspace_id": _field(row, "workspace_id", "workspaceId"),
        "user_id": _field(row, "user_id", "userId"),
        "agent_id": _field(row, "agent_id", "agentId"),
        "agent_name": _field(row, "agent_name", "agentName") or "伴生",
        "agent_city": _field(row, "agent_city", "agentCity"),
        "agent_occupation": _field(row, "agent_occupation", "agentOccupation"),
        "agent_mbti": _json(_field(row, "agent_mbti", "agentMbti"), {}),
        "user_name": _field(row, "user_name", "userName"),
        "conversation_id": _field(row, "conversation_id", "conversationId"),
        "user_created_at": _field(row, "user_created_at", "userCreatedAt"),
        "user_location_latitude": latitude,
        "user_location_longitude": longitude,
        "user_location_city": city,
        "user_location_region": region,
        "user_location_country": _field(row, "user_location_country", "userLocationCountry"),
        "user_location_permission_status": permission_status,
        "user_location_updated_at": _field(
            row, "user_location_updated_at", "userLocationUpdatedAt"
        ),
        "has_location": latitude is not None
        and longitude is not None
        and permission_status in {"whileInUse", "always"},
    }


async def ensure_trigger_state(user_id: str, agent_id: str, workspace_id: str | None) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        INSERT INTO real_world_trigger_states (id, user_id, agent_id, workspace_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, agent_id) DO UPDATE
        SET workspace_id = COALESCE(EXCLUDED.workspace_id, real_world_trigger_states.workspace_id),
            updated_at = CURRENT_TIMESTAMP
        RETURNING *
        """,
        new_id(),
        user_id,
        agent_id,
        workspace_id,
    )
    return dict(rows[0])


async def list_real_world_contexts(*, limit: int = 500) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
               a.city AS agent_city, c.id AS conversation_id, u.created_at AS user_created_at,
               u.location_latitude AS user_location_latitude,
               u.location_longitude AS user_location_longitude,
               u.location_city AS user_location_city,
               u.location_region AS user_location_region,
               u.location_country AS user_location_country,
               u.location_permission_status AS user_location_permission_status,
               u.location_updated_at AS user_location_updated_at,
               s.next_activity_recommendation_at, s.last_activity_recommendation_at,
               s.last_gift_paid_at
        FROM chat_workspaces w
        JOIN ai_agents a ON a.id = w.agent_id
        JOIN users u ON u.id = w.user_id
        LEFT JOIN real_world_trigger_states s ON s.user_id = w.user_id AND s.agent_id = w.agent_id
        LEFT JOIN LATERAL (
            SELECT id FROM conversations
            WHERE workspace_id = w.id AND is_deleted = FALSE
            ORDER BY updated_at DESC LIMIT 1
        ) c ON TRUE
        WHERE w.status = 'active'
        ORDER BY w.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [
        {
            "workspace_id": _field(row, "workspace_id", "workspaceId"),
            "user_id": _field(row, "user_id", "userId"),
            "agent_id": _field(row, "agent_id", "agentId"),
            "agent_name": _field(row, "agent_name", "agentName") or "伴生",
            "agent_city": _field(row, "agent_city", "agentCity"),
            "conversation_id": _field(row, "conversation_id", "conversationId"),
            "user_created_at": _field(row, "user_created_at", "userCreatedAt"),
            "user_location_latitude": _field(
                row, "user_location_latitude", "userLocationLatitude"
            ),
            "user_location_longitude": _field(
                row, "user_location_longitude", "userLocationLongitude"
            ),
            "user_location_city": _field(row, "user_location_city", "userLocationCity"),
            "user_location_region": _field(row, "user_location_region", "userLocationRegion"),
            "user_location_country": _field(row, "user_location_country", "userLocationCountry"),
            "user_location_permission_status": _field(
                row, "user_location_permission_status", "userLocationPermissionStatus"
            ),
            "user_location_updated_at": _field(
                row, "user_location_updated_at", "userLocationUpdatedAt"
            ),
            "has_location": _field(row, "user_location_latitude", "userLocationLatitude")
            is not None
            and _field(row, "user_location_longitude", "userLocationLongitude") is not None
            and _field(
                row, "user_location_permission_status", "userLocationPermissionStatus"
            )
            in {"whileInUse", "always"},
            "next_activity_recommendation_at": _field(
                row, "next_activity_recommendation_at", "nextActivityRecommendationAt"
            ),
            "last_activity_recommendation_at": _field(
                row, "last_activity_recommendation_at", "lastActivityRecommendationAt"
            ),
            "last_gift_paid_at": _field(row, "last_gift_paid_at", "lastGiftPaidAt"),
        }
        for row in rows or []
    ]


async def list_user_tags(
    user_id: str,
    workspace_id: str | None,
    *,
    agent_id: str | None = None,
    limit: int = 9,
) -> list[str]:
    if agent_id:
        try:
            persisted = await profile_tags.list_profile_tags(
                user_id,
                workspace_id,
                agent_id=agent_id,
                limit=limit,
            )
            if persisted:
                return persisted
        except Exception as exc:
            logger.warning("Falling back to rule profile tags: %s", exc)
    rows = await db.query_raw(
        """
        SELECT content, main_category, sub_category, importance, updated_at
        FROM memories_user
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND COALESCE(content, '') <> ''
          AND COALESCE(sub_category, '') <> '提醒'
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3 * 4
        """,
        user_id,
        workspace_id,
        limit,
    )
    return derive_user_tags(list(rows or []), limit=limit)


async def memory_brief(user_id: str, workspace_id: str | None, *, limit: int = 60) -> str:
    rows = await db.query_raw(
        """
        SELECT content, main_category, sub_category
        FROM (
            SELECT content, main_category, sub_category, importance, updated_at
            FROM memories_user
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = FALSE
            UNION ALL
            SELECT content, main_category, sub_category, importance, updated_at
            FROM memories_ai
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = FALSE
        ) m
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    parts: list[str] = []
    for row in rows or []:
        label = " / ".join(
            part for part in [
                str(_field(row, "main_category", "mainCategory") or ""),
                str(_field(row, "sub_category", "subCategory") or ""),
            ] if part
        )
        text = str(_field(row, "content") or "").strip()
        if text:
            parts.append(f"- {label}: {text}" if label else f"- {text}")
    return "\n".join(parts)[:3000]


async def create_activity(data: dict[str, Any]) -> dict[str, Any]:
    activity_id = data.get("id") or new_id()
    starts_at = _timestamp_or_none(data.get("starts_at"))
    ends_at = _timestamp_or_none(data.get("ends_at"))
    expires_at = _timestamp_or_none(data.get("expires_at"))
    rows = await db.query_raw(
        """
        INSERT INTO offline_activity_recommendations (
            id, user_id, agent_id, workspace_id, conversation_id, status, source,
            title, summary, description, category, city, location_name, address,
            starts_at, ends_at, official_url, image_urls, search_sources,
            easter_egg_task, task_hint, expires_at,
            place_lat, place_lng, place_key, vibe, suitable
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12, $13, $14,
            $15::timestamptz, $16::timestamptz, $17, $18::jsonb, $19::jsonb,
            $20::jsonb, $21, $22::timestamptz,
            $23, $24, $25, $26, $27
        )
        RETURNING *
        """,
        activity_id,
        data["user_id"],
        data["agent_id"],
        data.get("workspace_id"),
        data.get("conversation_id"),
        data.get("status", "pending"),
        data.get("source", "scheduled"),
        data["title"],
        data.get("summary", ""),
        data.get("description", ""),
        data.get("category"),
        data.get("city"),
        data.get("location_name"),
        data.get("address"),
        starts_at,
        ends_at,
        data.get("official_url"),
        json.dumps(data.get("image_urls") or [], ensure_ascii=False),
        json.dumps(data.get("search_sources") or [], ensure_ascii=False),
        json.dumps(data.get("easter_egg_task") or {}, ensure_ascii=False),
        data.get("task_hint"),
        expires_at,
        data.get("place_lat"),
        data.get("place_lng"),
        data.get("place_key"),
        (str(data.get("vibe") or "").strip() or None),
        (str(data.get("suitable") or "").strip() or None),
    )
    return activity_from_row(rows[0])


async def list_activities(user_id: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT 100
        """,
        user_id,
        workspace_id,
    )
    return [activity_from_row(row) for row in rows or []]


async def get_active_activity_brief(
    user_id: str, workspace_id: str | None = None
) -> dict[str, Any] | None:
    """当前进行中(accepted)活动的轻量概要，供聊天主回复注入外出情境。

    只取最近一条 accepted 活动（走 offline_activity_user_status_idx 索引，热路径友好）。
    只把预先校验过的安全线索放进聊天 prompt；物品名/类别/别名永不返回。
    """
    rows = await db.query_raw(
        """
        SELECT a.title, a.location_name, a.category, a.summary, a.reached,
               f.short_name AS focus_short_name,
               f.category AS focus_category,
               f.criteria AS focus_criteria,
               f.guidance_profile
        FROM offline_activity_recommendations a
        LEFT JOIN offline_shooting_conditions f
          ON f.id = a.focus_condition_id
         AND f.recommendation_id = a.id
         AND f.triggered = FALSE
        WHERE a.user_id = $1
          AND ($2::text IS NULL OR a.workspace_id = $2)
          AND a.status = 'accepted'
        ORDER BY a.reached DESC,
                 a.arrival_confirmed_at DESC NULLS LAST,
                 a.created_at DESC
        LIMIT 1
        """,
        user_id,
        workspace_id,
    )
    if not rows:
        return None
    r = rows[0]
    profile = _json(_field(r, "guidance_profile", "guidanceProfile"), {})
    focus_short_name = _field(r, "focus_short_name", "focusShortName")
    safe_hint = (
        safe_guidance(
            {
                "short_name": focus_short_name,
                "category": _field(r, "focus_category", "focusCategory"),
                "criteria": _field(r, "focus_criteria", "focusCriteria"),
                "guidance_profile": profile,
            },
            "weak",
        )
        if focus_short_name
        else ""
    )
    return {
        "title": str(_field(r, "title") or "").strip(),
        "location_name": str(_field(r, "location_name", "locationName") or "").strip(),
        "category": str(_field(r, "category") or "").strip(),
        "summary": str(_field(r, "summary") or "").strip(),
        "reached": bool(_field(r, "reached")),
        "safe_hint": safe_hint,
    }


async def list_recent_activity_fingerprints(
    user_id: str,
    workspace_id: str | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    rows = await db.query_raw(
        """
        SELECT title, location_name, address, category
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    items: list[dict[str, str]] = []
    for row in rows or []:
        items.append(
            {
                "title": str(_field(row, "title") or "").strip(),
                "location_name": str(
                    _field(row, "location_name", "locationName") or ""
                ).strip(),
                "address": str(_field(row, "address") or "").strip(),
                "category": str(_field(row, "category") or "").strip(),
            }
        )
    return items


async def clear_user_activities(user_id: str) -> dict[str, int]:
    feedback_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_feedback
        WHERE recommendation_id IN (
            SELECT id FROM offline_activity_recommendations WHERE user_id = $1
        )
        RETURNING id
        """,
        user_id,
    )
    activity_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_recommendations
        WHERE user_id = $1
        RETURNING id
        """,
        user_id,
    )
    return {
        "deleted_activities": len(activity_rows or []),
        "deleted_feedback": len(feedback_rows or []),
    }


async def get_activity(activity_id: str, user_id: str, *, reveal_task: bool = False) -> dict[str, Any] | None:
    rows = await db.query_raw(
        "SELECT * FROM offline_activity_recommendations WHERE id = $1 AND user_id = $2 LIMIT 1",
        activity_id,
        user_id,
    )
    return activity_from_row(rows[0], reveal_task=reveal_task) if rows else None


async def update_activity_status(
    activity_id: str,
    user_id: str,
    status: str,
    *,
    completed: bool = False,
) -> dict[str, Any] | None:
    column = {
        "accepted": "accepted_at",
        "ignored": "ignored_at",
        "completed": "completed_at",
        "expired": "updated_at",
    }.get(status, "updated_at")
    rows = await db.query_raw(
        f"""
        UPDATE offline_activity_recommendations AS activity
        SET status = $3,
            {column} = CURRENT_TIMESTAMP,
            next_companion_at = CASE
                WHEN $3 <> 'accepted' THEN NULL ELSE next_companion_at
            END,
            focus_condition_id = CASE
                WHEN $3 <> 'accepted' THEN NULL ELSE focus_condition_id
            END,
            companion_claim_token = NULL,
            companion_claimed_at = NULL,
            companion_state = CASE
                WHEN $3 <> 'accepted'
                THEN COALESCE(companion_state, '{{}}'::jsonb)
                     || '{{"mode":"stopped"}}'::jsonb
                ELSE companion_state
            END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        activity_id,
        user_id,
        status,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def mark_arrived(
    activity_id: str,
    user_id: str,
    *,
    lat: float | None,
    lng: float | None,
) -> dict[str, Any] | None:
    """确认到达：置 reached + 到达时间 + 24h 自动归档截止。仅 accepted 且未到达时生效。"""
    async with db.tx() as tx:
        current = await tx.query_raw(
            """
            SELECT workspace_id, status, reached
            FROM offline_activity_recommendations
            WHERE id = $1 AND user_id = $2
            FOR UPDATE
            """,
            activity_id,
            user_id,
        )
        if not current:
            return None
        row = current[0]
        if str(_field(row, "status") or "") != "accepted":
            return None
        if bool(_field(row, "reached")):
            existing = await tx.query_raw(
                "SELECT * FROM offline_activity_recommendations WHERE id = $1",
                activity_id,
            )
            return (
                activity_from_row(existing[0], reveal_task=True)
                if existing
                else None
            )
        workspace_id = _field(row, "workspace_id", "workspaceId")
        lock_scope = f"offline-arrive:{user_id}:{workspace_id or 'legacy'}"
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            lock_scope,
        )
        other = await tx.query_raw(
            """
            SELECT 1
            FROM offline_activity_recommendations
            WHERE user_id = $1
              AND id <> $2
              AND status = 'accepted'
              AND reached = TRUE
              AND workspace_id IS NOT DISTINCT FROM $3
            LIMIT 1
            """,
            user_id,
            activity_id,
            workspace_id,
        )
        if other:
            return None
        rows = await tx.query_raw(
            """
            UPDATE offline_activity_recommendations
            SET reached = TRUE,
                arrival_confirmed_at = CURRENT_TIMESTAMP,
                arrival_lat = $3,
                arrival_lng = $4,
                auto_archive_at = CURRENT_TIMESTAMP + INTERVAL '24 hours',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1 AND user_id = $2
              AND status = 'accepted' AND reached = FALSE
            RETURNING *
            """,
            activity_id,
            user_id,
            lat,
            lng,
        )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def set_prophecy(
    activity_id: str,
    user_id: str,
    text: str,
) -> dict[str, Any] | None:
    """写入此行小预言。原子守卫：仅「未抽过且未到达」时成功，防重复抽取。"""
    rows = await db.query_raw(
        """
        UPDATE offline_activity_recommendations
        SET prophecy_text = $3,
            prophecy_drawn_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
          AND prophecy_text IS NULL AND reached = FALSE AND status = 'accepted'
        RETURNING *
        """,
        activity_id,
        user_id,
        text,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def mark_archived(
    activity_id: str,
    user_id: str,
    *,
    auto: bool = False,
) -> dict[str, Any] | None:
    """归档「收好」：accepted -> completed，记归档时间。仅 accepted 生效（幂等由调用方处理）。"""
    rows = await db.query_raw(
        """
        UPDATE offline_activity_recommendations
        SET status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            archived_at = CURRENT_TIMESTAMP,
            auto_archived = $3,
            next_companion_at = NULL,
            focus_condition_id = NULL,
            companion_claim_token = NULL,
            companion_claimed_at = NULL,
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || '{"mode":"stopped"}'::jsonb,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2 AND status = 'accepted'
        RETURNING *
        """,
        activity_id,
        user_id,
        auto,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def list_due_for_auto_archive(*, limit: int = 200) -> list[dict[str, Any]]:
    """spec §3.6：进行中 + 已到达 + 距确认到达 ≥24h（auto_archive_at 到期）的活动。"""
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE status = 'accepted' AND reached = TRUE
          AND auto_archive_at IS NOT NULL
          AND auto_archive_at <= CURRENT_TIMESTAMP
        ORDER BY auto_archive_at ASC
        LIMIT $1
        """,
        limit,
    )
    return [activity_from_row(r, reveal_task=True) for r in rows]


async def find_accepted_by_place_key(
    user_id: str,
    place_key: str,
    *,
    exclude_id: str,
) -> dict[str, Any] | None:
    """同地点复用：找该用户同 place_key 的其它进行中(accepted)活动。"""
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE user_id = $1 AND place_key = $2 AND status = 'accepted' AND id != $3
        ORDER BY created_at DESC
        LIMIT 1
        """,
        user_id,
        place_key,
        exclude_id,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


# ---------------------------------------------------------------------------
# 拍摄条件集合（spec §3.4）：到达后由大模型生成，永不下发前端明文。
# ---------------------------------------------------------------------------
async def replace_shooting_items_and_initialize(
    *,
    recommendation_id: str,
    user_id: str,
    items: list[dict[str, Any]],
    focus_index: int,
    next_companion_at: datetime,
) -> tuple[list[dict[str, Any]], bool]:
    """Atomically replace a partial set and mark the active activity ready."""
    prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        short_name = str(item.get("short_name") or "").strip()[:60]
        if not short_name:
            continue
        profile = item.get("guidance_profile")
        prepared.append(
            {
                "id": new_id(),
                "short_name": short_name,
                "category": str(item.get("category") or "").strip()[:60] or None,
                "criteria": str(item.get("criteria") or "").strip()[:500],
                "guidance_profile": profile if isinstance(profile, dict) else {},
                "sort_order": idx,
            }
        )
    if not prepared:
        return [], False
    focus = prepared[max(0, min(focus_index, len(prepared) - 1))]
    async with db.tx() as tx:
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            recommendation_id,
        )
        current = await tx.query_raw(
            """
            SELECT conditions_ready_at
            FROM offline_activity_recommendations
            WHERE id = $1 AND user_id = $2
              AND status = 'accepted' AND reached = TRUE
            FOR UPDATE
            """,
            recommendation_id,
            user_id,
        )
        if not current:
            raise RuntimeError("activity no longer eligible for condition initialization")
        if _field(current[0], "conditions_ready_at", "conditionsReadyAt") is not None:
            existing = await tx.query_raw(
                """
                SELECT id, short_name, category, criteria, guidance_profile
                FROM offline_shooting_conditions
                WHERE recommendation_id = $1
                ORDER BY sort_order
                """,
                recommendation_id,
            )
            return [
                {
                    "id": str(_field(row, "id")),
                    "short_name": _field(row, "short_name", "shortName"),
                    "category": _field(row, "category"),
                    "criteria": _field(row, "criteria"),
                    "guidance_profile": _json(
                        _field(row, "guidance_profile", "guidanceProfile"),
                        {},
                    ),
                }
                for row in existing
            ], False
        await tx.execute_raw(
            "DELETE FROM offline_shooting_conditions WHERE recommendation_id = $1",
            recommendation_id,
        )
        for item in prepared:
            await tx.execute_raw(
                """
                INSERT INTO offline_shooting_conditions
                    (id, recommendation_id, short_name, criteria, category,
                     guidance_profile, sort_order)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                item["id"],
                recommendation_id,
                item["short_name"],
                item["criteria"],
                item["category"],
                json.dumps(item["guidance_profile"], ensure_ascii=False),
                item["sort_order"],
            )
        rows = await tx.query_raw(
            """
            UPDATE offline_activity_recommendations
            SET conditions_ready_at = CURRENT_TIMESTAMP,
                focus_condition_id = $3,
                next_companion_at = $4::timestamptz,
                companion_claim_token = NULL,
                companion_claimed_at = NULL,
                companion_state = COALESCE(companion_state, '{}'::jsonb)
                    || '{"mode":"guided","unanswered_count":0,"recent_modes":[]}'::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1 AND user_id = $2
              AND status = 'accepted' AND reached = TRUE
            RETURNING id
            """,
            recommendation_id,
            user_id,
            focus["id"],
            next_companion_at,
        )
        if not rows:
            raise RuntimeError("activity no longer eligible for condition initialization")
    return prepared, True


async def create_prewritten_fragments(
    recommendation_id: str,
    condition_id: str,
    tier_texts: dict[str, str],
) -> None:
    """为某个拍摄物品写入分档预生成回忆（tier -> text）。"""
    for tier, text in tier_texts.items():
        text = (text or "").strip()
        if not text:
            continue
        await db.execute_raw(
            """
            INSERT INTO offline_prewritten_fragments
                (id, recommendation_id, condition_id, tier, text)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (condition_id, tier) DO NOTHING
            """,
            new_id(),
            recommendation_id,
            condition_id,
            tier,
            text,
        )


async def get_prewritten_fragment(condition_id: str, tier: str) -> str | None:
    rows = await db.query_raw(
        """
        SELECT text FROM offline_prewritten_fragments
        WHERE condition_id = $1 AND tier = $2
        ORDER BY created_at ASC LIMIT 1
        """,
        condition_id,
        tier,
    )
    return str(_field(rows[0], "text")) if rows else None


async def list_prewritten_condition_tiers(
    recommendation_id: str,
) -> set[tuple[str, str]]:
    rows = await db.query_raw(
        """
        SELECT condition_id, tier
        FROM offline_prewritten_fragments
        WHERE recommendation_id = $1
        """,
        recommendation_id,
    )
    return {
        (
            str(_field(row, "condition_id", "conditionId") or ""),
            str(_field(row, "tier") or ""),
        )
        for row in rows or []
    }


async def increment_activity_counter(
    recommendation_id: str, column: str
) -> int:
    """miss_count / hint_count 自增 1，返回新值。column 白名单固定，无注入面。"""
    if column not in {"miss_count", "hint_count"}:
        raise ValueError(f"illegal counter column: {column}")
    rows = await db.query_raw(
        f"""
        UPDATE offline_activity_recommendations
        SET {column} = {column} + 1, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        RETURNING {column} AS n
        """,
        recommendation_id,
    )
    return int(_field(rows[0], "n") or 0) if rows else 0


async def ai_memory_brief(
    user_id: str, workspace_id: str | None, *, limit: int = 40
) -> str:
    """AI 自我记忆摘要（memories_ai），用作思绪预生成的「AI 经历库」素材。"""
    rows = await db.query_raw(
        """
        SELECT content, main_category, sub_category
        FROM memories_ai
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    parts: list[str] = []
    for row in rows or []:
        text = str(_field(row, "content") or "").strip()
        if text:
            parts.append(f"- {text}")
    return "\n".join(parts)[:3000]


async def mark_conditions_ready(
    recommendation_id: str,
    user_id: str,
    *,
    focus_condition_id: str | None = None,
    next_companion_at: datetime | None = None,
) -> None:
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET conditions_ready_at = CURRENT_TIMESTAMP,
            focus_condition_id = COALESCE($3, focus_condition_id),
            next_companion_at = COALESCE($4::timestamptz, next_companion_at),
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || jsonb_build_object(
                    'mode', 'guided',
                    'unanswered_count', 0,
                    'recent_modes', '[]'::jsonb
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
          AND status = 'accepted' AND reached = TRUE
        """,
        recommendation_id,
        user_id,
        focus_condition_id,
        next_companion_at,
    )


async def list_unready_reached_activities(
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(workspace_id, user_id)
                       ORDER BY arrival_confirmed_at DESC NULLS LAST,
                                created_at DESC
                   ) AS current_rank
            FROM offline_activity_recommendations
            WHERE status = 'accepted' AND reached = TRUE
        )
        SELECT *
        FROM ranked
        WHERE current_rank = 1
          AND conditions_ready_at IS NULL
        ORDER BY arrival_confirmed_at ASC
        LIMIT $1
        """,
        limit,
    )
    return [activity_from_row(row, reveal_task=True) for row in rows or []]


async def list_ready_activities_with_missing_prewritten(
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT activity.*
        FROM offline_activity_recommendations activity
        WHERE activity.status = 'accepted'
          AND activity.reached = TRUE
          AND activity.conditions_ready_at IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM offline_shooting_conditions condition
              CROSS JOIN (VALUES ('rare'), ('epic'), ('legendary')) tier(name)
              WHERE condition.recommendation_id = activity.id
                AND NOT EXISTS (
                    SELECT 1
                    FROM offline_prewritten_fragments fragment
                    WHERE fragment.condition_id = condition.id
                      AND fragment.tier = tier.name
                )
          )
        ORDER BY activity.arrival_confirmed_at ASC
        LIMIT $1
        """,
        limit,
    )
    return [activity_from_row(row, reveal_task=True) for row in rows or []]


async def list_untriggered_conditions(recommendation_id: str) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT id, short_name, category, criteria, guidance_profile
        FROM offline_shooting_conditions
        WHERE recommendation_id = $1 AND triggered = FALSE
        ORDER BY sort_order ASC
        """,
        recommendation_id,
    )
    return [
        {
            "id": str(_field(r, "id")),
            "short_name": _field(r, "short_name", "shortName"),
            "category": _field(r, "category"),
            "criteria": _field(r, "criteria"),
            "guidance_profile": _json(
                _field(r, "guidance_profile", "guidanceProfile"), {}
            ),
        }
        for r in rows
    ]


async def list_all_conditions(recommendation_id: str) -> list[dict[str, Any]]:
    """全部拍摄物品（含已触发），供预生成回忆逐物品遍历。"""
    rows = await db.query_raw(
        """
        SELECT id, short_name, category, criteria, guidance_profile, triggered
        FROM offline_shooting_conditions
        WHERE recommendation_id = $1
        ORDER BY sort_order ASC
        """,
        recommendation_id,
    )
    return [
        {
            "id": str(_field(r, "id")),
            "short_name": _field(r, "short_name", "shortName"),
            "category": _field(r, "category"),
            "criteria": _field(r, "criteria"),
            "guidance_profile": _json(
                _field(r, "guidance_profile", "guidanceProfile"), {}
            ),
            "triggered": bool(_field(r, "triggered")),
        }
        for r in rows
    ]


async def admin_list_conditions(recommendation_id: str) -> list[dict[str, Any]]:
    """管理员检视用：拍摄物品全量（含判定要点 + 触发状态）。仅供 admin 测试页。"""
    rows = await db.query_raw(
        """
        SELECT short_name, category, criteria, guidance_profile, triggered, sort_order
        FROM offline_shooting_conditions
        WHERE recommendation_id = $1
        ORDER BY sort_order
        """,
        recommendation_id,
    )
    return [
        {
            "short_name": _field(r, "short_name", "shortName"),
            "category": _field(r, "category"),
            "criteria": _field(r, "criteria"),
            "guidance_profile": _json(
                _field(r, "guidance_profile", "guidanceProfile"), {}
            ),
            "triggered": bool(_field(r, "triggered")),
        }
        for r in rows
    ]


async def mark_condition_triggered(condition_id: str) -> bool:
    rows = await db.query_raw(
        """
        UPDATE offline_shooting_conditions AS condition
        SET triggered = TRUE, triggered_at = CURRENT_TIMESTAMP
        FROM offline_activity_recommendations AS activity
        WHERE condition.id = $1
          AND condition.triggered = FALSE
          AND activity.id = condition.recommendation_id
          AND activity.status = 'accepted'
          AND activity.reached = TRUE
        RETURNING condition.id
        """,
        condition_id,
    )
    return bool(rows)


async def set_activity_focus(
    recommendation_id: str,
    condition_id: str | None,
    *,
    reason: str,
) -> None:
    """Persist the hidden guidance focus; no target text is exposed."""
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET focus_condition_id = $2,
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || jsonb_build_object(
                    'last_focus_reason', $3::text,
                    'focus_changed_at', CURRENT_TIMESTAMP
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND status = 'accepted' AND reached = TRUE
          AND (
              $2::text IS NULL
              OR EXISTS (
                  SELECT 1
                  FROM offline_shooting_conditions condition
                  WHERE condition.id = $2
                    AND condition.recommendation_id = $1
                    AND condition.triggered = FALSE
              )
          )
        """,
        recommendation_id,
        condition_id,
        reason,
    )


async def reset_activity_misses(recommendation_id: str) -> None:
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET miss_count = 0, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND miss_count <> 0
        """,
        recommendation_id,
    )


async def touch_activity_interaction(
    recommendation_id: str,
    *,
    next_companion_at: datetime | None = None,
) -> None:
    """A user interaction resumes companionship and clears ignored-push backoff."""
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET next_companion_at = COALESCE($2::timestamptz, next_companion_at),
            companion_claim_token = NULL,
            companion_claimed_at = NULL,
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || jsonb_build_object(
                    'unanswered_count', 0,
                    'last_user_interaction_at', CURRENT_TIMESTAMP
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = 'accepted' AND reached = TRUE
        """,
        recommendation_id,
        next_companion_at,
    )


async def claim_due_companion_activities(
    *,
    limit: int = 20,
    lease_minutes: int = 15,
) -> list[dict[str, Any]]:
    """Atomically claim due active activities for the distributed scanner."""
    claim_token = uuid4().hex
    rows = await db.query_raw(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(workspace_id, user_id)
                       ORDER BY arrival_confirmed_at DESC NULLS LAST,
                                created_at DESC
                   ) AS rn
            FROM offline_activity_recommendations
            WHERE status = 'accepted' AND reached = TRUE
        ),
        due AS (
            SELECT id
            FROM offline_activity_recommendations
            WHERE status = 'accepted'
              AND reached = TRUE
              AND conditions_ready_at IS NOT NULL
              AND next_companion_at IS NOT NULL
              AND next_companion_at <= CURRENT_TIMESTAMP
              AND id IN (SELECT id FROM ranked WHERE rn = 1)
            ORDER BY next_companion_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE offline_activity_recommendations a
        SET next_companion_at =
                CURRENT_TIMESTAMP + ($2::int * INTERVAL '1 minute'),
            companion_claim_token = $3,
            companion_claimed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        FROM due
        WHERE a.id = due.id
        RETURNING a.*
        """,
        limit,
        lease_minutes,
        claim_token,
    )
    return [activity_from_row(row, reveal_task=True) for row in rows or []]


async def save_companion_decision(
    recommendation_id: str,
    *,
    claim_token: str,
    state: dict[str, Any],
    next_companion_at: datetime | None,
    sent: bool,
) -> bool:
    rows = await db.query_raw(
        """
        UPDATE offline_activity_recommendations
        SET companion_state = $2::jsonb,
            next_companion_at = $3::timestamptz,
            last_companion_at = CASE
                WHEN $4::boolean THEN CURRENT_TIMESTAMP
                ELSE last_companion_at
            END,
            companion_claim_token = NULL,
            companion_claimed_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND status = 'accepted' AND reached = TRUE
          AND companion_claim_token = $5
        RETURNING id
        """,
        recommendation_id,
        json.dumps(state, ensure_ascii=False),
        next_companion_at,
        sent,
        claim_token,
    )
    return bool(rows)


async def reserve_companion_send(
    recommendation_id: str,
    *,
    claim_token: str,
    delivery_key: str,
    state: dict[str, Any],
    next_companion_at: datetime,
) -> bool:
    """Fence the claim before delivery; a crash may skip one turn, never duplicate it."""
    state = {**state, "delivery_key": delivery_key}
    rows = await db.query_raw(
        """
        UPDATE offline_activity_recommendations
        SET companion_state = $3::jsonb
                || jsonb_build_object(
                    'delivery_reserved_at', CURRENT_TIMESTAMP
                ),
            next_companion_at = $4::timestamptz,
            last_companion_at = CURRENT_TIMESTAMP,
            companion_claim_token = NULL,
            companion_claimed_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND status = 'accepted' AND reached = TRUE
          AND companion_claim_token = $2
        RETURNING id
        """,
        recommendation_id,
        claim_token,
        json.dumps(state, ensure_ascii=False),
        next_companion_at,
    )
    return bool(rows)


async def companion_message_exists(delivery_key: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT 1
        FROM messages
        WHERE metadata->>'offline_companion_delivery_key' = $1
        LIMIT 1
        """,
        delivery_key,
    )
    return bool(rows)


async def cancel_guarded_companion_delivery(
    recommendation_id: str,
    *,
    delivery_key: str,
    previous_last_companion_at: datetime | str | None,
) -> None:
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET last_companion_at = $3::timestamptz,
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || jsonb_build_object(
                    'delivery_canceled', true,
                    'unanswered_count',
                    GREATEST(
                        COALESCE(
                            NULLIF(companion_state->>'unanswered_count', '')::int,
                            0
                        ) - 1,
                        0
                    )
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND status = 'accepted' AND reached = TRUE
          AND companion_state->>'delivery_key' = $2
        """,
        recommendation_id,
        delivery_key,
        previous_last_companion_at,
    )


async def reschedule_failed_companion_delivery(
    recommendation_id: str,
    *,
    delivery_key: str,
    next_companion_at: datetime,
    previous_last_companion_at: datetime | str | None,
) -> None:
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET next_companion_at = $3::timestamptz,
            last_companion_at = $4::timestamptz,
            companion_state = COALESCE(companion_state, '{}'::jsonb)
                || jsonb_build_object(
                    'delivery_failed', true,
                    'unanswered_count',
                    GREATEST(
                        COALESCE(
                            NULLIF(companion_state->>'unanswered_count', '')::int,
                            0
                        ) - 1,
                        0
                    )
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
          AND status = 'accepted' AND reached = TRUE
          AND companion_state->>'delivery_key' = $2
        """,
        recommendation_id,
        delivery_key,
        next_companion_at,
        previous_last_companion_at,
    )


async def get_current_reached_activity(
    user_id: str,
    workspace_id: str | None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND status = 'accepted'
          AND reached = TRUE
        ORDER BY arrival_confirmed_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        user_id,
        workspace_id,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def find_other_reached_activity(
    user_id: str,
    workspace_id: str | None,
    *,
    exclude_id: str,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND id <> $3
          AND status = 'accepted'
          AND reached = TRUE
        ORDER BY arrival_confirmed_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        user_id,
        workspace_id,
        exclude_id,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


# ---------------------------------------------------------------------------
# 思绪碎片（spec §4）
# ---------------------------------------------------------------------------
async def count_fragments(recommendation_id: str) -> int:
    rows = await db.query_raw(
        "SELECT COUNT(*)::int AS n FROM offline_thought_fragments "
        "WHERE recommendation_id = $1",
        recommendation_id,
    )
    return int(_field(rows[0], "n") or 0) if rows else 0


async def fragment_fingerprint_exists(
    recommendation_id: str, fingerprint: str
) -> bool:
    if not fingerprint:
        return False
    rows = await db.query_raw(
        """
        SELECT 1 FROM offline_thought_fragments
        WHERE recommendation_id = $1 AND content_fingerprint = $2 LIMIT 1
        """,
        recommendation_id,
        fingerprint,
    )
    return bool(rows)


async def create_fragment(
    *,
    recommendation_id: str,
    tier: str,
    text: str,
    lead_in: str | None = None,
    condition_id: str | None = None,
    snapshot_media_id: str | None = None,
    source_message_id: str | None = None,
    content_fingerprint: str | None = None,
) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        INSERT INTO offline_thought_fragments
            (id, recommendation_id, tier, text, lead_in, condition_id,
             snapshot_media_id, source_message_id, content_fingerprint)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id, tier, text, lead_in, snapshot_media_id, created_at
        """,
        new_id(),
        recommendation_id,
        tier,
        text,
        lead_in,
        condition_id,
        snapshot_media_id,
        source_message_id,
        content_fingerprint,
    )
    return _fragment_from_row(rows[0]) if rows else {}


async def list_fragments(recommendation_id: str) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT id, tier, text, lead_in, snapshot_media_id, created_at
        FROM offline_thought_fragments
        WHERE recommendation_id = $1
        ORDER BY created_at ASC
        """,
        recommendation_id,
    )
    return [_fragment_from_row(r) for r in rows]


def _fragment_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": str(_field(row, "id")),
        "tier": str(_field(row, "tier") or "rare"),
        "text": str(_field(row, "text") or ""),
        "lead_in": _field(row, "lead_in", "leadIn"),
        "snapshot_media_id": _field(row, "snapshot_media_id", "snapshotMediaId"),
        "created_at": _iso(_field(row, "created_at", "createdAt")),
    }


async def create_captured_media(
    *,
    recommendation_id: str,
    user_id: str,
    storage_key: str,
    url: str,
    mime: str,
    size: int,
    width: int | None,
    height: int | None,
    source_message_id: str | None,
) -> str:
    """把聊天发来的现场图片登记为活动素材（role=material，url 指向聊天媒体服务）。"""
    media_id = new_id()
    await db.execute_raw(
        """
        INSERT INTO offline_activity_media
            (id, recommendation_id, user_id, kind, mime, size, width, height,
             storage_key, url, role, source_message_id)
        VALUES ($1, $2, $3, 'image', $4, $5, $6, $7, $8, $9, 'material', $10)
        """,
        media_id,
        recommendation_id,
        user_id,
        mime,
        int(size or 0),
        width,
        height,
        storage_key,
        url,
        source_message_id,
    )
    return media_id


async def mark_message_recognized(message_id: str | None, tier: str) -> None:
    """把识图命中标记持久化到源消息 metadata，使前端重载聊天记录后金框/标记/说明仍在
    （否则该标记只存在于实时 WS 的内存态，重载即丢）。metadata 用 jsonb 合并，不覆盖其他键。"""
    if not message_id:
        return
    await db.execute_raw(
        """
        UPDATE messages
        SET metadata = COALESCE(metadata, '{}'::jsonb)
                       || jsonb_build_object(
                              'offline_recognized', true,
                              'offline_fragment_tier', $2::text
                          )
        WHERE id = $1
        """,
        message_id,
        tier,
    )


async def claim_activity_followup(message_id: str | None) -> bool:
    """Allow at most one offline fragment/follow-up per user message."""
    if not message_id:
        return True
    rows = await db.query_raw(
        """
        UPDATE messages
        SET metadata = COALESCE(metadata, '{}'::jsonb)
            || '{"offline_activity_followup_claimed":true}'::jsonb
        WHERE id = $1
          AND COALESCE(metadata->>'offline_activity_followup_claimed', 'false')
              <> 'true'
        RETURNING id
        """,
        message_id,
    )
    return bool(rows)


async def find_arrival_card_message_id(
    activity_id: str, conversation_id: str | None
) -> str | None:
    """定位该活动「我到了」到达卡消息（供回顾页「查看原始聊天」跳转定位）。"""
    if not conversation_id:
        return None
    rows = await db.query_raw(
        """
        SELECT id FROM messages
        WHERE conversation_id = $1
          AND metadata->>'source_id' = $2
          AND metadata->>'trigger_type' = 'offline_activity_arrived_card'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        conversation_id,
        activity_id,
    )
    return str(_field(rows[0], "id")) if rows else None


async def mark_media_fragment_cover(media_id: str, fingerprint: str) -> None:
    """碎片封面图：从画廊素材里剔除（spec §3.5），并记内容指纹与已识别。"""
    await db.execute_raw(
        """
        UPDATE offline_activity_media
        SET role = 'fragment_cover', recognized = TRUE,
            content_fingerprint = $2, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """,
        media_id,
        fingerprint,
    )


async def list_gallery_media(recommendation_id: str) -> list[str]:
    """回顾画廊：仅 role=material 的图片；已作碎片封面的（fragment_cover）自动排除。"""
    rows = await db.query_raw(
        """
        SELECT url FROM offline_activity_media
        WHERE recommendation_id = $1 AND kind = 'image' AND role = 'material'
        ORDER BY created_at ASC
        """,
        recommendation_id,
    )
    return [str(_field(r, "url")) for r in rows if _field(r, "url")]


async def list_voice_transcripts(recommendation_id: str) -> list[str]:
    """活动期间用户语音转写文本（chat_capture 归档为 kind=voice_transcript）。"""
    rows = await db.query_raw(
        """
        SELECT text FROM offline_activity_feedback
        WHERE recommendation_id = $1 AND kind = 'voice_transcript'
        ORDER BY created_at ASC
        """,
        recommendation_id,
    )
    return [str(_field(r, "text")).strip() for r in rows if str(_field(r, "text")).strip()]


async def set_travel_note(
    recommendation_id: str, user_id: str, text: str
) -> None:
    await db.execute_raw(
        """
        UPDATE offline_activity_recommendations
        SET travel_note = $3, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        """,
        recommendation_id,
        user_id,
        text,
    )


async def create_activity_feedback(
    *,
    recommendation_id: str,
    user_id: str,
    kind: str,
    text: str = "",
    photo_attachment_ids: list[str] | None = None,
    audio_attachment_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    await db.execute_raw(
        """
        INSERT INTO offline_activity_feedback (
            id, recommendation_id, user_id, kind, text,
            photo_attachment_ids, audio_attachment_id, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb)
        """,
        new_id(),
        recommendation_id,
        user_id,
        kind,
        text,
        json.dumps(photo_attachment_ids or [], ensure_ascii=False),
        audio_attachment_id,
        json.dumps(metadata or {}, ensure_ascii=False),
    )

async def get_activity_completion_feedback(
    *,
    recommendation_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_feedback
        WHERE recommendation_id = $1
          AND user_id = $2
          AND kind = 'completion'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        recommendation_id,
        user_id,
    )
    if not rows:
        return None
    row = rows[0]
    attachment_ids = [
        str(item)
        for item in _json(
            _field(row, "photo_attachment_ids", "photoAttachmentIds"),
            [],
        )
        if str(item).strip()
    ][:3]
    audio_attachment_id = str(
        _field(row, "audio_attachment_id", "audioAttachmentId") or ""
    ).strip()
    media_ids = attachment_ids + ([audio_attachment_id] if audio_attachment_id else [])
    attachments: list[dict[str, Any]] = []
    audio_attachment: dict[str, Any] | None = None
    if media_ids:
        attachment_rows = await db.query_raw(
            """
            SELECT id, kind, name, mime, size, width, height, duration_seconds, url,
                   created_at
            FROM offline_activity_media
            WHERE id = ANY($1::text[])
              AND user_id = $2
              AND recommendation_id = $3
            """,
            media_ids,
            user_id,
            recommendation_id,
        )
        by_id = {str(_field(item, "id")): item for item in attachment_rows or []}
        for attachment_id in media_ids:
            item = by_id.get(attachment_id)
            if not item:
                continue
            payload = {
                "id": str(_field(item, "id")),
                "kind": str(_field(item, "kind") or "image"),
                "name": _field(item, "name"),
                "mime": str(_field(item, "mime") or "image/jpeg"),
                "size": int(_field(item, "size") or 0),
                "width": _field(item, "width"),
                "height": _field(item, "height"),
                "duration_seconds": _field(
                    item, "duration_seconds", "durationSeconds"
                ),
                "url": str(_field(item, "url") or ""),
                "vision_status": "ready",
                "vision_summary": None,
                "created_at": _iso(_field(item, "created_at", "createdAt")),
            }
            if attachment_id == audio_attachment_id:
                audio_attachment = payload
            else:
                attachments.append(payload)
    return {
        "text": str(_field(row, "text") or ""),
        "photo_attachments": attachments,
        "audio_attachment": audio_attachment,
        "created_at": _iso(_field(row, "created_at", "createdAt")),
    }


def next_activity_due(base: datetime | None = None, *, accepted_delta_days: int = 0) -> datetime:
    import random

    start = base or now_utc()
    days = random.randint(20, 40) + accepted_delta_days
    return start + timedelta(days=max(8, min(days, 70)))


async def update_next_activity_due(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    due_at: datetime,
) -> None:
    await db.execute_raw(
        """
        INSERT INTO real_world_trigger_states (
            id, user_id, agent_id, workspace_id, next_activity_recommendation_at,
            last_activity_recommendation_at
        )
        VALUES ($1, $2, $3, $4, $5::timestamptz, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, agent_id) DO UPDATE
        SET workspace_id = COALESCE(EXCLUDED.workspace_id, real_world_trigger_states.workspace_id),
            next_activity_recommendation_at = EXCLUDED.next_activity_recommendation_at,
            last_activity_recommendation_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        """,
        new_id(),
        user_id,
        agent_id,
        workspace_id,
        due_at,
    )
