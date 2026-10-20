"""Bounded admin reads; no generation, expiration, messages or state changes."""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote, urlsplit

from fastapi import HTTPException

from app.db import db
from app.models.admin_activities import AdminActivityDetail
from app.models.offline import OfflineActivityItem
from app.services.offline import repository as mobile_repo
from app.services.offline.gift_admin_repository import get_user

_STAGE = "CASE WHEN a.status='accepted' THEN CASE WHEN a.reached THEN 'in_progress' ELSE 'planned' END ELSE a.status END"
_SCOPE = """FROM offline_activity_recommendations a
    JOIN users u ON u.id=a.user_id
    LEFT JOIN ai_agents ag ON ag.id=a.agent_id AND ag.user_id=a.user_id"""
_COUNTS = """(SELECT COUNT(*)::int FROM offline_shooting_conditions t WHERE t.recommendation_id=a.id) AS task_count,
    (SELECT COUNT(*)::int FROM offline_activity_media t WHERE t.recommendation_id=a.id AND t.user_id=a.user_id) AS media_count,
    (SELECT COUNT(*)::int FROM offline_thought_fragments t WHERE t.recommendation_id=a.id) AS actual_fragment_count"""
_CONVERSATION = """c.user_id=a.user_id AND c.agent_id=a.agent_id
    AND (c.workspace_id IS NOT DISTINCT FROM a.workspace_id
      OR (a.workspace_id IS NULL AND c.id=a.conversation_id AND EXISTS(
        SELECT 1 FROM chat_workspaces w WHERE w.id=c.workspace_id AND w.user_id=a.user_id AND w.agent_id=a.agent_id)))
    AND c.is_deleted=FALSE"""
_LINKED = """(
    (m.metadata->>'real_world_type'='activity' AND m.metadata->>'source_id'=a.id)
    OR (m.metadata#>>'{component_card,type}'='offline_activity' AND m.metadata#>>'{component_card,payload,activity_id}'=a.id)
    OR EXISTS(SELECT 1 FROM offline_activity_media x WHERE x.recommendation_id=a.id AND x.user_id=a.user_id AND x.source_message_id=m.id)
    OR EXISTS(SELECT 1 FROM offline_thought_fragments x WHERE x.recommendation_id=a.id AND x.source_message_id=m.id)
    OR EXISTS(SELECT 1 FROM offline_activity_feedback x WHERE x.recommendation_id=a.id AND x.user_id=a.user_id AND x.metadata->>'source_message_id'=m.id)
)"""
_JOURNEY = """(c.id=a.conversation_id AND a.arrival_confirmed_at IS NOT NULL
    AND m.created_at>=a.arrival_confirmed_at
    AND m.created_at<=COALESCE(a.archived_at,a.completed_at,CURRENT_TIMESTAMP)
    AND (m.metadata->>'source_id' IS NULL OR m.metadata->>'source_id'=a.id))"""


def sampled_at() -> str:
    return datetime.now(UTC).isoformat()


async def user_context(user_id: str) -> dict:
    user = await get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    workspaces = await db.query_raw(
        """
        SELECT w.id, w.agent_id, w.status, ag.name AS agent_name
        FROM chat_workspaces w LEFT JOIN ai_agents ag ON ag.id=w.agent_id AND ag.user_id=w.user_id
        WHERE w.user_id=$1 ORDER BY w.updated_at DESC,w.id LIMIT 101
    """,
        user_id,
    )
    return {
        "user": user,
        "workspaces": workspaces[:100],
        "workspaces_truncated": len(workspaces) > 100,
        "sampled_at": sampled_at(),
    }


async def _workspace(user_id: str, workspace_id: str | None) -> None:
    if not await db.query_raw(
        """SELECT u.id FROM users u WHERE u.id=$1 AND ($2::text IS NULL OR EXISTS(
            SELECT 1 FROM chat_workspaces w WHERE w.id=$2 AND w.user_id=u.id)) LIMIT 1""",
        user_id,
        workspace_id,
    ):
        raise HTTPException(404, "Workspace not found")


def _counts(row: dict) -> dict:
    return {
        "tasks": row["task_count"],
        "media": row["media_count"],
        "fragments": row["actual_fragment_count"],
    }


async def list_activities(
    user_id: str | None,
    workspace_id: str | None,
    status: str,
    q: str,
    limit: int,
    offset: int,
) -> dict:
    if user_id is not None:
        await _workspace(user_id, workspace_id)
    elif workspace_id is not None:
        raise HTTPException(422, "workspace_id requires user_id")
    # Literal query, not SQL LIKE wildcard syntax. All values remain parameters.
    pattern = (
        "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        if q.strip()
        else None
    )
    where = f"""WHERE ($1::text IS NULL OR a.user_id=$1) AND ($2::text IS NULL OR a.workspace_id=$2)
        AND ($3='all' OR {_STAGE}=$3 OR ($3='closed' AND a.status IN ('ignored','cancelled','expired')))
        AND ($4::text IS NULL OR a.title ILIKE $4 OR a.location_name ILIKE $4 OR a.id=$5)"""
    args = (user_id, workspace_id, status, pattern, q)
    rows = await db.query_raw(
        f"""SELECT a.id,a.user_id,u.username,
        COALESCE((SELECT ai.raw_profile->>'nickname' FROM auth_identities ai
            WHERE ai.user_id=a.user_id AND ai.provider='wechat'
            ORDER BY ai.updated_at DESC,ai.id DESC LIMIT 1),NULLIF(u.display_name,'')) AS nickname,
        a.agent_id,ag.name AS agent_name,a.workspace_id,
        a.title,a.summary,a.status,a.reached,a.location_name,a.created_at,a.updated_at,
        {_STAGE} AS stage,{_COUNTS} {_SCOPE} {where}
        ORDER BY a.created_at DESC,a.id DESC LIMIT $6 OFFSET $7""",
        *args,
        limit,
        offset,
    )
    total = (
        await db.query_raw(f"SELECT COUNT(*)::int AS total {_SCOPE} {where}", *args)
    )[0]["total"]
    items = [
        {
            **r,
            "fragment_count": r["actual_fragment_count"],
            "title": mobile_repo.plain_text(r["title"]),
            "summary": mobile_repo.plain_text(r["summary"]),
            "created_at": mobile_repo._iso(r["created_at"]),
            "updated_at": mobile_repo._iso(r["updated_at"]),
        }
        for r in rows
    ]
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "sampled_at": sampled_at(),
    }


async def activity_row(
    activity_id: str, user_id: str, workspace_id: str | None = None
) -> dict:
    rows = await db.query_raw(
        f"""SELECT a.*,ag.name AS agent_name,{_STAGE} AS stage,{_COUNTS},
        EXISTS(SELECT 1 FROM conversations c WHERE c.id=a.conversation_id AND {_CONVERSATION}) AS conversation_available
        {_SCOPE} WHERE a.id=$1 AND a.user_id=$2 AND ($3::text IS NULL OR a.workspace_id=$3) LIMIT 1
    """,
        activity_id,
        user_id,
        workspace_id,
    )
    if not rows:
        raise HTTPException(404, "Activity not found")
    return rows[0]


async def detail(
    activity_id: str, user_id: str, workspace_id: str | None = None
) -> AdminActivityDetail:
    r = await activity_row(activity_id, user_id, workspace_id)
    activity = mobile_repo.activity_from_row(r, reveal_task=True)
    if r["status"] == "completed":
        activity["completion_feedback"] = (
            await mobile_repo.get_activity_completion_feedback(
                recommendation_id=activity_id, user_id=user_id
            )
        )
    generation = (
        ("ready" if r["task_count"] else "ready_empty")
        if r.get("conditions_ready_at")
        else (
            "partial"
            if r["task_count"]
            else (
                "pending"
                if r["status"] == "accepted" and r["reached"]
                else "not_generated"
            )
        )
    )
    return AdminActivityDetail(
        activity=OfflineActivityItem(**activity),
        user_id=user_id,
        agent_name=r.get("agent_name"),
        workspace_id=r.get("workspace_id"),
        conversation_id=(
            r.get("conversation_id") if r["conversation_available"] else None
        ),
        conversation_available=r["conversation_available"],
        stage=r["stage"],
        source=r["source"],
        conditions_ready_at=activity["conditions_ready_at"],
        task_generation=generation,
        archived_at=mobile_repo._iso(r.get("archived_at")),
        auto_archived=bool(r.get("auto_archived")),
        travel_note=r.get("travel_note"),
        counts=_counts(r),
        copy_quality={
            key: value if isinstance(value, str) else None
            for key in (
                "card_copy_status",
                "copy_status",
                "recommendation_message_status",
                "recommendation_message_length_status",
                "context_source_status",
            )
            for value in [activity["discovery_metadata"].get(key)]
        },
        sampled_at=sampled_at(),
    )


def asset_url(activity_id: str, user_id: str, resource: str) -> str:
    return f"/admin-api/activities/{quote(activity_id, safe='')}/{resource}?user_id={quote(user_id, safe='')}"


def _json(value, default):
    return mobile_repo._json(value, default)


def _iso_rows(rows: list[dict]) -> list[dict]:
    return [
        {k: mobile_repo._iso(v) if isinstance(v, datetime) else v for k, v in r.items()}
        for r in rows
    ]


def _message_from(extra_join: str = "") -> str:
    return f"""FROM offline_activity_recommendations a JOIN users u ON u.id=a.user_id
        JOIN conversations c ON {_CONVERSATION} JOIN messages m ON m.conversation_id=c.id {extra_join}
        WHERE a.id=$1 AND a.user_id=$2 AND m.role IN ('user','assistant')
        AND ({_LINKED} OR ($3='journey' AND {_JOURNEY}))"""


async def related(
    activity_id: str,
    user_id: str,
    section: str,
    mode: str,
    limit: int,
    offset: int,
    workspace_id: str | None = None,
) -> dict:
    activity = await activity_row(activity_id, user_id, workspace_id)
    if section == "messages":
        clause = _message_from()
        args = (activity_id, user_id, mode)
        rows = await db.query_raw(
            f"""SELECT m.id,m.conversation_id,m.role,m.content,m.created_at,
            m.metadata->>'trigger_type' AS trigger_type,(m.metadata->>'trace_id' IS NOT NULL) AS has_trace,
            m.metadata->'component_card' AS component_card,({_LINKED}) AS directly_linked
            {clause} ORDER BY m.created_at,m.id LIMIT $4 OFFSET $5""",
            *args,
            limit,
            offset,
        )
        total = (await db.query_raw(f"SELECT COUNT(*)::int AS total {clause}", *args))[
            0
        ]["total"]
        await _message_assets(rows, activity_id, user_id)
    else:
        # Table/column choices are constants, never interpolated request values.
        specs = {
            "tasks": (
                "offline_shooting_conditions",
                "t.id,t.short_name,t.category,t.criteria,t.triggered,t.triggered_at,t.created_at",
                "t.sort_order,t.id",
            ),
            "media": (
                "offline_activity_media",
                "t.id,t.kind,t.name,t.mime,t.size,t.width,t.height,t.duration_seconds,t.url,t.role,t.recognized,t.transcript,t.created_at",
                "t.created_at,t.id",
            ),
            "fragments": (
                "offline_thought_fragments",
                "t.id,t.tier,t.text,t.lead_in,t.condition_id,t.created_at",
                "t.created_at,t.id",
            ),
        }
        table, columns, order = specs[section]
        owner = " AND t.user_id=$2" if section == "media" else ""
        clause = f"FROM {table} t WHERE t.recommendation_id=$1{owner}"
        args = (activity_id, user_id) if section == "media" else (activity_id,)
        rows = await db.query_raw(
            f"SELECT {columns} {clause} ORDER BY {order} LIMIT ${len(args)+1} OFFSET ${len(args)+2}",
            *args,
            limit,
            offset,
        )
        total = (await db.query_raw(f"SELECT COUNT(*)::int AS total {clause}", *args))[
            0
        ]["total"]
        for row in rows:
            if section == "tasks":
                row["focused"] = activity.get("focus_condition_id") == row["id"]
                row["state"] = (
                    "triggered"
                    if row["triggered"]
                    else (
                        "pending" if activity["status"] == "accepted" else "untriggered"
                    )
                )
            elif section == "media":
                row["url"] = _media_url(
                    row.get("url"),
                    asset_url(activity_id, user_id, f"media/{row['id']}/content"),
                )
    return {
        "section": section,
        "mode": mode,
        "items": _iso_rows(rows),
        "total": total,
        "limit": limit,
        "offset": offset,
        "sampled_at": sampled_at(),
    }


def _media_url(original: str | None, gateway: str) -> str | None:
    if not original:
        return None
    if str(original).startswith(("/chat/media/", "/offline/media/")):
        return gateway
    try:
        parsed = urlsplit(original)
        return (
            original
            if parsed.scheme in {"https", "http"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            else None
        )
    except ValueError:
        return None


async def _message_assets(rows: list[dict], activity_id: str, user_id: str) -> None:
    attachments = (
        await db.query_raw(
            """SELECT att.id,att.message_id,att.kind,att.name,att.mime,att.size,att.width,att.height,att.duration_seconds,att.transcription_text
        FROM chat_message_attachments att JOIN messages m ON m.id=att.message_id AND m.conversation_id=att.conversation_id
        WHERE att.message_id=ANY($1::text[]) AND att.user_id=$2
        ORDER BY att.created_at,att.id LIMIT 500""",
            [r["id"] for r in rows],
            user_id,
        )
        if rows
        else []
    )
    for r in rows:
        card = _json(r.get("component_card"), {})
        payload = card.get("payload") if isinstance(card, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        r["component_card"] = (
            {
                k: card.get(k)
                for k in ("version", "type", "title", "subtitle", "body", "footer")
            }
            if (
                isinstance(card, dict)
                and card.get("type") == "offline_activity"
                and payload.get("activity_id") == activity_id
            )
            else None
        )
        if r["component_card"] is not None:
            urls = payload.get("image_urls") or (
                [payload["image_url"]] if payload.get("image_url") else []
            )
            r["component_card"]["image_urls"] = (
                [
                    url
                    for i, original in enumerate(urls[:20])
                    if isinstance(original, str)
                    and (
                        url := _media_url(
                            original,
                            asset_url(
                                activity_id, user_id, f"messages/{r['id']}/images/{i}"
                            ),
                        )
                    )
                ]
                if isinstance(urls, list)
                else []
            )
        r["attachments"] = [
            {
                **a,
                "url": asset_url(
                    activity_id, user_id, f"messages/{r['id']}/attachments/{a['id']}"
                ),
            }
            for a in attachments
            if a["message_id"] == r["id"]
        ]
        # Do not return trace IDs or arbitrary message metadata/tokens.


async def media_content(activity_id: str, user_id: str, media_id: str):
    rows = await db.query_raw(
        """SELECT t.storage_key,t.url FROM offline_activity_media t
        JOIN offline_activity_recommendations a ON a.id=t.recommendation_id AND a.user_id=t.user_id
        JOIN users u ON u.id=a.user_id
        WHERE t.id=$1 AND a.id=$2 AND a.user_id=$3 LIMIT 1""",
        media_id,
        activity_id,
        user_id,
    )
    if not rows:
        raise HTTPException(404, "Media not found")
    return _serve(rows[0]["url"], rows[0]["storage_key"], user_id)


async def attachment_content(
    activity_id: str, user_id: str, message_id: str, attachment_id: str
):
    joins = "JOIN chat_message_attachments att ON att.message_id=m.id AND att.user_id=a.user_id AND att.conversation_id=c.id"
    rows = await db.query_raw(
        f"""SELECT att.storage_key,att.url {_message_from(joins)}
        AND m.id=$4 AND att.id=$5 LIMIT 1""",
        activity_id,
        user_id,
        "journey",
        message_id,
        attachment_id,
    )
    if not rows:
        raise HTTPException(404, "Attachment not found")
    return _serve(rows[0]["url"], rows[0]["storage_key"], user_id)


async def recommendation_image(activity_id: str, user_id: str, index: int):
    rows = await db.query_raw(
        """SELECT a.image_urls FROM offline_activity_recommendations a
        JOIN users u ON u.id=a.user_id WHERE a.id=$1 AND a.user_id=$2 LIMIT 1""",
        activity_id,
        user_id,
    )
    if not rows:
        raise HTTPException(404, "Activity not found")
    r = rows[0]
    urls = _json(r.get("image_urls"), [])
    if not 0 <= index < len(urls):
        raise HTTPException(404, "Image not found")
    url = str(urls[index])
    return _serve(url, None, user_id)


async def message_image(activity_id: str, user_id: str, message_id: str, index: int):
    rows = await db.query_raw(
        f"SELECT m.metadata->'component_card' AS card {_message_from()} AND m.id=$4 LIMIT 1",
        activity_id,
        user_id,
        "linked",
        message_id,
    )
    card = _json(rows[0]["card"], {}) if rows else {}
    payload = card.get("payload") if isinstance(card, dict) else None
    if not isinstance(payload, dict) or payload.get("activity_id") != activity_id:
        raise HTTPException(404, "Image not found")
    urls = payload.get("image_urls") or (
        [payload["image_url"]] if payload.get("image_url") else []
    )
    if not isinstance(urls, list) or not 0 <= index < len(urls):
        raise HTTPException(404, "Image not found")
    url = str(urls[index])
    return _serve(url, None, user_id)


def _serve(url: str, storage_key: str | None, user_id: str):
    from app.services.chat_media import storage as chat_storage
    from app.services.offline import activity_media_storage

    try:
        parsed = urlsplit(url)
    except ValueError:
        raise HTTPException(404, "Media not available") from None
    if parsed.scheme or parsed.netloc:
        raise HTTPException(404, "Media not available")
    path = parsed.path
    storage_key = storage_key or path.rsplit("/", 1)[-1]
    if path == f"/chat/media/{storage_key}":
        response = chat_storage.serve_media(
            storage_key, user_id=user_id, is_admin=False
        )
    elif path == f"/offline/media/{storage_key}":
        response = activity_media_storage.serve_media(
            storage_key, user_id=user_id, is_admin=False
        )
    else:
        raise HTTPException(404, "Media not available")
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
