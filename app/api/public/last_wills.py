from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.api.jwt_auth import require_user
from app.db import db
from app.models.last_will import (
    LastWillContact,
    LastWillCreate,
    LastWillDelivery,
    LastWillResponse,
    LastWillUpdate,
)
from app.services.last_will_crypto import (
    protect_contact,
    protect_text,
    reveal_contact,
    reveal_text,
)

router = APIRouter(prefix="/last-wills", tags=["last-wills"])

_VALID_STATUSES = {"draft", "active", "paused", "triggered", "cancelled"}


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _json_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    data = getattr(value, "data", value)
    return data if isinstance(data, list) else []


def _contacts(value: Any) -> list[LastWillContact]:
    contacts: list[LastWillContact] = []
    for item in _json_list(value)[:3]:
        try:
            contacts.append(LastWillContact.model_validate(reveal_contact(item)))
        except Exception:
            continue
    return contacts


def _contact_payload(contacts: list[LastWillContact] | None) -> str:
    data = [protect_contact(contact.model_dump()) for contact in (contacts or [])]
    return json.dumps(data, ensure_ascii=False)


def _response(row: Any) -> LastWillResponse:
    return LastWillResponse(
        id=_field(row, "id"),
        user_id=_field(row, "userId"),
        agent_id=_field(row, "agentId"),
        workspace_id=_field(row, "workspaceId"),
        content=reveal_text(_field(row, "content")),
        inactivity_days=int(_field(row, "inactivityDays") or 30),
        contacts=_contacts(_field(row, "contacts")),
        status=_field(row, "status") or "draft",
        last_seen_at=_iso(_field(row, "lastSeenAt")),
        started_at=_iso(_field(row, "startedAt")),
        triggered_at=_iso(_field(row, "triggeredAt")),
        delivered_at=_iso(_field(row, "deliveredAt")),
        created_at=_iso(_field(row, "createdAt")) or "",
        updated_at=_iso(_field(row, "updatedAt")) or "",
    )


async def _ensure_agent_scope(
    *,
    agent_id: str,
    workspace_id: str | None,
    user: dict,
) -> None:
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    if user.get("role") != "admin" and agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    if workspace_id:
        workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
        if (
            not workspace
            or workspace.userId != agent.userId
            or workspace.agentId != agent_id
        ):
            raise HTTPException(status_code=400, detail="Workspace does not match agent")


def _normalize_status(
    status: str | None,
    contacts: list[LastWillContact] | None,
    content: str | None = None,
) -> str:
    normalized = (status or "draft").strip().lower()
    if normalized not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid last will status")
    if normalized == "active" and not contacts:
        raise HTTPException(status_code=400, detail="请至少添加 1 个联系人后再开始触发")
    if normalized == "active" and not (content or "").strip():
        raise HTTPException(status_code=400, detail="请先写下遗言内容后再开始触发")
    return normalized


async def _fetch_will(will_id: str) -> Any | None:
    rows = await db.query_raw(
        """
        SELECT
            lw.id,
            lw.user_id AS "userId",
            lw.agent_id AS "agentId",
            lw.workspace_id AS "workspaceId",
            lw.content,
            lw.inactivity_days AS "inactivityDays",
            lw.contacts,
            lw.status,
            u.last_seen_at AS "lastSeenAt",
            lw.started_at AS "startedAt",
            lw.triggered_at AS "triggeredAt",
            lw.delivered_at AS "deliveredAt",
            lw.created_at AS "createdAt",
            lw.updated_at AS "updatedAt"
        FROM last_wills lw
        JOIN users u ON u.id = lw.user_id
        WHERE lw.id = $1
        LIMIT 1
        """,
        will_id,
    )
    return rows[0] if rows else None


async def _require_will_owner(will_id: str, user: dict) -> Any:
    row = await _fetch_will(will_id)
    if not row:
        raise HTTPException(status_code=404, detail="Last will not found")
    if user.get("role") != "admin" and _field(row, "userId") != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your last will")
    return row


@router.get("", response_model=list[LastWillResponse])
async def list_last_wills(
    agent_id: str = Query(...),
    workspace_id: str | None = None,
    user: dict = Depends(require_user),
):
    await _ensure_agent_scope(agent_id=agent_id, workspace_id=workspace_id, user=user)
    rows = await db.query_raw(
        """
        SELECT
            lw.id,
            lw.user_id AS "userId",
            lw.agent_id AS "agentId",
            lw.workspace_id AS "workspaceId",
            lw.content,
            lw.inactivity_days AS "inactivityDays",
            lw.contacts,
            lw.status,
            u.last_seen_at AS "lastSeenAt",
            lw.started_at AS "startedAt",
            lw.triggered_at AS "triggeredAt",
            lw.delivered_at AS "deliveredAt",
            lw.created_at AS "createdAt",
            lw.updated_at AS "updatedAt"
        FROM last_wills lw
        JOIN users u ON u.id = lw.user_id
        WHERE lw.agent_id = $1 AND lw.user_id = $2
        ORDER BY lw.updated_at DESC
        """,
        agent_id,
        user.get("sub"),
    )
    return [_response(row) for row in rows]


@router.post("", response_model=LastWillResponse)
async def create_last_will(
    data: LastWillCreate,
    user: dict = Depends(require_user),
):
    await _ensure_agent_scope(
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        user=user,
    )
    status = _normalize_status(data.status, data.contacts, data.content)
    started_at = datetime.now(UTC) if status == "active" else None
    will_id = str(uuid.uuid4())
    inserted = await db.query_raw(
        """
        INSERT INTO last_wills (
            id, user_id, agent_id, workspace_id, content, inactivity_days,
            contacts, status, started_at, created_at, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::timestamp,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
        ON CONFLICT (user_id, agent_id) DO NOTHING
        RETURNING id
        """,
        will_id,
        user["sub"],
        data.agent_id,
        data.workspace_id,
        protect_text(data.content),
        data.inactivity_days,
        _contact_payload(data.contacts),
        status,
        started_at,
    )
    if not inserted:
        raise HTTPException(status_code=409, detail="Only one last will is allowed")
    row = await _fetch_will(will_id)
    if not row:
        raise HTTPException(status_code=404, detail="Last will not found")
    return _response(row)


@router.patch("/{will_id}", response_model=LastWillResponse)
async def update_last_will(
    will_id: str,
    data: LastWillUpdate,
    user: dict = Depends(require_user),
):
    row = await _require_will_owner(will_id, user)
    current_contacts = _contacts(_field(row, "contacts"))
    target_contacts = data.contacts if data.contacts is not None else current_contacts
    target_status = data.status if data.status is not None else _field(row, "status")
    current_content = reveal_text(_field(row, "content"))
    target_content = data.content if data.content is not None else current_content
    status = _normalize_status(target_status, target_contacts, target_content)

    fields_set = (
        data.model_fields_set
        if hasattr(data, "model_fields_set")
        else getattr(data, "__fields_set__", set())
    )
    assignments: list[str] = []
    values: list[Any] = []
    if data.content is not None:
        values.append(protect_text(data.content))
        assignments.append(f"content = ${len(values)}")
    if data.inactivity_days is not None:
        values.append(data.inactivity_days)
        assignments.append(f"inactivity_days = ${len(values)}")
    if "contacts" in fields_set:
        values.append(_contact_payload(data.contacts))
        assignments.append(f"contacts = ${len(values)}::jsonb")
    if data.status is not None:
        values.append(status)
        assignments.append(f"status = ${len(values)}")
        if status == "active" and _field(row, "startedAt") is None:
            assignments.append("started_at = CURRENT_TIMESTAMP")
        if status in {"draft", "cancelled"}:
            assignments.append("started_at = NULL")
        if status in {"draft", "paused", "cancelled"}:
            assignments.append("triggered_at = NULL")
            assignments.append("delivered_at = NULL")

    if assignments:
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(will_id)
        await db.execute_raw(
            f"""
            UPDATE last_wills
            SET {", ".join(assignments)}
            WHERE id = ${len(values)}
            """,
            *values,
        )
    updated = await _fetch_will(will_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Last will not found")
    return _response(updated)


@router.post("/{will_id}/start", response_model=LastWillResponse)
async def start_last_will(
    will_id: str,
    user: dict = Depends(require_user),
):
    row = await _require_will_owner(will_id, user)
    contacts = _contacts(_field(row, "contacts"))
    _normalize_status("active", contacts, _field(row, "content") or "")
    await db.execute_raw(
        """
        UPDATE last_wills
        SET status = 'active',
            started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
            triggered_at = NULL,
            delivered_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """,
        will_id,
    )
    updated = await _fetch_will(will_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Last will not found")
    return _response(updated)


@router.post("/{will_id}/pause", response_model=LastWillResponse)
async def pause_last_will(
    will_id: str,
    user: dict = Depends(require_user),
):
    await _require_will_owner(will_id, user)
    await db.execute_raw(
        """
        UPDATE last_wills
        SET status = 'paused', updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = 'active'
        """,
        will_id,
    )
    updated = await _fetch_will(will_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Last will not found")
    return _response(updated)


@router.delete("/{will_id}", status_code=204)
async def delete_last_will(
    will_id: str,
    user: dict = Depends(require_user),
):
    await _require_will_owner(will_id, user)
    await db.execute_raw(
        """
        UPDATE last_wills
        SET content = '',
            status = 'cancelled',
            started_at = NULL,
            triggered_at = NULL,
            delivered_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """,
        will_id,
    )
    return Response(status_code=204)


@router.get("/{will_id}/deliveries", response_model=list[LastWillDelivery])
async def list_deliveries(
    will_id: str,
    user: dict = Depends(require_user),
):
    await _require_will_owner(will_id, user)
    rows = await db.query_raw(
        """
        SELECT
            id,
            last_will_id AS "lastWillId",
            channel,
            contact,
            status,
            error,
            created_at AS "createdAt",
            updated_at AS "updatedAt"
        FROM last_will_deliveries
        WHERE last_will_id = $1
        ORDER BY created_at ASC
        """,
        will_id,
    )
    deliveries: list[LastWillDelivery] = []
    for row in rows:
        try:
            contact = LastWillContact.model_validate(reveal_contact(_field(row, "contact")))
        except Exception:
            continue
        deliveries.append(
            LastWillDelivery(
                id=_field(row, "id"),
                last_will_id=_field(row, "lastWillId"),
                channel=_field(row, "channel"),
                contact=contact,
                status=_field(row, "status"),
                error=_field(row, "error"),
                created_at=_iso(_field(row, "createdAt")) or "",
                updated_at=_iso(_field(row, "updatedAt")) or "",
            )
        )
    return deliveries
