from datetime import UTC, date, datetime, time
from typing import Any
import base64

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from prisma import Json

from app.api.jwt_auth import require_user
from app.db import db
from app.models.time_capsule import (
    TimeCapsuleCreate,
    TimeCapsuleResponse,
    TimeCapsuleUpdate,
)

router = APIRouter(prefix="/capsules", tags=["time-capsules"])

_VALID_STATUSES = {"draft", "sealed"}
_VALID_STATES = {"draft", "pending", "opened"}
_VALID_SKINS = {
    "paper",
    "warm",
    "mint",
    "night",
    "rose",
    "lavender",
    "sky",
    "linen",
}
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_AUDIO_SECONDS = 20
_MAX_AUDIO_BYTES = 512 * 1024


def _date_to_datetime(value: date | None) -> datetime | None:
    if value is None:
        return None
    return datetime.combine(value, time.min, tzinfo=UTC)


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _derive_state(status: str, open_date: Any) -> str:
    if status == "draft":
        return "draft"
    raw = _date_string(open_date)
    if not raw:
        return "pending"
    try:
        target = date.fromisoformat(raw)
    except ValueError:
        return "pending"
    return "opened" if target <= datetime.now(UTC).date() else "pending"


def _json_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    data = getattr(value, "data", None)
    return data if isinstance(data, dict) else None


def _response(row) -> TimeCapsuleResponse:
    return TimeCapsuleResponse(
        id=row.id,
        user_id=row.userId,
        agent_id=row.agentId,
        workspace_id=row.workspaceId,
        title=row.title,
        content=row.content,
        media=_json_dict(row.media),
        skin=row.skin or "paper",
        open_date=_date_string(row.openDate),
        status=row.status,
        state=_derive_state(row.status, row.openDate),
        sealed_at=row.sealedAt.isoformat() if row.sealedAt else None,
        created_at=row.createdAt.isoformat(),
        updated_at=row.updatedAt.isoformat(),
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


def _normalize_create(data: TimeCapsuleCreate) -> tuple[str, datetime | None, datetime | None]:
    status = data.status
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid capsule status")
    if status == "sealed" and data.open_date is None:
        raise HTTPException(status_code=400, detail="open_date is required when sealing")
    open_date = _date_to_datetime(data.open_date)
    sealed_at = datetime.now(UTC) if status == "sealed" else None
    return status, open_date, sealed_at


def _decoded_size(base64_value: str) -> int:
    try:
        return len(base64.b64decode(base64_value, validate=True))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid media base64") from exc


def _normalize_media(media: dict | None) -> dict | None:
    if not media:
        return None
    images = media.get("images") or []
    if not isinstance(images, list):
        raise HTTPException(status_code=400, detail="media.images must be a list")
    if len(images) > 1:
        raise HTTPException(status_code=400, detail="Only one capsule image is allowed")
    clean_images: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            raise HTTPException(status_code=400, detail="Invalid image media")
        data = str(image.get("base64") or "")
        if not data:
            raise HTTPException(status_code=400, detail="Image base64 is required")
        size = int(image.get("size") or _decoded_size(data))
        if size > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Image must be under 2MB")
        clean_images.append({
            "name": str(image.get("name") or "capsule-image")[:120],
            "mime": str(image.get("mime") or "image/jpeg")[:80],
            "size": size,
            "base64": data,
        })

    audio = media.get("audio")
    clean_audio = None
    if audio is not None:
        if not isinstance(audio, dict):
            raise HTTPException(status_code=400, detail="Invalid audio media")
        data = str(audio.get("base64") or "")
        if not data:
            raise HTTPException(status_code=400, detail="Audio base64 is required")
        duration = float(audio.get("duration_seconds") or 0)
        if duration <= 0 or duration > _MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail="Audio must be 20 seconds or shorter")
        size = int(audio.get("size") or _decoded_size(data))
        if size > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail="Audio is too large")
        clean_audio = {
            "name": str(audio.get("name") or "capsule-voice.m4a")[:120],
            "mime": str(audio.get("mime") or "audio/mp4")[:80],
            "size": size,
            "duration_seconds": duration,
            "base64": data,
        }

    normalized = {"images": clean_images}
    if clean_audio is not None:
        normalized["audio"] = clean_audio
    return normalized


def _normalize_skin(skin: str | None) -> str:
    value = (skin or "paper").strip()
    if value not in _VALID_SKINS:
        raise HTTPException(status_code=400, detail="Invalid capsule skin")
    return value


def _title_from_content(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if not first_line:
        return "未命名胶囊"
    return first_line[:18]


async def _clear_capsule_media(capsule_id: str) -> None:
    await db.execute_raw(
        """
        UPDATE "time_capsules"
        SET "media" = NULL, "updated_at" = NOW()
        WHERE "id" = $1
        """,
        capsule_id,
    )


@router.get("", response_model=list[TimeCapsuleResponse])
async def list_capsules(
    agent_id: str = Query(...),
    workspace_id: str | None = None,
    state: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    if state is not None and state not in _VALID_STATES:
        raise HTTPException(status_code=400, detail="Invalid capsule state")
    await _ensure_agent_scope(agent_id=agent_id, workspace_id=workspace_id, user=user)
    where: dict[str, Any] = {"agentId": agent_id, "userId": user.get("sub")}
    if workspace_id:
        where["workspaceId"] = workspace_id
    rows = await db.timecapsule.find_many(
        where=where,
        order=[{"openDate": "asc"}, {"updatedAt": "desc"}],
    )
    responses = [_response(row) for row in rows]
    if state:
        responses = [item for item in responses if item.state == state]
    return responses


@router.post("", response_model=TimeCapsuleResponse)
async def create_capsule(
    data: TimeCapsuleCreate,
    user: dict = Depends(require_user),
):
    await _ensure_agent_scope(
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        user=user,
    )
    status, open_date, sealed_at = _normalize_create(data)
    content = data.content.strip()
    media = _normalize_media(data.media)
    create_data: dict[str, Any] = {
        "user": {"connect": {"id": user["sub"]}},
        "agent": {"connect": {"id": data.agent_id}},
        "title": (data.title or _title_from_content(content)).strip()[:80],
        "content": content,
        "skin": _normalize_skin(data.skin),
        "status": status,
    }
    if data.workspace_id:
        create_data["workspace"] = {"connect": {"id": data.workspace_id}}
    if media is not None:
        create_data["media"] = Json(media)
    if open_date is not None:
        create_data["openDate"] = open_date
    if sealed_at is not None:
        create_data["sealedAt"] = sealed_at
    row = await db.timecapsule.create(
        data=create_data,
    )
    return _response(row)


@router.patch("/{capsule_id}", response_model=TimeCapsuleResponse)
async def update_capsule(
    capsule_id: str,
    data: TimeCapsuleUpdate,
    user: dict = Depends(require_user),
):
    row = await db.timecapsule.find_unique(where={"id": capsule_id})
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    if user.get("role") != "admin" and row.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your capsule")
    update_data: dict[str, Any] = {}
    if data.content is not None:
        content = data.content.strip()
        update_data["content"] = content
        if data.title is None:
            update_data["title"] = _title_from_content(content)
    if data.title is not None:
        update_data["title"] = data.title.strip()[:80] or None
    fields_set = (
        data.model_fields_set
        if hasattr(data, "model_fields_set")
        else getattr(data, "__fields_set__", set())
    )
    clear_media = False
    if "media" in fields_set:
        media = _normalize_media(data.media)
        if media is not None:
            update_data["media"] = Json(media)
        else:
            clear_media = True
    if data.skin is not None:
        update_data["skin"] = _normalize_skin(data.skin)
    if data.open_date is not None:
        update_data["openDate"] = _date_to_datetime(data.open_date)
    if data.status is not None:
        if data.status not in _VALID_STATUSES:
            raise HTTPException(status_code=400, detail="Invalid capsule status")
        if data.status == "sealed" and (
            data.open_date is None and row.openDate is None
        ):
            raise HTTPException(status_code=400, detail="open_date is required when sealing")
        update_data["status"] = data.status
        update_data["sealedAt"] = datetime.now(UTC) if data.status == "sealed" else None
    if not update_data and not clear_media:
        return _response(row)
    updated = row
    if update_data:
        updated = await db.timecapsule.update(
            where={"id": capsule_id},
            data=update_data,
        )
    if clear_media:
        await _clear_capsule_media(capsule_id)
        updated = await db.timecapsule.find_unique(where={"id": capsule_id})
        if not updated:
            raise HTTPException(status_code=404, detail="Capsule not found")
    return _response(updated)


@router.delete("/{capsule_id}", status_code=204)
async def delete_capsule(
    capsule_id: str,
    user: dict = Depends(require_user),
):
    row = await db.timecapsule.find_unique(where={"id": capsule_id})
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    if user.get("role") != "admin" and row.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your capsule")
    await db.timecapsule.delete(where={"id": capsule_id})
    return Response(status_code=204)
