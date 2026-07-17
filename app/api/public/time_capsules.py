from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Any
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
import time as time_module
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from app.api.jwt_auth import require_user
from app.db import db
from app.models.time_capsule import (
    TimeCapsuleCreate,
    TimeCapsuleMediaUpload,
    TimeCapsuleResponse,
    TimeCapsuleUpdate,
)
from app.services.runtime.tasks import fire_background

router = APIRouter(prefix="/capsules", tags=["time-capsules"])
logger = logging.getLogger(__name__)

_VALID_STATUSES = {"draft", "sealed"}
_VALID_STATES = {"draft", "pending", "ready", "opened"}
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
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_AUDIO_SECONDS = 20
_MAX_AUDIO_BYTES = 512 * 1024
_MEDIA_DIR = Path(os.getenv("CAPSULE_MEDIA_DIR", "var/capsule_media"))
_MEDIA_PUBLIC_PREFIX = (
    os.getenv("CAPSULE_MEDIA_PUBLIC_PREFIX", "/capsules/media").strip().rstrip("/")
    or "/capsules/media"
)
mimetypes.add_type("audio/mp4", ".m4a")
_LOCAL_TZ = timezone(timedelta(hours=8))


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


def _derive_state(status: str, open_date: Any, opened_at: Any = None) -> str:
    if status == "draft":
        return "draft"
    if opened_at is not None:
        return "opened"
    raw = _date_string(open_date)
    if not raw:
        return "pending"
    try:
        target = date.fromisoformat(raw)
    except ValueError:
        return "pending"
    return "ready" if target <= datetime.now(_LOCAL_TZ).date() else "pending"


def _json_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    data = getattr(value, "data", None)
    return data if isinstance(data, dict) else None


def _strip_base64_prefix(value: str) -> str:
    raw = value.strip()
    comma = raw.find(",")
    return raw[comma + 1 :] if comma >= 0 else raw


def _mime_ext(mime: str, fallback: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "audio/mp4": ".m4a",
        "audio/aac": ".m4a",
        "audio/mpeg": ".mp3",
    }
    return mapping.get(mime.lower(), fallback)


def _storage_path(storage_key: str) -> Path:
    if "/" in storage_key or "\\" in storage_key or ".." in storage_key:
        raise HTTPException(status_code=400, detail="Invalid media storage key")
    return _MEDIA_DIR / storage_key


def _media_url(storage_key: str) -> str:
    return f"{_MEDIA_PUBLIC_PREFIX}/{storage_key}"


def _read_media_base64(storage_key: str) -> str | None:
    path = _storage_path(storage_key)
    if not path.exists() or not path.is_file():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _validate_media_storage_key(storage_key: str, user_id: str | None) -> None:
    path = _storage_path(storage_key)
    if user_id is not None and not storage_key.startswith(f"{user_id}_"):
        raise HTTPException(status_code=403, detail="Media does not belong to this user")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="Media file not found")


def _hydrate_media(media: dict | None) -> dict | None:
    if not media:
        return None
    hydrated = dict(media)
    images = hydrated.get("images")
    if isinstance(images, list):
        clean_images = []
        for image in images:
            if not isinstance(image, dict):
                continue
            item = dict(image)
            if "base64" not in item and item.get("storage_key"):
                data = _read_media_base64(str(item["storage_key"]))
                if data is not None:
                    item["base64"] = data
            clean_images.append(item)
        hydrated["images"] = clean_images
    audio = hydrated.get("audio")
    if isinstance(audio, dict):
        item = dict(audio)
        if "base64" not in item and item.get("storage_key"):
            data = _read_media_base64(str(item["storage_key"]))
            if data is not None:
                item["base64"] = data
        hydrated["audio"] = item
    return hydrated


def _media_storage_keys(media: dict | None) -> set[str]:
    if not media:
        return set()
    keys: set[str] = set()
    images = media.get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and image.get("storage_key"):
                keys.add(str(image["storage_key"]))
    audio = media.get("audio")
    if isinstance(audio, dict) and audio.get("storage_key"):
        keys.add(str(audio["storage_key"]))
    return keys


def _delete_media_files(
    media: dict | None,
    *,
    keep_keys: set[str] | None = None,
    strict: bool = False,
) -> None:
    keep = keep_keys or set()
    storage_keys = _media_storage_keys(media) - keep
    if strict and storage_keys and (not _MEDIA_DIR.exists() or not _MEDIA_DIR.is_dir()):
        raise RuntimeError(f"Capsule media directory is unavailable: {_MEDIA_DIR}")
    for storage_key in storage_keys:
        try:
            path = _storage_path(storage_key)
            if path.exists() and not path.is_file():
                raise OSError(f"capsule media path is not a file: {path}")
            if path.exists():
                path.unlink()
                if path.exists():
                    raise OSError(f"capsule media file still exists after deletion: {path}")
        except Exception as exc:
            logger.warning(
                "[capsule:media] failed to delete storage_key=%s",
                storage_key,
                exc_info=True,
            )
            if strict:
                raise RuntimeError(
                    f"Failed to delete capsule media file: {storage_key}",
                ) from exc


async def _cleanup_unreferenced_media(user_id: str, *, max_age_seconds: int = 86400) -> None:
    try:
        if not _MEDIA_DIR.exists():
            return
        now = time_module.time()
        candidates = {
            path.name
            for path in _MEDIA_DIR.glob(f"{user_id}_*")
            if path.is_file() and now - path.stat().st_mtime > max_age_seconds
        }
        if not candidates:
            return
        rows = await db.query_raw(
            """
            SELECT media
            FROM time_capsules
            WHERE user_id = $1 AND media IS NOT NULL
            """,
            user_id,
        )
        used: set[str] = set()
        for row in rows:
            used.update(_media_storage_keys(_json_dict(_field(row, "media"))))
        for storage_key in candidates - used:
            try:
                path = _storage_path(storage_key)
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception:
                logger.warning(
                    "[capsule:media] failed to cleanup orphan storage_key=%s",
                    storage_key,
                    exc_info=True,
                )
    except Exception:
        logger.warning("[capsule:media] orphan cleanup failed", exc_info=True)


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _iso_string(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _response(row, *, include_media: bool = True, hydrate_media: bool = False) -> TimeCapsuleResponse:
    open_date = _field(row, "openDate")
    sealed_at = _field(row, "sealedAt")
    opened_at = _field(row, "openedAt")
    media = _json_dict(_field(row, "media")) if include_media else None
    return TimeCapsuleResponse(
        id=_field(row, "id"),
        user_id=_field(row, "userId"),
        agent_id=_field(row, "agentId"),
        workspace_id=_field(row, "workspaceId"),
        title=_field(row, "title"),
        content=_field(row, "content"),
        media=_hydrate_media(media) if hydrate_media else media,
        skin=_field(row, "skin") or "paper",
        open_date=_date_string(open_date),
        status=_field(row, "status"),
        state=_derive_state(_field(row, "status"), open_date, opened_at),
        sealed_at=_iso_string(sealed_at) if sealed_at else None,
        opened_at=_iso_string(opened_at) if opened_at else None,
        created_at=_iso_string(_field(row, "createdAt")),
        updated_at=_iso_string(_field(row, "updatedAt")),
    )


def _redact_locked_response(response: TimeCapsuleResponse) -> TimeCapsuleResponse:
    if response.state not in {"pending", "ready"}:
        return response
    update = {
        "title": None,
        "content": "",
        "media": None,
    }
    if hasattr(response, "model_copy"):
        return response.model_copy(update=update)
    return response.copy(update=update)


async def _ensure_capsule_context_scope(
    *,
    agent_id: str | None,
    workspace_id: str | None,
    user: dict,
) -> None:
    if agent_id:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent or getattr(agent, "status", "active") == "archived":
            raise HTTPException(status_code=404, detail="Agent not found")
        if user.get("role") != "admin" and agent.userId != user.get("sub"):
            raise HTTPException(status_code=403, detail="Not your agent")
    if workspace_id:
        workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
        if not workspace or (
            user.get("role") != "admin" and workspace.userId != user.get("sub")
        ):
            raise HTTPException(status_code=400, detail="Workspace does not belong to user")
        if agent_id and workspace.agentId != agent_id:
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
        return len(base64.b64decode(_strip_base64_prefix(base64_value), validate=True))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid media base64") from exc


def _normalize_media(media: dict | None, *, user_id: str | None = None) -> dict | None:
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
        storage_key = str(image.get("storage_key") or "")
        if not data and not storage_key:
            raise HTTPException(status_code=400, detail="Image media reference is required")
        size = _decoded_size(data) if data else int(image.get("size") or 0)
        if size > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Image must be under 10MB")
        clean_image = {
            "name": str(image.get("name") or "capsule-image")[:120],
            "mime": str(image.get("mime") or "image/jpeg")[:80],
            "size": size,
        }
        if storage_key:
            _validate_media_storage_key(storage_key, user_id)
            clean_image["storage_key"] = storage_key
            clean_image["url"] = _media_url(storage_key)
        else:
            clean_image["base64"] = _strip_base64_prefix(data)
        clean_images.append(clean_image)

    audio = media.get("audio")
    clean_audio = None
    if audio is not None:
        if not isinstance(audio, dict):
            raise HTTPException(status_code=400, detail="Invalid audio media")
        data = str(audio.get("base64") or "")
        storage_key = str(audio.get("storage_key") or "")
        if not data and not storage_key:
            raise HTTPException(status_code=400, detail="Audio media reference is required")
        duration = float(audio.get("duration_seconds") or 0)
        if duration <= 0 or duration > _MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail="Audio must be 20 seconds or shorter")
        size = int(audio.get("size") or (_decoded_size(data) if data else 0))
        if size > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail="Audio is too large")
        clean_audio = {
            "name": str(audio.get("name") or "capsule-voice.m4a")[:120],
            "mime": str(audio.get("mime") or "audio/mp4")[:80],
            "size": size,
            "duration_seconds": duration,
        }
        if storage_key:
            _validate_media_storage_key(storage_key, user_id)
            clean_audio["storage_key"] = storage_key
            clean_audio["url"] = _media_url(storage_key)
        else:
            clean_audio["base64"] = _strip_base64_prefix(data)

    normalized = {"images": clean_images}
    if clean_audio is not None:
        normalized["audio"] = clean_audio
    return normalized


def _media_summary(media: dict | None) -> str:
    if not media:
        return "none"
    images = media.get("images") or []
    image_bytes = sum(int(item.get("size") or 0) for item in images if isinstance(item, dict))
    audio = media.get("audio")
    audio_bytes = int(audio.get("size") or 0) if isinstance(audio, dict) else 0
    duration = audio.get("duration_seconds") if isinstance(audio, dict) else None
    return (
        f"images={len(images)} image_bytes={image_bytes} "
        f"audio={'yes' if audio else 'no'} audio_bytes={audio_bytes} duration={duration}"
    )


def _elapsed_ms(start: float) -> int:
    return int((time_module.perf_counter() - start) * 1000)


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


async def _fetch_capsule_light(capsule_id: str) -> Any | None:
    rows = await db.query_raw(
        """
        SELECT
            id,
            user_id AS "userId",
            agent_id AS "agentId",
            workspace_id AS "workspaceId",
            title,
            content,
            NULL AS media,
            skin,
            open_date AS "openDate",
            status,
            sealed_at AS "sealedAt",
            opened_at AS "openedAt",
            created_at AS "createdAt",
            updated_at AS "updatedAt"
        FROM time_capsules
        WHERE id = $1
        LIMIT 1
        """,
        capsule_id,
    )
    return rows[0] if rows else None


async def _fetch_capsule_full(capsule_id: str) -> Any | None:
    rows = await db.query_raw(
        """
        SELECT
            id,
            user_id AS "userId",
            agent_id AS "agentId",
            workspace_id AS "workspaceId",
            title,
            content,
            media,
            skin,
            open_date AS "openDate",
            status,
            sealed_at AS "sealedAt",
            opened_at AS "openedAt",
            created_at AS "createdAt",
            updated_at AS "updatedAt"
        FROM time_capsules
        WHERE id = $1
        LIMIT 1
        """,
        capsule_id,
    )
    return rows[0] if rows else None


def _state_sql_clause(state: str) -> str:
    today_sql = "(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Shanghai')::date"
    if state == "draft":
        return "status = 'draft'"
    if state == "opened":
        return "status = 'sealed' AND opened_at IS NOT NULL"
    if state == "ready":
        return (
            "status = 'sealed' AND opened_at IS NULL "
            f"AND open_date IS NOT NULL AND open_date::date <= {today_sql}"
        )
    if state == "pending":
        return (
            "status = 'sealed' AND opened_at IS NULL "
            f"AND (open_date IS NULL OR open_date::date > {today_sql})"
        )
    raise HTTPException(status_code=400, detail="Invalid capsule state")


async def _insert_capsule_raw(
    *,
    capsule_id: str,
    user_id: str,
    agent_id: str | None,
    workspace_id: str | None,
    title: str,
    content: str,
    media: dict | None,
    skin: str,
    status: str,
    open_date: datetime | None,
    sealed_at: datetime | None,
) -> None:
    columns = [
        "id",
        "user_id",
        "agent_id",
        "title",
        "content",
        "skin",
        "status",
    ]
    values: list[Any] = [
        capsule_id,
        user_id,
        agent_id,
        title,
        content,
        skin,
        status,
    ]
    placeholders = [f"${index}" for index in range(1, len(values) + 1)]

    if workspace_id is not None:
        columns.append("workspace_id")
        values.append(workspace_id)
        placeholders.append(f"${len(values)}")
    if media is not None:
        columns.append("media")
        values.append(json.dumps(media, ensure_ascii=False))
        placeholders.append(f"${len(values)}::jsonb")
    if open_date is not None:
        columns.append("open_date")
        values.append(open_date)
        placeholders.append(f"${len(values)}::timestamp")
    if sealed_at is not None:
        columns.append("sealed_at")
        values.append(sealed_at)
        placeholders.append(f"${len(values)}::timestamp")

    await db.execute_raw(
        f"""
        INSERT INTO time_capsules ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        """,
        *values,
    )


async def _update_capsule_raw(
    capsule_id: str,
    *,
    fields: dict[str, Any],
    media: dict | None,
    set_media: bool,
    clear_media: bool,
) -> None:
    assignments: list[str] = []
    values: list[Any] = []
    for column, value in fields.items():
        values.append(value)
        cast = "::timestamp" if column in {"open_date", "sealed_at"} else ""
        assignments.append(f"{column} = ${len(values)}{cast}")
    if set_media:
        values.append(json.dumps(media, ensure_ascii=False))
        assignments.append(f"media = ${len(values)}::jsonb")
    elif clear_media:
        assignments.append("media = NULL")

    if not assignments:
        return

    assignments.append("updated_at = NOW()")
    values.append(capsule_id)
    await db.execute_raw(
        f"""
        UPDATE time_capsules
        SET {", ".join(assignments)}
        WHERE id = ${len(values)}
        """,
        *values,
    )


@router.get("", response_model=list[TimeCapsuleResponse])
async def list_capsules(
    agent_id: str | None = Query(default=None),
    workspace_id: str | None = None,
    state: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    if state is not None and state not in _VALID_STATES:
        raise HTTPException(status_code=400, detail="Invalid capsule state")
    values: list[Any] = [user.get("sub")]
    where = ["user_id = $1"]
    if state:
        where.append(_state_sql_clause(state))
    values.append(limit)
    limit_placeholder = f"${len(values)}"
    values.append(offset)
    offset_placeholder = f"${len(values)}"
    rows = await db.query_raw(
        f"""
            SELECT
                id,
                user_id AS "userId",
                agent_id AS "agentId",
                workspace_id AS "workspaceId",
                title,
                content,
                NULL AS media,
                skin,
                open_date AS "openDate",
                status,
                sealed_at AS "sealedAt",
                opened_at AS "openedAt",
                created_at AS "createdAt",
                updated_at AS "updatedAt"
            FROM time_capsules
            WHERE {" AND ".join(where)}
            ORDER BY open_date ASC, updated_at DESC
            LIMIT {limit_placeholder}
            OFFSET {offset_placeholder}
            """,
        *values,
    )
    logger.info(
        "[capsule:list] loaded rows=%s workspace=%s elapsed_ms=%s",
        len(rows),
        workspace_id,
        _elapsed_ms(started),
    )
    responses = [
        _redact_locked_response(_response(row, include_media=False))
        for row in rows
    ]
    logger.info(
        "[capsule:list] done rows=%s state=%s elapsed_ms=%s",
        len(responses),
        state,
        _elapsed_ms(started),
    )
    return responses


@router.post("/media")
async def upload_capsule_media(
    data: TimeCapsuleMediaUpload,
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    kind = data.kind.strip().lower()
    if kind not in {"image", "audio"}:
        raise HTTPException(status_code=400, detail="Invalid media kind")
    raw_base64 = _strip_base64_prefix(data.base64)
    try:
        blob = base64.b64decode(raw_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid media base64") from exc
    size = data.size or len(blob)
    if size != len(blob):
        size = len(blob)
    mime = (data.mime or ("image/jpeg" if kind == "image" else "audio/mp4")).strip()
    if kind == "image" and size > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be under 10MB")
    if kind == "audio":
        duration = float(data.duration_seconds or 0)
        if duration <= 0 or duration > _MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail="Audio must be 20 seconds or shorter")
        if size > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail="Audio is too large")
    else:
        duration = None

    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    ext = _mime_ext(mime, ".jpg" if kind == "image" else ".m4a")
    storage_key = f"{user['sub']}_{uuid.uuid4().hex}{ext}"
    _storage_path(storage_key).write_bytes(blob)
    fire_background(_cleanup_unreferenced_media(user["sub"]))
    logger.info(
        "[capsule:media] uploaded kind=%s size=%s elapsed_ms=%s",
        kind,
        size,
        _elapsed_ms(started),
    )
    response: dict[str, Any] = {
        "name": (data.name or ("capsule-image" if kind == "image" else "capsule-voice.m4a"))[:120],
        "mime": mime[:80],
        "size": size,
        "storage_key": storage_key,
        "url": _media_url(storage_key),
    }
    if duration is not None:
        response["duration_seconds"] = duration
    return response


@router.get("/media/{storage_key}")
async def get_capsule_media(
    storage_key: str,
    user: dict = Depends(require_user),
):
    path = _storage_path(storage_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    if user.get("role") != "admin" and not storage_key.startswith(f"{user['sub']}_"):
        raise HTTPException(status_code=403, detail="Not your media")
    media_type, _ = mimetypes.guess_type(path.name)
    return Response(
        content=path.read_bytes(),
        media_type=media_type or "application/octet-stream",
    )


@router.get("/{capsule_id}", response_model=TimeCapsuleResponse)
async def get_capsule(
    capsule_id: str,
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    row = await _fetch_capsule_full(capsule_id)
    logger.info("[capsule:detail] loaded id=%s elapsed_ms=%s", capsule_id, _elapsed_ms(started))
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    if user.get("role") != "admin" and _field(row, "userId") != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your capsule")
    response = _response(row, include_media=True, hydrate_media=True)
    if response.state in {"pending", "ready"}:
        raise HTTPException(status_code=403, detail="Capsule is not opened yet")
    return response


@router.post("/{capsule_id}/open", response_model=TimeCapsuleResponse)
async def open_capsule(
    capsule_id: str,
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    row = await _fetch_capsule_full(capsule_id)
    logger.info("[capsule:open] loaded id=%s elapsed_ms=%s", capsule_id, _elapsed_ms(started))
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    if user.get("role") != "admin" and _field(row, "userId") != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your capsule")
    state = _derive_state(
        _field(row, "status"),
        _field(row, "openDate"),
        _field(row, "openedAt"),
    )
    if state == "draft":
        raise HTTPException(status_code=400, detail="Draft capsule cannot be opened")
    if state == "pending":
        raise HTTPException(status_code=400, detail="Capsule is not ready to open")
    if state != "opened":
        await db.execute_raw(
            """
            UPDATE time_capsules
            SET opened_at = NOW(), updated_at = NOW()
            WHERE id = $1
            """,
            capsule_id,
        )
        row = await _fetch_capsule_full(capsule_id)
        if not row:
            raise HTTPException(status_code=404, detail="Capsule not found")
    logger.info("[capsule:open] done id=%s elapsed_ms=%s", capsule_id, _elapsed_ms(started))
    return _response(row, include_media=True, hydrate_media=True)


@router.post("", response_model=TimeCapsuleResponse)
async def create_capsule(
    data: TimeCapsuleCreate,
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    logger.info(
        "[capsule:create] start agent=%s workspace=%s status=%s media=%s",
        data.agent_id,
        data.workspace_id,
        data.status,
        _media_summary(data.media),
    )
    await _ensure_capsule_context_scope(
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        user=user,
    )
    logger.info("[capsule:create] scope_ok elapsed_ms=%s", _elapsed_ms(started))
    status, open_date, sealed_at = _normalize_create(data)
    content = data.content.strip()
    media = _normalize_media(data.media, user_id=user["sub"])
    logger.info("[capsule:create] normalized elapsed_ms=%s", _elapsed_ms(started))
    capsule_id = str(uuid.uuid4())
    await _insert_capsule_raw(
        capsule_id=capsule_id,
        user_id=user["sub"],
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        title=(data.title or _title_from_content(content)).strip()[:80],
        content=content,
        media=media,
        skin=_normalize_skin(data.skin),
        status=status,
        open_date=open_date,
        sealed_at=sealed_at,
    )
    logger.info("[capsule:create] insert_done elapsed_ms=%s", _elapsed_ms(started))
    row = await _fetch_capsule_light(capsule_id)
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    logger.info("[capsule:create] fetch_light_done elapsed_ms=%s", _elapsed_ms(started))
    return _response(row, include_media=False)


@router.patch("/{capsule_id}", response_model=TimeCapsuleResponse)
async def update_capsule(
    capsule_id: str,
    data: TimeCapsuleUpdate,
    user: dict = Depends(require_user),
):
    started = time_module.perf_counter()
    fields_set = (
        data.model_fields_set
        if hasattr(data, "model_fields_set")
        else getattr(data, "__fields_set__", set())
    )
    logger.info(
        "[capsule:update] start id=%s fields=%s media=%s",
        capsule_id,
        sorted(fields_set),
        _media_summary(data.media) if "media" in fields_set else "omitted",
    )
    row = await db.timecapsule.find_unique(where={"id": capsule_id})
    logger.info("[capsule:update] loaded elapsed_ms=%s", _elapsed_ms(started))
    if not row:
        raise HTTPException(status_code=404, detail="Capsule not found")
    if user.get("role") != "admin" and row.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your capsule")
    if row.openedAt is not None:
        raise HTTPException(status_code=409, detail="Opened capsules are read-only")
    previous_media = _json_dict(row.media)
    update_fields: dict[str, Any] = {}
    if data.content is not None:
        content = data.content.strip()
        update_fields["content"] = content
        if data.title is None:
            update_fields["title"] = _title_from_content(content)
    if data.title is not None:
        update_fields["title"] = data.title.strip()[:80] or None
    clear_media = False
    set_media = False
    media: dict | None = None
    if "media" in fields_set:
        media = _normalize_media(data.media, user_id=user["sub"])
        if media is not None:
            set_media = True
        else:
            clear_media = True
    if data.skin is not None:
        update_fields["skin"] = _normalize_skin(data.skin)
    if data.open_date is not None:
        update_fields["open_date"] = _date_to_datetime(data.open_date)
    if data.status is not None:
        if data.status not in _VALID_STATUSES:
            raise HTTPException(status_code=400, detail="Invalid capsule status")
        if data.status == "sealed" and (
            data.open_date is None and row.openDate is None
        ):
            raise HTTPException(status_code=400, detail="open_date is required when sealing")
        update_fields["status"] = data.status
        update_fields["sealed_at"] = datetime.now(UTC) if data.status == "sealed" else None
    logger.info(
        "[capsule:update] normalized fields=%s clear_media=%s elapsed_ms=%s",
        sorted(update_fields.keys()) + (["media"] if set_media else []),
        clear_media,
        _elapsed_ms(started),
    )
    if not update_fields and not set_media and not clear_media:
        logger.info("[capsule:update] noop elapsed_ms=%s", _elapsed_ms(started))
        return _response(row, include_media=False)
    await _update_capsule_raw(
        capsule_id,
        fields=update_fields,
        media=media,
        set_media=set_media,
        clear_media=clear_media,
    )
    if set_media or clear_media:
        _delete_media_files(previous_media, keep_keys=_media_storage_keys(media))
    logger.info("[capsule:update] raw_update_done elapsed_ms=%s", _elapsed_ms(started))
    updated = await _fetch_capsule_light(capsule_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Capsule not found")
    logger.info("[capsule:update] fetch_light_done elapsed_ms=%s", _elapsed_ms(started))
    logger.info("[capsule:update] done elapsed_ms=%s", _elapsed_ms(started))
    return _response(updated, include_media=False)


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
    media = _json_dict(row.media)
    _delete_media_files(media, strict=True)
    await db.timecapsule.delete(where={"id": capsule_id})
    return Response(status_code=204)
