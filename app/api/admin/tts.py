from __future__ import annotations

from datetime import UTC, datetime
import logging
import mimetypes
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.speech_output.client import SpeechSynthesisError, synthesize_speech
from app.services.speech_output.style import (
    MAX_INSTRUCTION_BILLABLE_CHARACTERS,
    decorate_text_with_emotion,
    instruction_billable_characters,
    resolve_style_instruction,
)
from app.services.speech_output.usage import record_tts_usage
from app.services.speech_output.voice_enrollment import (
    create_cloned_voice,
    delete_cloned_voice,
    delete_enrollment_audio_later,
    enrollment_storage_path,
    save_enrollment_audio,
    signed_enrollment_url,
    verify_signed_enrollment_url,
)
from app.services.speech_output.voices import QWEN_AUDIO_TTS_MODEL
from app.services.runtime.tasks import fire_background


router = APIRouter(
    prefix="/admin-api/tts",
    tags=["admin", "tts"],
    dependencies=[Depends(require_admin_jwt)],
)
public_router = APIRouter(tags=["tts-enrollment"])
logger = logging.getLogger(__name__)


class AgentTtsPayload(BaseModel):
    voice_profile_id: str
    rate: float = Field(ge=0.5, le=2.0)
    pitch: float = Field(ge=0.5, le=2.0)
    volume: int = Field(ge=0, le=100)
    seed: int = Field(ge=0, le=65_535)
    instruction: str | None = None
    auto_emotion: bool = True
    emotion_scale: float = Field(ge=0, le=2.0)


class AgentTtsPreviewPayload(AgentTtsPayload):
    text: str = Field(min_length=1, max_length=300)
    emotion: str | None = None
    intensity: int = Field(default=50, ge=0, le=100)


class VoiceProfilePatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    gender: str | None = None
    enabled: bool | None = None


def _validate_instruction(value: str | None) -> str | None:
    cleaned = (value or "").strip() or None
    if instruction_billable_characters(cleaned) > (
        MAX_INSTRUCTION_BILLABLE_CHARACTERS
    ):
        raise HTTPException(
            status_code=422,
            detail="风格指令超过 100 个计费字符（汉字按 2 个计算）",
        )
    return cleaned


async def _voice_profile(profile_id: str, *, require_enabled: bool) -> dict:
    rows = await db.query_raw(
        """
        SELECT id, display_name, provider, model, voice_id, gender, source,
               enabled, provider_request_id, consent_confirmed_at,
               consent_confirmed_by, created_at, updated_at
        FROM tts_voice_profiles
        WHERE id = $1
        LIMIT 1
        """,
        profile_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="音色不存在")
    row = dict(rows[0])
    if require_enabled and not row.get("enabled"):
        raise HTTPException(status_code=422, detail="该音色已停用")
    if row.get("model") != QWEN_AUDIO_TTS_MODEL:
        raise HTTPException(status_code=422, detail="音色与当前 TTS 模型不兼容")
    return row


async def _agent_tts_response(agent_id: str) -> dict:
    rows = await db.query_raw(
        """
        SELECT id, name, gender, user_id, tts_voice_id, tts_rate, tts_pitch,
               tts_volume, tts_seed, tts_instruction, tts_auto_emotion,
               tts_emotion_scale
        FROM ai_agents
        WHERE id = $1
        LIMIT 1
        """,
        agent_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Agent not found")
    row = dict(rows[0])
    profile_rows = await db.query_raw(
        """
        SELECT id
        FROM tts_voice_profiles
        WHERE provider = 'dashscope'
          AND model = $1
          AND voice_id = $2
        LIMIT 1
        """,
        QWEN_AUDIO_TTS_MODEL,
        str(row.get("tts_voice_id") or ""),
    )
    return {
        "agent_id": agent_id,
        "agent_name": str(row.get("name") or ""),
        "gender": row.get("gender"),
        "voice_profile_id": (
            str(profile_rows[0].get("id")) if profile_rows else None
        ),
        "voice_id": row.get("tts_voice_id"),
        "rate": float(row.get("tts_rate") or 1.0),
        "pitch": float(row.get("tts_pitch") or 1.0),
        "volume": int(
            row.get("tts_volume")
            if row.get("tts_volume") is not None
            else 50
        ),
        "seed": int(row.get("tts_seed") or 0),
        "instruction": row.get("tts_instruction"),
        "instruction_billable_characters": instruction_billable_characters(
            row.get("tts_instruction")
        ),
        "auto_emotion": bool(
            row.get("tts_auto_emotion")
            if row.get("tts_auto_emotion") is not None
            else True
        ),
        "emotion_scale": float(
            row.get("tts_emotion_scale")
            if row.get("tts_emotion_scale") is not None
            else 1.0
        ),
    }


@router.get("/voices")
async def list_voice_profiles() -> dict:
    rows = await db.query_raw(
        """
        SELECT v.id, v.display_name, v.provider, v.model, v.voice_id,
               v.gender, v.source, v.enabled, v.provider_request_id,
               v.consent_confirmed_at, v.consent_confirmed_by,
               v.created_at, v.updated_at,
               (
                   SELECT COUNT(*)::int
                   FROM ai_agents a
                   WHERE a.tts_voice_id = v.voice_id
               ) AS agent_count
        FROM tts_voice_profiles v
        WHERE v.model = $1
        ORDER BY v.gender, v.source, v.created_at, v.id
        """,
        QWEN_AUDIO_TTS_MODEL,
    )
    return {"voices": [dict(row) for row in rows]}


@router.patch("/voices/{profile_id}")
async def update_voice_profile(
    profile_id: str,
    payload: VoiceProfilePatch,
) -> dict:
    current = await _voice_profile(profile_id, require_enabled=False)
    gender = payload.gender or str(current.get("gender") or "")
    if gender not in {"female", "male"}:
        raise HTTPException(status_code=422, detail="gender 必须是 female 或 male")
    enabled = (
        payload.enabled
        if payload.enabled is not None
        else bool(current.get("enabled"))
    )
    current_gender = str(current.get("gender") or "")
    leaving_current_pool = bool(current.get("enabled")) and (
        not enabled or gender != current_gender
    )
    if leaving_current_pool:
        rows = await db.query_raw(
            """
            SELECT COUNT(*)::int AS count
            FROM tts_voice_profiles
            WHERE model = $1 AND gender = $2 AND enabled = true AND id <> $3
            """,
            QWEN_AUDIO_TTS_MODEL,
            current_gender,
            profile_id,
        )
        if int(rows[0].get("count") or 0) <= 0:
            raise HTTPException(
                status_code=409,
                detail="每个性别至少需要保留一个启用音色",
            )
    rows = await db.query_raw(
        """
        UPDATE tts_voice_profiles
        SET display_name = $1, gender = $2, enabled = $3, updated_at = NOW()
        WHERE id = $4
        RETURNING *
        """,
        (payload.display_name or str(current.get("display_name") or "")).strip(),
        gender,
        enabled,
        profile_id,
    )
    return dict(rows[0])


@router.delete("/voices/{profile_id}")
async def delete_voice_profile(profile_id: str) -> dict[str, str]:
    current = await _voice_profile(profile_id, require_enabled=False)
    if current.get("source") == "system":
        raise HTTPException(status_code=409, detail="系统音色不能删除")
    count_rows = await db.query_raw(
        "SELECT COUNT(*)::int AS count FROM ai_agents WHERE tts_voice_id = $1",
        str(current.get("voice_id") or ""),
    )
    if int(count_rows[0].get("count") or 0) > 0:
        raise HTTPException(status_code=409, detail="该音色仍被 Agent 使用")
    await delete_cloned_voice(str(current.get("voice_id") or ""))
    await db.execute_raw(
        "DELETE FROM tts_voice_profiles WHERE id = $1",
        profile_id,
    )
    return {"status": "ok"}


@router.post("/voices/clone")
async def clone_voice_profile(
    request: Request,
    file: UploadFile = File(...),
    display_name: str = Form(...),
    gender: str = Form(...),
    prefix: str = Form("voice"),
    consent_confirmed: bool = Form(...),
    admin: dict = Depends(require_admin_jwt),
) -> dict:
    if gender not in {"female", "male"}:
        raise HTTPException(status_code=422, detail="gender 必须是 female 或 male")
    if not consent_confirmed:
        raise HTTPException(status_code=422, detail="必须确认已获得音色授权")
    clean_display_name = display_name.strip()
    if not clean_display_name:
        raise HTTPException(status_code=422, detail="音色名称不能为空")
    blob = await file.read()
    try:
        storage_key, _ = await save_enrollment_audio(
            blob=blob,
            mime=file.content_type,
            filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "[TTS-VOICE] enrollment started gender=%s source_bytes=%s",
        gender,
        len(blob),
    )
    try:
        try:
            audio_url = signed_enrollment_url(
                storage_key=storage_key,
                request_base_url=str(request.base_url),
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            result = await create_cloned_voice(
                prefix=prefix,
                audio_url=audio_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SpeechSynthesisError as exc:
            logger.warning("[TTS-VOICE] provider enrollment failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        try:
            rows = await db.query_raw(
                """
                INSERT INTO tts_voice_profiles (
                    id, display_name, provider, model, voice_id, gender, source,
                    enabled, provider_request_id, consent_confirmed_at,
                    consent_confirmed_by, created_at, updated_at
                )
                VALUES (
                    $1, $2, 'dashscope', $3, $4, $5, 'cloned',
                    true, $6, $7, $8, NOW(), NOW()
                )
                RETURNING *
                """,
                str(uuid.uuid4()),
                clean_display_name[:80],
                QWEN_AUDIO_TTS_MODEL,
                result.voice_id,
                gender,
                result.request_id,
                datetime.now(UTC),
                str(admin.get("sub") or ""),
            )
            logger.info(
                "[TTS-VOICE] enrollment created voice_id=%s",
                result.voice_id,
            )
            return dict(rows[0])
        except Exception:
            logger.exception("[TTS-VOICE] enrollment persistence failed")
            try:
                await delete_cloned_voice(result.voice_id)
            except Exception:
                pass
            raise
    finally:
        fire_background(delete_enrollment_audio_later(storage_key))


@router.get("/agents/{agent_id}")
async def get_agent_tts_config(agent_id: str) -> dict:
    return await _agent_tts_response(agent_id)


@router.put("/agents/{agent_id}")
async def update_agent_tts_config(
    agent_id: str,
    payload: AgentTtsPayload,
) -> dict:
    profile = await _voice_profile(
        payload.voice_profile_id,
        require_enabled=True,
    )
    instruction = _validate_instruction(payload.instruction)
    changed = await db.execute_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1,
            tts_rate = $2,
            tts_pitch = $3,
            tts_volume = $4,
            tts_seed = $5,
            tts_instruction = $6,
            tts_auto_emotion = $7,
            tts_emotion_scale = $8,
            updated_at = NOW()
        WHERE id = $9
        """,
        str(profile.get("voice_id") or ""),
        payload.rate,
        payload.pitch,
        payload.volume,
        payload.seed,
        instruction,
        payload.auto_emotion,
        payload.emotion_scale,
        agent_id,
    )
    if not changed:
        raise HTTPException(status_code=404, detail="Agent not found")
    return await _agent_tts_response(agent_id)


@router.post("/agents/{agent_id}/preview")
async def preview_agent_tts(
    agent_id: str,
    payload: AgentTtsPreviewPayload,
) -> Response:
    profile = await _voice_profile(
        payload.voice_profile_id,
        require_enabled=True,
    )
    instruction = resolve_style_instruction(_validate_instruction(payload.instruction))
    text = decorate_text_with_emotion(
        payload.text,
        payload.emotion,
        payload.intensity,
        enabled=payload.auto_emotion,
        scale=payload.emotion_scale,
    )
    agent_rows = await db.query_raw(
        "SELECT user_id FROM ai_agents WHERE id = $1 LIMIT 1",
        agent_id,
    )
    if not agent_rows:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        speech = await synthesize_speech(
            text=text,
            voice_id=str(profile.get("voice_id") or ""),
            instruction=instruction,
            rate=payload.rate,
            pitch=payload.pitch,
            volume=payload.volume,
            seed=payload.seed,
            model=QWEN_AUDIO_TTS_MODEL,
        )
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await record_tts_usage(
        speech=speech,
        user_id=str(agent_rows[0].get("user_id") or ""),
        agent_id=agent_id,
        conversation_id=None,
        message_id=None,
        source="admin_preview",
    )
    return Response(
        content=speech.audio,
        media_type=speech.mime,
        headers={
            "X-TTS-Duration-Milliseconds": str(speech.duration_milliseconds),
            "X-TTS-Billable-Characters": str(speech.billable_characters),
            "X-TTS-Cost-CNY": f"{speech.cost_cny:.6f}",
        },
    )


@public_router.get("/admin-api/tts/enrollment-audio/{storage_key}")
async def get_signed_enrollment_audio(
    storage_key: str,
    expires: int = Query(...),
    signature: str = Query(...),
):
    if not verify_signed_enrollment_url(storage_key, expires, signature):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")
    try:
        path = enrollment_storage_path(storage_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Enrollment audio not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": "private, no-store"},
    )
