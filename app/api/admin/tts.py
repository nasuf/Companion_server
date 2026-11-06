from __future__ import annotations

import logging
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.speech_output.client import (
    TTS_PROVIDER,
    SpeechSynthesisError,
    SynthesizedSpeech,
    synthesize_speech,
)
from app.services.speech_output.style import (
    DEFAULT_STYLE_INSTRUCTION,
    MAX_INSTRUCTION_BILLABLE_CHARACTERS,
    build_speech_plan,
    instruction_billable_characters,
    resolve_voice_emotion,
)
from app.services.speech_output.policy import effective_tts_probability
from app.services.speech_output.usage import record_tts_usage
from app.services.speech_output.voice_enrollment import (
    create_cloned_voice,
    delete_cloned_voice,
    prepare_enrollment_audio,
)
from app.services.speech_output.voices import (
    DEFAULT_SPEECH_RATE,
    TTS_MODEL,
    resolve_agent_tts_model,
)


router = APIRouter(
    prefix="/admin-api/tts",
    tags=["admin", "tts"],
    dependencies=[Depends(require_admin_jwt)],
)
logger = logging.getLogger(__name__)


class AgentTtsPayload(BaseModel):
    # pitch/seed were DashScope-only; extra fields from older clients are
    # ignored by pydantic's default config.
    voice_profile_id: str
    rate: float = Field(ge=0.5, le=2.0)
    volume: int = Field(ge=0, le=100)
    instruction: str | None = None
    auto_emotion: bool = True
    emotion_scale: float = Field(ge=0, le=2.0)
    # Omitted keeps the stored value; null inherits the global probability.
    output_probability: int | None = Field(default=None, ge=0, le=100)


class AgentTtsPreviewPayload(AgentTtsPayload):
    text: str = Field(min_length=1, max_length=300)
    emotion: str | None = None
    intensity: int = Field(default=50, ge=0, le=100)


# Library auditions use the bare voice: no agent instruction or emotion.
VOICE_SAMPLE_TEXT = "你好呀，今天过得怎么样？有什么想和我聊聊的吗？"


class VoicePreviewPayload(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=300)
    rate: float = Field(default=DEFAULT_SPEECH_RATE, ge=0.5, le=2.0)
    volume: int = Field(default=50, ge=0, le=100)


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
               enabled, is_default, provider_request_id, consent_confirmed_at,
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
    if row.get("provider") != TTS_PROVIDER or row.get("model") != TTS_MODEL:
        raise HTTPException(status_code=422, detail="音色与当前 TTS 模型不兼容")
    return row


async def _agent_tts_response(agent_id: str) -> dict:
    rows = await db.query_raw(
        """
        SELECT id, name, gender, user_id, tts_voice_id, tts_rate,
               tts_volume, tts_instruction, tts_auto_emotion,
               tts_emotion_scale, tts_output_probability
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
        WHERE provider = $1
          AND model = $2
          AND voice_id = $3
        LIMIT 1
        """,
        TTS_PROVIDER,
        TTS_MODEL,
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
        "rate": float(row.get("tts_rate") or DEFAULT_SPEECH_RATE),
        "volume": int(
            row.get("tts_volume")
            if row.get("tts_volume") is not None
            else 50
        ),
        "instruction": row.get("tts_instruction"),
        "default_instruction": DEFAULT_STYLE_INSTRUCTION,
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
        "output_probability": (
            int(row["tts_output_probability"])
            if row.get("tts_output_probability") is not None
            else None
        ),
        "global_output_probability": await effective_tts_probability(),
    }


async def _voice_list() -> list[dict]:
    rows = await db.query_raw(
        """
        SELECT v.id, v.display_name, v.provider, v.model, v.voice_id,
               v.gender, v.source, v.enabled, v.is_default,
               v.provider_request_id,
               v.consent_confirmed_at, v.consent_confirmed_by,
               v.created_at, v.updated_at,
               (
                   SELECT COUNT(*)::int
                   FROM ai_agents a
                   WHERE a.tts_voice_id = v.voice_id
               ) AS agent_count
        FROM tts_voice_profiles v
        WHERE v.provider = $1 AND v.model = $2
        ORDER BY v.gender, v.source, v.created_at, v.id
        """,
        TTS_PROVIDER,
        TTS_MODEL,
    )
    return [dict(row) for row in rows]


@router.get("/voices")
async def list_voice_profiles() -> dict:
    return {"voices": await _voice_list()}


@router.put("/voices/{profile_id}/default")
async def set_default_voice_profile(profile_id: str) -> dict:
    """New agents of this gender bind to it; existing agents keep theirs."""
    current = await _voice_profile(profile_id, require_enabled=True)
    async with db.tx() as tx:
        # Clear first: the partial unique index allows one default per gender.
        await tx.execute_raw(
            """
            UPDATE tts_voice_profiles
            SET is_default = false, updated_at = NOW()
            WHERE provider = $1 AND model = $2 AND gender = $3
              AND is_default AND id <> $4
            """,
            TTS_PROVIDER,
            TTS_MODEL,
            str(current.get("gender") or ""),
            profile_id,
        )
        await tx.execute_raw(
            """
            UPDATE tts_voice_profiles
            SET is_default = true, updated_at = NOW()
            WHERE id = $1 AND NOT is_default
            """,
            profile_id,
        )
    logger.info(
        "[TTS-VOICE] default voice gender=%s voice_id=%s",
        current.get("gender"),
        current.get("voice_id"),
    )
    return {"voices": await _voice_list()}


@router.post("/voices/{profile_id}/preview")
async def preview_voice_profile(
    profile_id: str,
    payload: VoicePreviewPayload,
    admin: dict = Depends(require_admin_jwt),
) -> Response:
    # Disabled voices can be auditioned before they are enabled.
    profile = await _voice_profile(profile_id, require_enabled=False)
    try:
        model = await resolve_agent_tts_model()
        speech = await synthesize_speech(
            text=(payload.text or "").strip() or VOICE_SAMPLE_TEXT,
            voice_id=str(profile.get("voice_id") or ""),
            instruction=None,
            rate=payload.rate,
            volume=payload.volume,
            model=model,
        )
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await record_tts_usage(
        speech=speech,
        user_id=str(admin.get("sub") or ""),
        agent_id=None,
        conversation_id=None,
        message_id=None,
        source="admin_voice_preview",
    )
    return _audio_response(speech)


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
    if current.get("is_default") and (not enabled or gender != current_gender):
        raise HTTPException(
            status_code=409,
            detail="默认音色不能停用或改性别，请先把其他音色设为默认",
        )
    leaving_current_pool = bool(current.get("enabled")) and (
        not enabled or gender != current_gender
    )
    if leaving_current_pool:
        rows = await db.query_raw(
            """
            SELECT COUNT(*)::int AS count
            FROM tts_voice_profiles
            WHERE provider = $1 AND model = $2 AND gender = $3
              AND enabled = true AND id <> $4
            """,
            TTS_PROVIDER,
            TTS_MODEL,
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
    if current.get("is_default"):
        raise HTTPException(status_code=409, detail="默认音色不能删除")
    count_rows = await db.query_raw(
        "SELECT COUNT(*)::int AS count FROM ai_agents WHERE tts_voice_id = $1",
        str(current.get("voice_id") or ""),
    )
    if int(count_rows[0].get("count") or 0) > 0:
        raise HTTPException(status_code=409, detail="该音色仍被 Agent 使用")
    # Delete at DashScope first: a library row without a provider voice is
    # harmless, a provider voice without a row is a leaked clone slot.
    try:
        await delete_cloned_voice(str(current.get("voice_id") or ""))
    except SpeechSynthesisError as exc:
        logger.warning("[TTS-VOICE] provider deletion failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await db.execute_raw(
        "DELETE FROM tts_voice_profiles WHERE id = $1",
        profile_id,
    )
    return {"status": "ok"}


@router.post("/voices/clone")
async def clone_voice_profile(
    file: UploadFile = File(...),
    display_name: str = Form(...),
    gender: str = Form(...),
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
        audio = await prepare_enrollment_audio(
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
        result = await create_cloned_voice(audio=audio)
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
                $1, $2, $3, $4, $5, $6, 'cloned',
                true, $7, timezone('UTC', statement_timestamp()), $8, NOW(), NOW()
            )
            RETURNING *
            """,
            str(uuid.uuid4()),
            clean_display_name[:80],
            TTS_PROVIDER,
            TTS_MODEL,
            result.voice_id,
            gender,
            result.request_id,
            str(admin.get("sub") or ""),
        )
    except Exception as exc:
        # Roll the clone back so the account's clone slots do not leak.
        logger.exception(
            "[TTS-VOICE] enrollment persistence failed; deleting provider voice_id=%s",
            result.voice_id,
        )
        try:
            await delete_cloned_voice(result.voice_id)
        except SpeechSynthesisError:
            logger.exception(
                "[TTS-VOICE] orphaned provider voice_id=%s gender=%s display_name=%s request_id=%s",
                result.voice_id, gender, clean_display_name[:80], result.request_id,
            )
        raise HTTPException(
            status_code=503,
            detail="音色保存失败，请重试",
        ) from exc
    logger.info("[TTS-VOICE] enrollment created voice_id=%s", result.voice_id)
    return dict(rows[0])


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
    keep_probability = "output_probability" not in payload.model_fields_set
    changed = await db.execute_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1,
            tts_rate = $2,
            tts_volume = $3,
            tts_instruction = $4,
            tts_auto_emotion = $5,
            tts_emotion_scale = $6,
            tts_output_probability = CASE WHEN $8 THEN tts_output_probability ELSE $9::int END,
            updated_at = NOW()
        WHERE id = $7
        """,
        str(profile.get("voice_id") or ""),
        payload.rate,
        payload.volume,
        instruction,
        payload.auto_emotion,
        payload.emotion_scale,
        agent_id,
        keep_probability,
        payload.output_probability,
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
    agent_rows = await db.query_raw(
        "SELECT user_id FROM ai_agents WHERE id = $1 LIMIT 1",
        agent_id,
    )
    if not agent_rows:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        model = await resolve_agent_tts_model()
    except SpeechSynthesisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    emotion, intensity = await resolve_voice_emotion(
        payload.text, payload.emotion, payload.intensity,
        enabled=payload.auto_emotion and payload.emotion_scale > 0,
        detect_missing=True,
    )
    try:
        plan = build_speech_plan(
            payload.text, emotion, intensity,
            instruction=_validate_instruction(payload.instruction),
            enabled=payload.auto_emotion, scale=payload.emotion_scale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        speech = await synthesize_speech(
            text=plan.text,
            voice_id=str(profile.get("voice_id") or ""),
            instruction=plan.instruction,
            rate=payload.rate,
            volume=payload.volume,
            model=model,
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
    return _audio_response(speech)


def _audio_response(speech: SynthesizedSpeech) -> Response:
    return Response(
        content=speech.audio,
        media_type=speech.mime,
        headers={
            "X-TTS-Duration-Milliseconds": str(speech.duration_milliseconds),
            "X-TTS-Billable-Characters": str(speech.billable_characters),
            "X-TTS-Cost-CNY": f"{speech.cost_cny:.6f}",
        },
    )
