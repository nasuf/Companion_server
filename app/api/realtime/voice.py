"""Bidirectional voice calls over a dedicated websocket.

Text chat stays on ``/ws/{conversation_id}``. This endpoint proxies PCM16 to
DashScope's Qwen-Omni realtime model. Server VAD finds the end of each turn;
the bridge fetches memories, then asks for the reply (bridge.py).

Client handshake matches text chat: subprotocols ``companion.chat.v1`` and
``companion.ticket.<ticket>``. Audio is mono little-endian PCM16 at the
sample rate in the ``ready`` event (24 kHz). Send binary frames, or JSON
``{"type":"audio.append","audio":"<base64>"}``.

Other client events: ``interrupt``, ``greet``, ``ping``.
Server events: ``ready``, ``speech.started``, ``speech.stopped``,
``transcript.user``, ``transcript.assistant``, ``audio.delta``, ``audio.done``,
``response.started``, ``response.done``, ``interrupted``, ``warning``,
``error``, ``session.expiring``, ``pong``.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.db import db
from app.redis_client import is_redis_healthy
from app.services.interaction.boundary import get_patience, get_patience_zone
from app.services.realtime_voice.bridge import VoiceBridge
from app.services.realtime_voice.context import load_call_context
from app.services.realtime_voice.instructions import clock_line, memory_lines, turn_voice_hint
from app.services.realtime_voice.lock import acquire_call, refresh_call, release_call
from app.services.realtime_voice.persist import (
    save_assistant_transcript,
    save_user_transcript,
    schedule_end_of_call_extraction,
    schedule_user_present,
)
from app.services.realtime_voice.protocol import SAMPLE_RATE
from app.services.realtime_voice.billing import tokens_by_model
from app.services.realtime_voice.provider import (
    RealtimeSetupError,
    RealtimeSocket,
    configure_session,
    connect_realtime,
    realtime_url,
)
from app.services.runtime_config import get_effective_realtime_model
from app.services.speech_output.voices import ensure_agent_voice
from app.services.runtime.ws_auth import (
    CHAT_PROTOCOL,
    TicketInvalid,
    TicketUnavailable,
    consume_ticket,
    offered_ticket,
    origin_allowed,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime-voice"])

# Max call duration: 28 minutes
_MAX_CALL_SECONDS = 28 * 60
# Warn client 60s before forced disconnect
_EXPIRY_WARNING_SECONDS = 60
# Idle timeout: close if no audio or event for this long
_IDLE_TIMEOUT_SECONDS = 120


@router.websocket("/ws/voice/{conversation_id}")
async def voice_websocket(websocket: WebSocket, conversation_id: str):
    # --- Pre-accept checks ---
    if not is_redis_healthy():
        await websocket.close(code=1011, reason="redis_unavailable")
        return
    try:
        ticket = offered_ticket(websocket.scope.get("subprotocols", []))
        if not origin_allowed(websocket.headers.get("origin")):
            raise TicketInvalid()
        principal = await consume_ticket(ticket, conversation_id)
    except TicketInvalid:
        await websocket.close(code=4401, reason="authentication_required")
        return
    except TicketUnavailable:
        await websocket.close(code=1011, reason="authentication_unavailable")
        return

    conv = await db.conversation.find_unique(
        where={"id": conversation_id},
        include={"agent": True},
    )
    if not conv or conv.isDeleted or not conv.agent:
        await websocket.close(code=4004, reason="conversation_not_found")
        return
    if (principal.role != "admin" and principal.user_id != conv.userId) or conv.agent.userId != conv.userId:
        await websocket.close(code=4403, reason="conversation_access_denied")
        return
    if conv.agent.status == "provisioning":
        await websocket.close(code=1013, reason="agent_initializing")
        return
    try:
        patience = await get_patience(str(conv.agent.id), conv.userId)
    except Exception:
        logger.warning("[realtime-voice] patience lookup failed")
        await websocket.close(code=1011, reason="patience_unavailable")
        return
    if get_patience_zone(patience) == "blocked":
        await websocket.close(code=4403, reason="agent_unavailable")
        return
    if principal.expires_at <= time.time():
        await websocket.close(code=4401, reason="session_expired")
        return
    if not await _accounts_current(principal, conv.userId):
        await websocket.close(code=4401, reason="authentication_required")
        return

    # --- Accept ---
    await websocket.accept(subprotocol=CHAT_PROTOCOL)
    if not settings.dashscope_api_key.strip():
        await _fail(websocket, "not_configured", "实时语音未配置，服务端需要 DASHSCOPE_API_KEY")
        return

    # --- Acquire per-conversation call lock ---
    try:
        token = await acquire_call(conversation_id)
    except Exception:
        logger.warning("[realtime-voice] call lock unavailable")
        await _fail(websocket, "unavailable", "实时语音暂时不可用")
        return
    if token is None:
        await _fail(websocket, "busy", "这个会话已经有一通语音通话")
        return

    bridge: VoiceBridge | None = None
    model = ""
    try:
        logger.info(
            "[realtime-voice] call starting conv=%s user=%s agent=%s platform=%s",
            conversation_id[:8], conv.userId[:8], str(conv.agent.id)[:8], _client_platform(websocket),
        )
        persona, instructions, history, greet_cue = await load_call_context(
            agent=conv.agent,
            user_id=conv.userId,
            workspace_id=getattr(conv, "workspaceId", None),
            conversation_id=conversation_id,
        )
        # The agent's library voice, the same one its voice messages use.
        try:
            voice = await ensure_agent_voice(conv.agent)
        except Exception:
            logger.warning("[realtime-voice] agent voice lookup failed", exc_info=True)
            await _fail(websocket, "not_configured", "音色库没有可用音色")
            return
        model = (await get_effective_realtime_model()) or settings.dashscope_realtime_model
        logger.info("[realtime-voice] using voice=%s model=%s", voice, model)

        async with connect_realtime(realtime_url(model), settings.dashscope_api_key.strip()) as raw:
            sock = RealtimeSocket(raw)
            await configure_session(
                sock,
                instructions=instructions,
                voice=voice,
                history=history,
                silence_duration_ms=settings.dashscope_realtime_silence_ms,
                vad_threshold=settings.dashscope_realtime_vad_threshold,
                temperature=settings.dashscope_realtime_temperature,
            )
            from app.services.realtime_voice.call_stats import CallStats
            bridge = _bridge(
                sock=sock,
                websocket=websocket,
                conversation_id=conversation_id,
                user_id=conv.userId,
                workspace_id=getattr(conv, "workspaceId", None),
                model=model,
                instructions=instructions,
                greet_cue=greet_cue,
                earlier_replies=[t["text"] for t in history if t["role"] == "assistant"],
                stats=CallStats(
                    platform=_client_platform(websocket), model=model,
                    vad_threshold=settings.dashscope_realtime_vad_threshold,
                    silence_ms=settings.dashscope_realtime_silence_ms,
                ),
            )
            await websocket.send_json({
                "type": "ready",
                "sample_rate": SAMPLE_RATE,
                "encoding": "pcm16",
                "channels": 1,
                "voice": voice,
                "model": model,
                "agent_name": persona.name,
            })
            schedule_user_present(conversation_id)
            await _run_pumps(
                websocket=websocket,
                sock=sock,
                bridge=bridge,
                conversation_id=conversation_id,
                token=token,
            )
    except RealtimeSetupError as exc:
        logger.warning("[realtime-voice] setup failed: %s", exc)
        await _fail(websocket, "provider_unavailable", "实时语音暂时连不上")
    except Exception:
        logger.warning("[realtime-voice] call failed", exc_info=True)
        await _fail(websocket, "provider_unavailable", "实时语音中断了")
    finally:
        if bridge is not None:
            n_pairs = len(bridge.transcript_pairs)
            logger.info(
                "[realtime-voice] call ended conv=%s transcript_pairs=%d mic %s",
                conversation_id[:8], n_pairs, bridge.mic_summary,
            )
            await bridge.aclose()
            await _record_call_usage(
                bridge.usage,
                model=model,
                conversation_id=conversation_id,
                agent_id=str(conv.agent.id),
                user_id=conv.userId,
            )
            from app.services.realtime_voice.call_stats import persist as persist_call_stats
            await persist_call_stats(
                bridge.stats, bridge.stats_summary,
                user_id=conv.userId, workspace_id=getattr(conv, "workspaceId", None),
                conversation_id=conversation_id, agent_id=str(conv.agent.id),
            )
            # End-of-call: trigger memory extraction via standard pipeline
            try:
                schedule_end_of_call_extraction(
                    user_id=conv.userId,
                    workspace_id=getattr(conv, "workspaceId", None),
                    conversation_id=conversation_id,
                    agent_id=str(conv.agent.id),
                    message_ids=bridge.message_ids,
                )
                logger.info("[realtime-voice] end-of-call extraction scheduled conv=%s", conversation_id[:8])
            except Exception:
                logger.warning("[realtime-voice] end-of-call extraction scheduling failed")
        try:
            await release_call(conversation_id, token)
        except Exception:
            logger.warning("[realtime-voice] call lock release failed")


async def _record_call_usage(
    usage: dict[str, int],
    *,
    model: str,
    conversation_id: str,
    agent_id: str,
    user_id: str,
) -> None:
    """One llm_usage row per call; text and audio tokens priced separately."""
    if not usage.get("responses"):
        return
    from app.services.llm.usage_repo import persist_usage

    input_tokens = usage["input_text"] + usage["input_audio"]
    output_tokens = usage["output_text"] + usage["output_audio"]
    summary = {
        "tokens_by_model": tokens_by_model(model, usage),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": 0,
        "call_count": usage["responses"],
        "web_search_calls": 0,
        "latency_ms_total": 0,
        "latency_count": 0,
        "failure_count": 0,
        "fallback_count": 0,
        "circuit_open_count": 0,
    }
    logger.info(
        "[realtime-voice] usage conv=%s responses=%d input_text=%d input_audio=%d"
        " output_text=%d output_audio=%d",
        conversation_id[:8], usage["responses"], usage["input_text"],
        usage["input_audio"], usage["output_text"], usage["output_audio"],
    )
    try:
        await persist_usage(
            summary=summary,
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            trace_id=None,
            scope="realtime_voice",
        )
    except Exception:
        logger.warning("[realtime-voice] usage metering failed", exc_info=True)


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def _client_platform(websocket) -> str:
    """The app sends ?platform=android|ios; older builds send nothing."""
    params = getattr(websocket, "query_params", None) or {}
    value = str(params.get("platform") or "").strip().lower()
    return value if value in {"android", "ios"} else "unknown"


_SHORT_FOLLOW_UP_CHARS = 8
_FOLLOW_UP_CONTEXT_CHARS = 60


def _bridge(
    *,
    sock,
    websocket,
    conversation_id,
    user_id,
    workspace_id,
    model,
    instructions: str | None = None,
    greet_cue: str | None = None,
    earlier_replies: list[str] | None = None,
    stats=None,
) -> VoiceBridge:
    from app.services.realtime_voice.call_stats import CallStats

    stats = stats or CallStats()

    async def retrieve(text: str, previous: str | None) -> list[dict[str, str]]:
        from app.services.chat.memory_usage import submit_memory_usage
        from app.services.memory.retrieval.hybrid import hybrid_retrieve

        # Short follow-ups ("那后来呢") name nothing; borrow the topic from
        # the agent's last line for the embedding, keep the user's words for
        # time parsing.
        enhanced = None
        if previous and len(text) <= _SHORT_FOLLOW_UP_CHARS:
            enhanced = f"{previous[-_FOLLOW_UP_CONTEXT_CHARS:]} {text}"
        found = await hybrid_retrieve(
            text,
            user_id,
            workspace_id=workspace_id,
            token_budget=500,
            enhanced_query=enhanced,
        )
        memories = found.get("memories") if isinstance(found, dict) else None
        entries = memory_lines(memories if isinstance(memories, list) else [])
        selected_ids = [entry["id"] for entry in entries if entry["id"]]
        candidates = list(found.get("candidate_ids") or []) if isinstance(found, dict) else []
        stats.retrieval_result(candidates=len(candidates), selected=len(entries))
        if selected_ids or candidates:
            try:
                submit_memory_usage(
                    event_id=None,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    selected_ids=selected_ids,
                    candidate_ids=candidates,
                )
            except Exception:
                logger.debug("[realtime-voice] memory usage not recorded")
        return entries

    async def save_user(text: str) -> str | None:
        return await save_user_transcript(
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            model=model,
            text=text,
        )

    async def save_assistant(text: str, user_message_id: str | None, interrupted: bool) -> str | None:
        return await save_assistant_transcript(
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            model=model,
            text=text,
            user_message_id=user_message_id,
            interrupted=interrupted,
        )

    return VoiceBridge(
        send_provider=sock.send_event,
        send_client=websocket.send_json,
        retrieve_memories=retrieve,
        save_user_line=save_user,
        save_assistant_line=save_assistant,
        retrieval_timeout_s=settings.dashscope_realtime_retrieval_timeout_s,
        memory_wait_s=settings.dashscope_realtime_memory_wait_s,
        instructions=instructions,
        greet_cue=greet_cue,
        earlier_replies=earlier_replies,
        clock_line=clock_line,
        voice_hint=turn_voice_hint,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Pump loop
# ---------------------------------------------------------------------------

async def _run_pumps(*, websocket, sock, bridge, conversation_id: str, token: str) -> None:
    """Run client, provider, refresh and idle-timeout pumps concurrently."""

    last_activity = time.monotonic()

    async def pump_client():
        nonlocal last_activity
        try:
            while True:
                message = await websocket.receive()
                last_activity = time.monotonic()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is not None:
                    await bridge.feed_audio(data)
                    continue
                text = message.get("text")
                if isinstance(text, str):
                    await bridge.feed_client_text(text)
        except WebSocketDisconnect:
            return

    async def pump_provider():
        nonlocal last_activity
        async for event in sock:
            last_activity = time.monotonic()
            await bridge.feed_provider_event(event)

    async def refresh():
        while True:
            await asyncio.sleep(10)
            if not await refresh_call(conversation_id, token):
                await _send(websocket, {"type": "error", "code": "call_lock_lost"})
                return

    async def idle_watchdog():
        """Close if no activity for _IDLE_TIMEOUT_SECONDS."""
        while True:
            await asyncio.sleep(15)
            if time.monotonic() - last_activity > _IDLE_TIMEOUT_SECONDS:
                await _send(websocket, {
                    "type": "warning",
                    "message": "通话因长时间无活动自动结束",
                })
                return

    max_seconds = min(
        settings.dashscope_realtime_max_seconds,
        _MAX_CALL_SECONDS,
    )

    tasks = {
        asyncio.create_task(pump_client()),
        asyncio.create_task(pump_provider()),
        asyncio.create_task(refresh()),
        asyncio.create_task(idle_watchdog()),
    }
    done, pending = await asyncio.wait(
        tasks,
        timeout=max_seconds,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not done:
        # Timed out — send expiry warning
        await _send(websocket, {"type": "session.expiring"})
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            logger.warning("[realtime-voice] pump ended: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _accounts_current(principal, owner_id: str) -> bool:
    owner = await db.user.find_unique(where={"id": owner_id})
    if not owner or owner.status != "active" or owner.archivedAt is not None:
        return False
    if principal.user_id == owner_id:
        return owner.role == principal.role
    actor = await db.user.find_unique(where={"id": principal.user_id})
    return bool(
        actor
        and actor.status == "active"
        and actor.archivedAt is None
        and actor.role == principal.role
    )


async def _fail(websocket: WebSocket, code: str, message: str) -> None:
    await _send(websocket, {"type": "error", "code": code, "message": message})
    try:
        await websocket.close(code=1011)
    except RuntimeError:
        pass


async def _send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except RuntimeError:
        pass
