"""Voice call bridge: client audio ↔ StepFun provider ↔ memory tools.

With server_vad enabled, StepFun handles turn detection, barge-in and
response triggering natively. This bridge:
  1. Forwards client PCM to the provider
  2. Handles function_call events (recall_memory tool)
  3. Prefetches memories in the background on each user transcript
  4. Persists transcripts (extraction deferred to call end)
  5. Relays audio deltas and status events to the client
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable

from app.services.realtime_voice.protocol import (
    append_audio_event,
    assistant_transcript,
    decode_client_pcm,
    function_call_output_event,
    parse_client_message,
    response_cancel_event,
    response_create_event,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------
SendProvider = Callable[[dict], Awaitable[None]]
SendClient = Callable[[dict], Awaitable[None]]
RetrieveMemories = Callable[[str], Awaitable[str | None]]
SaveUserLine = Callable[[str], Awaitable[str | None]]
SaveAssistantLine = Callable[[str, str | None, bool], Awaitable[None]]

# Max number of memory prefetch results to cache
_PREFETCH_CACHE_SIZE = 3


class VoiceBridge:
    """Stateful bridge for one voice call.

    Thread safety: all public methods acquire ``_lock`` so concurrent
    calls from the client pump and provider pump are serialized.
    """

    def __init__(
        self,
        *,
        send_provider: SendProvider,
        send_client: SendClient,
        retrieve_memories: RetrieveMemories,
        save_user_line: SaveUserLine,
        save_assistant_line: SaveAssistantLine,
        retrieval_timeout_s: float = 2.0,
    ) -> None:
        self._send_provider = send_provider
        self._send_client = send_client
        self._retrieve_memories = retrieve_memories
        self._save_user_line = save_user_line
        self._save_assistant_line = save_assistant_line
        self._retrieval_timeout_s = retrieval_timeout_s

        self._lock = asyncio.Lock()
        self._closed = False

        # Transcript tracking
        self._user_text = ""
        self._user_message_id: str | None = None
        self._assistant_parts: list[str] = []

        # Audio output tracking
        self._audio_seq = 0
        self._output_active = False
        self._responding = False
        # Set on barge-in: audio and the final transcript of the interrupted
        # response are dropped until the provider starts the next one.
        self._discarding = False

        # Function call accumulation
        self._pending_calls: dict[str, dict] = {}  # call_id -> {name, arguments}

        # Memory prefetch cache: most recent query result
        self._prefetch_result: str | None = None
        self._prefetch_task: asyncio.Task | None = None
        self._prefetch_query: str = ""

        # Collected transcripts for end-of-call extraction
        self._transcript_pairs: list[dict] = []

        # Provider token usage, summed over every response.done
        self._usage = {"input": 0, "output": 0, "cached_input": 0, "responses": 0}

    # ------------------------------------------------------------------
    # Public entry points (all acquire lock)
    # ------------------------------------------------------------------

    async def feed_audio(self, pcm: bytes) -> None:
        """Forward raw PCM from client to provider."""
        async with self._lock:
            await self._feed_audio(pcm)

    async def feed_client_text(self, text: str) -> None:
        """Handle a JSON text frame from the client."""
        async with self._lock:
            await self._feed_client_text(text)

    async def feed_provider_event(self, event: dict) -> None:
        """Handle an event from the StepFun provider."""
        async with self._lock:
            await self._feed_provider_event(event)

    async def aclose(self) -> None:
        """Clean up on call end."""
        if self._closed:
            return
        self._closed = True
        self._cancel_prefetch()

    @property
    def transcript_pairs(self) -> list[dict]:
        """Collected user/assistant transcript pairs for end-of-call extraction."""
        return list(self._transcript_pairs)

    @property
    def usage(self) -> dict[str, int]:
        """Token usage StepFun reported for this call (billed per token)."""
        return dict(self._usage)

    # ------------------------------------------------------------------
    # Audio forwarding (server_vad handles turn detection)
    # ------------------------------------------------------------------

    async def _feed_audio(self, pcm: bytes) -> None:
        if self._closed:
            return
        pcm = decode_client_pcm(pcm) or b""
        if not pcm:
            return
        await self._send_provider(append_audio_event(pcm))

    # ------------------------------------------------------------------
    # Client text messages
    # ------------------------------------------------------------------

    async def _feed_client_text(self, text: str) -> None:
        if self._closed:
            return
        payload = parse_client_message(text)
        if payload is None:
            return
        kind = payload["type"]
        if kind == "ping":
            await self._send_client({"type": "pong"})
            return
        if kind == "audio.append":
            pcm = decode_client_pcm(payload.get("audio"))
            if pcm:
                await self._feed_audio(pcm)
            return
        if kind == "interrupt":
            await self._interrupt()
            return
        if kind == "greet":
            await self._greet()
            return

    # ------------------------------------------------------------------
    # Provider event dispatch
    # ------------------------------------------------------------------

    async def _feed_provider_event(self, event: dict) -> None:
        if self._closed or not isinstance(event, dict):
            return
        kind = event.get("type")

        # --- VAD events ---
        if kind == "input_audio_buffer.speech_started":
            await self._on_speech_started()
            return
        if kind == "input_audio_buffer.speech_stopped":
            await self._on_speech_stopped()
            return

        # --- User transcript ---
        if kind == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript")
            text = transcript.strip() if isinstance(transcript, str) else ""
            await self._on_user_transcript(text)
            return

        # --- Response lifecycle ---
        if kind == "response.created":
            self._responding = True
            self._discarding = False
            return
        if kind == "response.audio.delta":
            await self._on_audio_delta(event.get("delta"))
            return
        if kind == "response.audio_transcript.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta and not self._discarding:
                self._assistant_parts.append(delta)
            return
        if kind == "response.audio_transcript.done":
            await self._on_assistant_transcript_done(event)
            return
        if kind == "response.audio.done":
            if self._output_active:
                self._output_active = False
                await self._send_client({"type": "audio.done"})
            return
        if kind == "response.done":
            await self._on_response_done(event)
            return

        # --- Function call events ---
        if kind == "conversation.item.created":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id") or ""
                name = item.get("name") or ""
                if call_id:
                    self._pending_calls[call_id] = {"name": name, "arguments": ""}
            return
        if kind == "response.function_call_arguments.delta":
            call_id = event.get("call_id") or ""
            delta = event.get("delta") or ""
            if call_id in self._pending_calls:
                self._pending_calls[call_id]["arguments"] += delta
            return
        if kind == "response.function_call_arguments.done":
            call_id = event.get("call_id") or ""
            args_str = event.get("arguments") or ""
            if call_id in self._pending_calls:
                self._pending_calls[call_id]["arguments"] = args_str
                await self._execute_tool_call(call_id)
            return

        # --- Errors ---
        if kind == "error":
            await self._on_provider_error(event)
            return

    # ------------------------------------------------------------------
    # VAD callbacks (from StepFun server_vad)
    # ------------------------------------------------------------------

    async def _on_speech_started(self) -> None:
        """Barge-in: stop the reply now and let the user finish.

        The client drops its queued audio on speech.started. A reply still
        being generated is cancelled here and its remaining audio is not
        forwarded; one that already finished needs nothing server-side.
        """
        interrupted = self._output_active or self._responding
        cancelled = False
        if self._responding:
            try:
                await self._send_provider(response_cancel_event())
                cancelled = True
            except Exception:
                logger.warning("[realtime-voice] cancel on barge-in failed")
        # One line per detected user turn: shows whether VAD fired while the
        # agent was talking and whether generation had to be cancelled.
        logger.info(
            "[realtime-voice] speech started generating=%s cancelled=%s",
            interrupted, cancelled,
        )
        if interrupted:
            self._discarding = True
        self._output_active = False
        self._responding = False
        # Save partial assistant transcript if interrupted
        partial = "".join(self._assistant_parts).strip()
        if partial and interrupted:
            try:
                await self._save_assistant_line(partial, self._user_message_id, True)
                self._collect_transcript("assistant", partial, interrupted=True)
            except Exception:
                logger.warning("[realtime-voice] interrupted transcript not stored")
            self._assistant_parts = []
        await self._send_client({"type": "speech.started"})

    async def _on_speech_stopped(self) -> None:
        """User stopped speaking — StepFun will auto-commit and start response."""
        await self._send_client({"type": "speech.stopped"})

    # ------------------------------------------------------------------
    # Transcript handling
    # ------------------------------------------------------------------

    async def _on_user_transcript(self, text: str) -> None:
        """User's speech was transcribed. Save it and prefetch memories."""
        self._user_text = text
        self._user_message_id = None
        if text:
            logger.info("[realtime-voice] user transcript len=%d", len(text))
            try:
                self._user_message_id = await self._save_user_line(text)
            except Exception:
                logger.warning("[realtime-voice] user transcript not stored")
            self._collect_transcript("user", text)
            # Prefetch memories in background for potential tool call
            self._start_prefetch(text)
        if not self._closed:
            await self._send_client({
                "type": "transcript.user",
                "text": text,
                "message_id": self._user_message_id,
            })

    async def _on_assistant_transcript_done(self, event: dict) -> None:
        """Full transcript of one assistant audio segment."""
        text = assistant_transcript(event) or "".join(self._assistant_parts).strip()
        if not text:
            return
        # Don't overwrite if we already have the text from delta accumulation
        self._assistant_parts = []

    async def _on_response_done(self, event: dict) -> None:
        """Response finished — save complete assistant transcript."""
        self._responding = False
        self._output_active = False
        text = assistant_transcript(event) or "".join(self._assistant_parts).strip()
        self._assistant_parts = []
        if self._discarding:
            # The barge-in already stored what was said before the cut.
            self._discarding = False
            text = ""

        status = ""
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("status"), str):
            status = response["status"]
        interrupted = status in {"cancelled", "incomplete"}
        self._record_usage(response)

        if text:
            logger.info(
                "[realtime-voice] assistant transcript len=%d interrupted=%s",
                len(text), interrupted,
            )
            try:
                await self._save_assistant_line(text, self._user_message_id, interrupted)
                self._collect_transcript("assistant", text, interrupted=interrupted)
            except Exception:
                logger.warning("[realtime-voice] assistant transcript not stored")
            if not self._closed:
                await self._send_client({
                    "type": "transcript.assistant",
                    "text": text,
                    "interrupted": interrupted,
                })

        await self._send_client({"type": "response.done", "interrupted": interrupted})

    def _record_usage(self, response: object) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return
        details = usage.get("input_token_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else 0
        for key, value in (
            ("input", usage.get("input_tokens")),
            ("output", usage.get("output_tokens")),
            ("cached_input", cached),
        ):
            if isinstance(value, int) and value > 0:
                self._usage[key] += value
        self._usage["responses"] += 1

    # ------------------------------------------------------------------
    # Audio relay
    # ------------------------------------------------------------------

    async def _on_audio_delta(self, delta: object) -> None:
        if self._discarding or not isinstance(delta, str) or not delta:
            return
        try:
            base64.b64decode(delta, validate=True)
        except Exception:
            return
        if not self._output_active:
            self._output_active = True
            self._audio_seq = 0
            await self._send_client({"type": "response.started"})
        self._audio_seq += 1
        await self._send_client({
            "type": "audio.delta",
            "seq": self._audio_seq,
            "audio": delta,
        })

    # ------------------------------------------------------------------
    # Function call execution (recall_memory tool)
    # ------------------------------------------------------------------

    async def _execute_tool_call(self, call_id: str) -> None:
        """Execute a pending tool call and return result to provider."""
        call_info = self._pending_calls.pop(call_id, None)
        if not call_info:
            return

        name = call_info["name"]
        args_str = call_info["arguments"]
        logger.info("[realtime-voice] tool_call name=%s call_id=%s", name, call_id[:16])

        if name != "recall_memory":
            # Unknown tool — return error to model
            await self._send_provider(function_call_output_event(
                call_id=call_id,
                output=json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False),
            ))
            await self._send_provider(response_create_event())
            return

        # Parse the query argument
        query = ""
        try:
            args = json.loads(args_str) if args_str else {}
            query = str(args.get("query") or "")
        except (json.JSONDecodeError, TypeError):
            pass

        # Try prefetch cache first, then fetch
        result = await self._get_memories(query)
        output = result or "没有找到相关记忆。"

        await self._send_provider(function_call_output_event(
            call_id=call_id,
            output=output,
        ))
        await self._send_provider(response_create_event())

    async def _get_memories(self, query: str) -> str | None:
        """Get memories: use prefetch cache if available, else fetch live."""
        # If prefetch is running for a similar query, wait for it
        if self._prefetch_task and not self._prefetch_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._prefetch_task), timeout=self._retrieval_timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("[realtime-voice] prefetch wait timed out")
            except Exception:
                pass

        # Use prefetch result if query is close enough
        if self._prefetch_result:
            result = self._prefetch_result
            self._prefetch_result = None
            return result

        # Fresh retrieval
        try:
            return await asyncio.wait_for(
                self._retrieve_memories(query),
                timeout=self._retrieval_timeout_s,
            )
        except TimeoutError:
            logger.warning("[realtime-voice] memory retrieval timed out")
            return None
        except Exception:
            logger.warning("[realtime-voice] memory retrieval failed")
            return None

    # ------------------------------------------------------------------
    # Memory prefetch (background, on user transcript)
    # ------------------------------------------------------------------

    def _start_prefetch(self, query: str) -> None:
        """Start background memory retrieval so tool calls return faster."""
        self._cancel_prefetch()
        self._prefetch_query = query
        self._prefetch_result = None
        self._prefetch_task = asyncio.create_task(self._do_prefetch(query))
        self._prefetch_task.add_done_callback(self._prefetch_done)

    async def _do_prefetch(self, query: str) -> None:
        try:
            result = await asyncio.wait_for(
                self._retrieve_memories(query),
                timeout=self._retrieval_timeout_s,
            )
            self._prefetch_result = result
        except (TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            logger.debug("[realtime-voice] prefetch failed")

    def _prefetch_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug("[realtime-voice] prefetch task error: %s", exc)

    def _cancel_prefetch(self) -> None:
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetch_task = None

    # ------------------------------------------------------------------
    # Client-initiated actions
    # ------------------------------------------------------------------

    async def _interrupt(self) -> None:
        """Client requests interruption."""
        if self._responding or self._output_active:
            try:
                await self._send_provider(response_cancel_event())
            except Exception:
                logger.warning("[realtime-voice] cancel failed")
            self._output_active = False
            self._responding = False
            await self._send_client({"type": "interrupted"})

    async def _greet(self) -> None:
        """Client requests an opening greeting."""
        if self._responding or self._output_active:
            return
        await self._send_provider(response_create_event())

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    async def _on_provider_error(self, event: dict) -> None:
        error = event.get("error")
        message = ""
        if isinstance(error, dict):
            raw = error.get("message")
            if isinstance(raw, str):
                message = raw[:180]
        await self._send_client({
            "type": "warning",
            "message": message or "语音服务返回了一个错误",
        })

    # ------------------------------------------------------------------
    # Transcript collection (for end-of-call extraction)
    # ------------------------------------------------------------------

    def _collect_transcript(self, role: str, text: str, *, interrupted: bool = False) -> None:
        """Record a transcript pair for batch extraction at call end."""
        self._transcript_pairs.append({
            "role": role,
            "text": text,
            "interrupted": interrupted,
            "ts": time.time(),
        })
