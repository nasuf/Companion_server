"""Voice call bridge: client audio ↔ DashScope Qwen-Omni realtime ↔ memory.

Server VAD finds the end of each user turn but neither replies nor cancels
on its own (see protocol.py). For every turn the bridge:
  1. starts memory retrieval as soon as the user stops talking, from the
     live transcript;
  2. drops turns that are only the agent's own voice leaking back into the
     microphone, or a one-word backchannel while the agent is talking;
  3. waits briefly for the retrieval, sends this turn's memories as a system
     note (none for small talk such as "喂喂" or "嗯"), asks for the reply,
     and deletes the note once the reply is done;
  4. persists transcripts (extraction runs at call end) and relays audio.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import time
from array import array
from collections import deque
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher

from app.services.realtime_voice.billing import split_usage
from app.services.realtime_voice.call_stats import CallStats
from app.services.realtime_voice.instructions import format_turn_memories, is_small_talk, spoken_key
from app.services.realtime_voice.memory_slot import MemorySlot
from app.services.realtime_voice.protocol import (
    SAMPLE_RATE,
    append_audio_event,
    assistant_transcript,
    decode_client_pcm,
    function_call_output_event,
    history_item_event,
    item_delete_event,
    parse_client_message,
    response_cancel_event,
    response_create_event,
    system_note_event,
)

logger = logging.getLogger(__name__)

# A bare response.create gives no greeting to answer, so a one-off user cue
# makes the agent open the call; it is deleted once the greeting is done.
# The endpoint passes a cue that knows how long ago the two last talked.
_GREET_CUE = "（用户给你打来电话，你接起来先开口，简短自然地打个招呼。）"

# A reply this close to one of the last few is a repeat. The 2026-11-05 call
# repeated a 43-character line differing by one character (ratio 0.99);
# short acknowledgements ("对呀") are allowed to recur.
_REPEAT_WINDOW = 3
_REPEAT_RATIO = 0.8
_REPEAT_MIN_CHARS = 8

# Server VAD hears the agent's own voice leaking back from the speaker at any
# threshold (2026-11-06), so speech that starts while the agent is audible is
# held until its transcript shows whether it is the user or the echo. The
# client plays audio after it is generated: playback is estimated from the
# audio sent (plus the client's start-up buffer), and held for this tail.
_CLIENT_LEAD_S = 0.25
_ECHO_TAIL_S = 0.5
# speech_started is placed on the local clock from its audio_start_ms; a
# result further back than this means the provider counts differently.
_MAX_VAD_LAG_S = 3.0
# Echo cannot precede the agent's first audio; the speech start is only an
# estimate, so the audible window opens this much earlier.
_VAD_SLACK_S = 0.3
# Share of a transcript's characters found, in order, in what the agent just
# said. Recognition of echo swaps homophones ("喂，枫，你找我啊" came back as
# "风，你找我啊", 0.8; 2026-11-06), so the bar sits below that.
_ECHO_MATCH = 0.75
# In the tail, only a transcript this long can be echo: a quick "吃了" right
# after "你吃了吗" is an answer.
_TAIL_ECHO_MIN_CHARS = 4
# Characters of live transcript that confirm the user talks over the agent.
# Echo from the speaker is often garbled ("喂，枫" heard as "外封",
# 2026-11-06), so a short fragment that matches nothing proves nothing.
_BARGE_IN_MIN_CHARS = 4
# A requested reply normally starts within a second.
_REPLY_START_TIMEOUT_S = 8.0
# Longest stretch of user speech before the segment is ended without a stop.
_SEGMENT_MAX_S = 60.0
# Short words that do stop the agent: the user wants the floor. Other short
# speech over the agent ("嗯", "对", "哈哈") is a backchannel or echo.
_STOP_WORDS = frozenset({
    "等等", "等一下", "等下", "等会", "等会儿", "停", "停一下", "停停", "打住", "别说了",
    "不是", "不对", "什么", "慢着",
})
# A final transcript this close to its live preview keeps the retrieval
# started from the preview.
_SAME_QUERY_RATIO = 0.8

SendProvider = Callable[[dict], Awaitable[None]]
SendClient = Callable[[dict], Awaitable[None]]
# (query, previous assistant line) -> memory_lines() entries
RetrieveMemories = Callable[[str, str | None], Awaitable[list[dict[str, str]]]]
SaveUserLine = Callable[[str], Awaitable[str | None]]
SaveAssistantLine = Callable[[str, str | None, bool], Awaitable[str | None]]

def _echo_share(heard: str, said: str) -> float:
    """Share of ``heard`` that appears, in order, in ``said`` (both keys)."""
    if not heard or not said:
        return 0.0
    matcher = SequenceMatcher(None, heard, said, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(heard)


class _Segment:
    """One stretch of microphone speech found by server VAD."""

    __slots__ = ("item_id", "mid_reply", "surfaced", "live", "query", "done", "stopped_at")

    def __init__(self, item_id: str | None, *, held: bool, mid_reply: bool = False) -> None:
        self.item_id = item_id
        # Started while the agent was playing (not just in the tail after it).
        self.mid_reply = mid_reply
        # The client was told (speech.started) and has stopped playback. Held
        # segments (started while the agent was audible) wait for their live
        # transcript: they may be the agent's own voice in the microphone.
        self.surfaced = not held
        self.live = ""
        # Text the memory retrieval for this segment was started with.
        self.query = ""
        self.done = False
        # When server VAD reported the end of this speech (monotonic seconds).
        self.stopped_at: float | None = None


class VoiceBridge:
    """Stateful bridge for one voice call.

    Thread safety: all public methods acquire ``_lock`` so concurrent
    calls from the client pump and provider pump are serialized. Waiting
    for memories happens in a task outside the lock, so audio keeps flowing.
    """

    def __init__(
        self,
        *,
        send_provider: SendProvider,
        send_client: SendClient,
        retrieve_memories: RetrieveMemories,
        save_user_line: SaveUserLine,
        save_assistant_line: SaveAssistantLine,
        retrieval_timeout_s: float = 1.5,
        memory_wait_s: float = 0.6,
        instructions: str | None = None,
        greet_cue: str | None = None,
        clock_line: Callable[[], str] | None = None,
        stats: CallStats | None = None,
    ) -> None:
        self._send_provider = send_provider
        self._send_client = send_client
        self._retrieve_memories = retrieve_memories
        self._save_user_line = save_user_line
        self._save_assistant_line = save_assistant_line
        self._retrieval_timeout_s = retrieval_timeout_s
        self._memory_wait_s = memory_wait_s
        # None: replies carry no memory note (tests, tools only).
        self._slot = MemorySlot(instructions) if instructions else None
        # Current time for each reply's note; the call instructions only
        # carry the hour at connect time.
        self._clock_line = clock_line
        # Per-call diagnostics (VOICE01.01); the endpoint stores them at call end.
        self.stats = stats or CallStats()
        self._turn = 0
        # The turn being answered is only small talk: its reply gets no memories.
        self._quiet_turn = False
        self._last_assistant_text = ""
        self._recent_replies: deque[str] = deque(maxlen=_REPEAT_WINDOW)

        self._lock = asyncio.Lock()
        self._closed = False

        # Transcript tracking: speech segments by provider item id
        self._segments: dict[str, _Segment] = {}
        # Ended segments: a transcript arriving after the watchdog gave up
        # on it must not become a second turn.
        self._finished: deque[str] = deque(maxlen=32)
        self._watchdogs: set[asyncio.Task] = set()
        self._user_message_id: str | None = None
        self._assistant_parts: list[str] = []

        # Reply and audio output tracking
        self._audio_seq = 0
        self._output_active = False
        # A reply was requested and has not finished (covers the gap before
        # response.created, so a second request is never sent alongside).
        self._responding = False
        # A user turn was committed and has not been answered yet; the reply
        # goes out once this turn's memories are in (or the wait ran out).
        self._unanswered = False
        self._memory_ready = False
        # recall_memory is running between a reply and its follow-up.
        self._tool_running = False
        # response.created seen for the requested reply; an error before it
        # means the reply never started.
        self._reply_seen = False
        # Waits for this turn's memories, then asks for the reply.
        self._reply_task: asyncio.Task | None = None
        # Runs recall_memory between a tool call and its follow-up reply.
        self._tool_task: asyncio.Task | None = None
        # Estimated client playback of the latest reply (monotonic seconds).
        self._playback_started = 0.0
        self._playback_until = 0.0
        # Local time of the first microphone audio sent to the provider.
        self._mic_t0: float | None = None
        # Set on barge-in: audio and the final transcript of the interrupted
        # response are dropped until the provider starts the next one.
        self._discarding = False

        # Per-reply system note: text sent, then its provider item id.
        self._note_text: str | None = None
        self._note_id: str | None = None
        # event_id of the pending response.create; a reply that has not
        # started within _REPLY_START_TIMEOUT_S is released (errors are not
        # matched to requests: DashScope omits event_id on most of them).
        self._create_event_id: str | None = None
        # Barge-in before response.created: DashScope rejects that cancel,
        # so it is sent once the reply has started.
        self._cancel_pending = False
        # Item deletes wait while a reply is generating: deleting an item it
        # builds on closes the session ("previous item id does not exist").
        self._pending_deletes: list[tuple[str, str]] = []

        # Greeting cue: sent, then its provider item id once created
        self._greet_cue = greet_cue or _GREET_CUE
        self._greet_cue_sent = False
        self._greet_cue_id: str | None = None

        # Tool calls of the current response, run once it is done
        self._ready_calls: list[dict] = []

        # Most recent retrieval, for the recall_memory fallback
        self._prefetch_result: list[dict[str, str]] | None = None
        self._prefetch_task: asyncio.Task | None = None

        # Collected transcripts for end-of-call extraction
        self._transcript_pairs: list[dict] = []
        # Persisted message ids, in order: end-of-call extraction binds to them.
        self._message_ids: list[str] = []

        # Microphone input seen by the server, for diagnosing silent calls
        self._mic = _MicMeter()
        self._speech_turns = 0
        self._echo_dropped = 0
        self._memory_skipped = 0  # small-talk turns that injected no memories

        # Provider token usage per modality, summed over every response.done
        self._usage = {
            "input_text": 0, "input_audio": 0,
            "output_text": 0, "output_audio": 0,
            "responses": 0,
        }

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
        """Handle an event from the DashScope realtime session."""
        async with self._lock:
            await self._feed_provider_event(event)

    async def aclose(self) -> None:
        """Clean up on call end."""
        if self._closed:
            return
        self._closed = True
        self._cancel_prefetch()
        for task in (self._reply_task, self._tool_task, *self._watchdogs):
            if task is not None and not task.done():
                task.cancel()

    @property
    def message_ids(self) -> list[str]:
        """Ids of the transcript rows this call stored, oldest first."""
        return list(self._message_ids)

    @property
    def transcript_pairs(self) -> list[dict]:
        """Collected user/assistant transcript pairs for end-of-call extraction."""
        return list(self._transcript_pairs)

    @property
    def stats_summary(self) -> dict:
        """This call's diagnostics, for the realtime_call_sessions row."""
        return self.stats.summary(
            turns=self._turn, speech_turns=self._speech_turns, echo_dropped=self._echo_dropped,
            memory_skipped=self._memory_skipped, mic=self._mic.as_dict(), usage=dict(self._usage),
        )

    @property
    def mic_summary(self) -> str:
        """One log line: how much microphone audio arrived and how loud."""
        return (
            f"{self._mic.summary()} speech_turns={self._speech_turns}"
            f" echo_dropped={self._echo_dropped} memory_skipped={self._memory_skipped}"
        )

    @property
    def usage(self) -> dict[str, int]:
        """Token usage the provider reported for this call, per modality."""
        return dict(self._usage)

    # ------------------------------------------------------------------
    # Audio forwarding
    # ------------------------------------------------------------------

    async def _feed_audio(self, pcm: bytes) -> None:
        if self._closed:
            return
        pcm = decode_client_pcm(pcm) or b""
        if not pcm:
            return
        if self._mic_t0 is None:
            self._mic_t0 = time.monotonic()
        if self._mic.add(pcm):
            logger.info("[realtime-voice] mic first 3s %s", self._mic.summary())
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

        # --- User speech (server VAD + live transcription) ---
        if kind == "input_audio_buffer.speech_started":
            await self._on_speech_started(event)
            return
        if kind == "conversation.item.input_audio_transcription.delta":
            await self._on_transcription_delta(event)
            return
        if kind == "input_audio_buffer.speech_stopped":
            await self._on_speech_stopped(event)
            return
        if kind == "conversation.item.input_audio_transcription.completed":
            await self._on_user_transcript(event)
            return

        # --- Response lifecycle ---
        if kind == "response.created":
            # Replies are only ever requested by this bridge; mark defensively.
            self._responding = True
            self._reply_seen = True
            if self._cancel_pending:
                self._cancel_pending = False
                await self._send_cancel()
            return
        if kind == "response.audio.delta":
            await self._on_audio_delta(event.get("delta"))
            return
        if kind == "response.audio_transcript.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta and not self._discarding:
                self._assistant_parts.append(delta)
            return
        if kind == "response.audio.done":
            if self._output_active:
                self._output_active = False
                await self._send_client({"type": "audio.done"})
            return
        if kind == "response.function_call_arguments.done":
            if not self._discarding and event.get("call_id"):
                self._ready_calls.append({
                    "call_id": str(event["call_id"]),
                    "name": str(event.get("name") or ""),
                    "arguments": str(event.get("arguments") or ""),
                })
            return
        if kind == "response.done":
            await self._on_response_done(event)
            return

        # --- Items we add for one reply: greeting cue, memory note ---
        if kind == "conversation.item.created":
            await self._on_item_created(event.get("item"))
            return

        if kind == "error":
            await self._on_provider_error(event)
            return

    def _speech_time(self, event: dict) -> float:
        """When the speech began, from the provider's own audio timeline.

        speech_started can arrive over a second after the speech it reports
        (2026-11-06); the client streams in real time, so audio_start_ms
        from the first audio sent places it on the local clock.
        """
        now = time.monotonic()
        start_ms = event.get("audio_start_ms")
        if self._mic_t0 is None or not isinstance(start_ms, (int, float)) or start_ms < 0:
            return now
        at = self._mic_t0 + start_ms / 1000
        # Outside a plausible lag the timeline is not the one assumed here.
        return at if now - _MAX_VAD_LAG_S <= at <= now else now

    def _agent_state(self, at: float | None = None) -> tuple[bool, bool]:
        """(playing, in_tail) at ``at``: generating or playing, or just finished."""
        now = time.monotonic() if at is None else at
        started = self._playback_started
        playing = bool(started) and started - _VAD_SLACK_S <= now < self._playback_until
        in_tail = not playing and self._playback_until <= now < self._playback_until + _ECHO_TAIL_S
        return playing, in_tail

    def _agent_key(self) -> str:
        # Echo is of what is playing: the reply in progress or the one before.
        last = self._recent_replies[-1] if self._recent_replies else ""
        return last + spoken_key("".join(self._assistant_parts))

    # ------------------------------------------------------------------
    # User speech: barge-in, echo and backchannel filtering
    # ------------------------------------------------------------------

    def _segment_for(self, event: dict) -> _Segment | None:
        item_id = event.get("item_id")
        return self._segments.get(item_id) if isinstance(item_id, str) else None

    def _user_speaking(self) -> bool:
        return any(seg.surfaced and not seg.done for seg in self._segments.values())

    async def _on_speech_started(self, event: dict) -> None:
        item_id = event.get("item_id") if isinstance(event.get("item_id"), str) else None
        playing, in_tail = self._agent_state(self._speech_time(event))
        segment = _Segment(item_id, held=playing or in_tail, mid_reply=playing)
        if item_id:
            self._segments[item_id] = segment
            # Bounds a segment whose speech_stopped never comes, which would
            # otherwise hold every later reply.
            self._watch_segment(segment, after_s=_SEGMENT_MAX_S)
        self._speech_turns += 1
        if segment.surfaced:
            if self._responding:
                # The user talks before the reply is audible: stop it.
                await self._barge_in(segment)
            else:
                # A pending reply waits until this segment ends
                # (_maybe_reply): more words join the turn.
                await self._send_client({"type": "speech.started"})
        # Speech over the agent is held: it may be the agent's own voice
        # coming back through the microphone. Live transcription decides.

    async def _on_transcription_delta(self, event: dict) -> None:
        segment = self._segment_for(event)
        if segment is None:
            return
        segment.live = f"{event.get('text') or ''}{event.get('stash') or ''}"
        if segment.surfaced:
            return
        key = spoken_key(segment.live)
        if key in _STOP_WORDS or (
            len(key) >= _BARGE_IN_MIN_CHARS and _echo_share(key, self._agent_key()) < _ECHO_MATCH
        ):
            await self._barge_in(segment)

    async def _barge_in(self, segment: _Segment) -> None:
        """The user talks over the agent: stop the reply and let them finish."""
        segment.surfaced = True
        if self._responding or self._output_active:
            self.stats.barge_ins += 1
        generating = self._responding
        cancelled = await self._cut_reply() if generating else False
        self._output_active = False
        self._playback_started = self._playback_until = 0.0
        logger.info(
            "[realtime-voice] speech started generating=%s cancelled=%s",
            generating, cancelled,
        )
        await self._send_client({"type": "speech.started"})

    async def _cut_reply(self) -> bool:
        """Cancel the reply in progress and store what was said before the cut."""
        cancelled = False
        if self._responding and not self._reply_seen:
            self._cancel_pending = True
        else:
            cancelled = await self._send_cancel()
        self._discarding = True
        partial = "".join(self._assistant_parts).strip()
        if partial:
            await self._store_assistant(partial, interrupted=True)
        self._assistant_parts = []
        return cancelled

    async def _send_cancel(self) -> bool:
        try:
            await self._send_provider(response_cancel_event())
            return True
        except Exception:
            logger.warning("[realtime-voice] cancel failed")
            return False

    async def _on_speech_stopped(self, event: dict) -> None:
        segment = self._segment_for(event)
        if segment is None:
            return
        segment.stopped_at = time.monotonic()
        self._watch_segment(segment)
        if not segment.surfaced:
            return
        await self._send_client({"type": "speech.stopped"})
        # Retrieval starts now, from the live transcript; the final one
        # arrives about 0.2 s later and restarts it only if it differs.
        if spoken_key(segment.live) and self._start_prefetch(segment.live):
            segment.query = segment.live

    def _watch_segment(self, segment: _Segment, *, after_s: float = 4) -> None:
        """A stopped segment whose transcript never arrives still ends."""
        async def watch() -> None:
            await asyncio.sleep(after_s)
            async with self._lock:
                if not segment.done and segment.item_id in self._segments:
                    logger.warning("[realtime-voice] transcript missing; using live text")
                    await self._finish_segment(segment, segment.live.strip())

        task = asyncio.create_task(watch())
        self._watchdogs.add(task)
        task.add_done_callback(self._watchdogs.discard)

    async def _on_user_transcript(self, event: dict) -> None:
        transcript = event.get("transcript")
        text = transcript.strip() if isinstance(transcript, str) else ""
        if event.get("item_id") in self._finished:
            return
        segment = self._segment_for(event)
        if segment is None:
            # No speech_started seen for it: treat as an ordinary user turn.
            item_id = event.get("item_id") if isinstance(event.get("item_id"), str) else None
            segment = _Segment(item_id, held=False)
        await self._finish_segment(segment, text)

    async def _finish_segment(self, segment: _Segment, text: str) -> None:
        """The provider's final transcript for one stretch of speech."""
        if segment.done:
            return
        segment.done = True
        if segment.item_id:
            self._segments.pop(segment.item_id, None)
            self._finished.append(segment.item_id)
        key = spoken_key(text)
        if not segment.surfaced:
            verdict = self._held_verdict(segment, key)
            if verdict != "user":
                if verdict == "echo" and key:
                    self._echo_dropped += 1
                logger.info("[realtime-voice] %s dropped len=%d", verdict, len(key))
                await self._discard_item(segment.item_id)
                await self._maybe_reply()
                return
            await self._barge_in(segment)
        if not key:
            await self._discard_item(segment.item_id)
            await self._maybe_reply()
            return

        logger.info("[realtime-voice] user transcript len=%d", len(text))
        self._user_message_id = None
        try:
            self._user_message_id = await self._save_user_line(text)
            self._remember_id(self._user_message_id)
        except Exception:
            logger.warning("[realtime-voice] user transcript not stored")
        self._collect_transcript("user", text)
        self._turn += 1
        self.stats.turn_ended(segment.stopped_at)
        started = spoken_key(segment.query)
        if not started or SequenceMatcher(None, started, key).ratio() < _SAME_QUERY_RATIO:
            self._start_prefetch(text)
        # Small talk after a turn still waiting for its reply keeps that turn's memories.
        self._quiet_turn = is_small_talk(text) and not self._unanswered
        if self._quiet_turn:
            self._memory_skipped += 1
        self._unanswered = True
        self._memory_ready = False
        self._reply_task = asyncio.create_task(self._reply_when_ready(self._turn))
        if not self._closed:
            await self._send_client({
                "type": "transcript.user",
                "text": text,
                "message_id": self._user_message_id,
            })

    def _held_verdict(self, segment: _Segment, key: str) -> str:
        """echo, backchannel or user, for speech that began over the agent."""
        if not key:
            return "echo"
        if segment.mid_reply:
            if key in _STOP_WORDS:
                return "user"
            if len(key) < _BARGE_IN_MIN_CHARS:
                return "backchannel"
            return "echo" if _echo_share(key, self._agent_key()) >= _ECHO_MATCH else "user"
        # Tail after playback: the reply has finished, so compare with it.
        last = self._recent_replies[-1] if self._recent_replies else ""
        if _echo_share(key, last) < _ECHO_MATCH:
            return "user"
        # A short one is echo only if it is how the line ended ("饭呢" after
        # "做晚饭呢"); "吃了" after "你吃了吗" is an answer.
        if len(key) >= _TAIL_ECHO_MIN_CHARS or last.endswith(key):
            return "echo"
        return "user"

    async def _discard_item(self, item_id: str | None) -> None:
        """Remove a turn that gets no reply, so later replies never see it."""
        await self._delete_item(item_id, "discarded turn")

    async def _delete_item(self, item_id: str | None, what: str) -> None:
        if not item_id:
            return
        if self._responding:
            self._pending_deletes.append((item_id, what))
            return
        try:
            await self._send_provider(item_delete_event(item_id))
        except Exception:
            logger.warning("[realtime-voice] %s not removed", what)

    async def _flush_deletes(self) -> None:
        pending, self._pending_deletes = self._pending_deletes, []
        for item_id, what in pending:
            await self._delete_item(item_id, what)

    # ------------------------------------------------------------------
    # Replies: memories first, then speak
    # ------------------------------------------------------------------

    async def _reply_when_ready(self, turn: int) -> None:
        """Wait (bounded) for this turn's retrieval, then ask for the reply."""
        task = self._prefetch_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._memory_wait_s)
            except (TimeoutError, asyncio.CancelledError):
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise
            except Exception:
                pass
        async with self._lock:
            if self._closed or turn != self._turn:
                return
            self._memory_ready = True
            await self._maybe_reply()

    async def _maybe_reply(self) -> None:
        if (
            self._closed
            or not self._unanswered
            or not self._memory_ready
            or self._responding
            or self._tool_running
            or self._user_speaking()
        ):
            return
        await self._request_reply()

    async def _request_reply(self, *, with_note: bool = True) -> None:
        note = self._reply_note() if with_note else ""
        self._unanswered = False
        self._responding = True
        self._reply_seen = False
        self._discarding = False
        self._cancel_pending = False
        if note:
            self._note_text = note
            self._note_id = None
            await self._send_provider(system_note_event(note))
        create = response_create_event()
        self._create_event_id = create["event_id"]
        await self._send_provider(create)
        self._watch_reply_start(create["event_id"])

    def _reply_note(self) -> str:
        """Note for the reply about to be requested; counts the memories as shown."""
        parts = [self._clock_line() if self._clock_line else ""]
        if self._slot is not None:
            parts.append(self._slot.note(with_memories=not self._quiet_turn))
            self.stats.reply_note(0 if self._quiet_turn else len(self._slot.lines))
            if not self._quiet_turn:
                self._slot.mark_shown()
        return "\n\n".join(part for part in parts if part)

    def _watch_reply_start(self, create_id: str) -> None:
        """Release a requested reply that never starts, so the call goes on."""
        async def watch() -> None:
            await asyncio.sleep(_REPLY_START_TIMEOUT_S)
            async with self._lock:
                if (
                    self._closed
                    or not self._responding
                    or self._reply_seen
                    or self._create_event_id != create_id
                ):
                    return
                logger.warning("[realtime-voice] reply did not start; released")
                self._responding = False
                self._cancel_pending = False
                await self._flush_deletes()
                await self._drop_greet_cue()
                await self._drop_note()
                await self._maybe_reply()

        task = asyncio.create_task(watch())
        self._watchdogs.add(task)
        task.add_done_callback(self._watchdogs.discard)

    # ------------------------------------------------------------------
    # Provider-side items and the end of each reply
    # ------------------------------------------------------------------

    async def _on_item_created(self, item: object) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return
        text = _item_text(item)
        if item.get("role") == "user" and text == self._greet_cue and self._greet_cue_sent:
            self._greet_cue_id = item["id"]
        elif item.get("role") == "system" and self._note_text and self._note_id is None:
            # Notes are the only system items sent mid-call; DashScope
            # replaces a client-chosen id, so take the one it reports.
            self._note_id = item["id"]

    async def _on_audio_delta(self, delta: object) -> None:
        if self._discarding or not isinstance(delta, str) or not delta:
            return
        try:
            pcm_bytes = len(base64.b64decode(delta, validate=True))
        except (binascii.Error, ValueError):
            return
        if not self._output_active:
            self.stats.first_audio()
            self._output_active = True
            self._audio_seq = 0
            await self._send_client({"type": "response.started"})
        # Generation runs ahead of playback; track when the client will stop
        # playing, so speech in the meantime is checked for echo.
        now = time.monotonic()
        if self._playback_until < now:
            # A new stretch of playback: audible from roughly now (the client
            # buffers briefly first).
            self._playback_started = now
            self._playback_until = now + _CLIENT_LEAD_S
        self._playback_until += pcm_bytes / (SAMPLE_RATE * 2)
        self._audio_seq += 1
        await self._send_client({"type": "audio.delta", "seq": self._audio_seq, "audio": delta})

    async def _on_response_done(self, event: dict) -> None:
        self._responding = False
        self._output_active = False
        self._cancel_pending = False
        await self._flush_deletes()
        await self._drop_greet_cue()
        await self._drop_note()
        text = assistant_transcript(event) or "".join(self._assistant_parts).strip()
        self._assistant_parts = []
        discarded = self._discarding
        if discarded:
            # The barge-in already stored what was said before the cut.
            self._discarding = False
            text = ""
        response = event.get("response")
        status = response.get("status") if isinstance(response, dict) else None
        interrupted = discarded or status in {"cancelled", "incomplete"}
        self._record_usage(response)
        if text:
            await self._store_assistant(text, interrupted=interrupted)
            await self._guard_repeat(text)
            if not self._closed:
                await self._send_client({
                    "type": "transcript.assistant",
                    "text": text,
                    "interrupted": interrupted,
                })
        await self._send_client({"type": "response.done", "interrupted": interrupted})
        calls, self._ready_calls = self._ready_calls, []
        if calls and not interrupted:
            self._tool_running = True
            self._tool_task = asyncio.create_task(self._run_tools(calls))
            return
        # A turn that arrived while this reply was going gets its own now.
        await self._maybe_reply()

    async def _store_assistant(self, text: str, *, interrupted: bool) -> None:
        logger.info(
            "[realtime-voice] assistant transcript len=%d interrupted=%s",
            len(text), interrupted,
        )
        try:
            self._remember_id(
                await self._save_assistant_line(text, self._user_message_id, interrupted),
            )
            self._collect_transcript("assistant", text, interrupted=interrupted)
            self._last_assistant_text = text
        except Exception:
            logger.warning("[realtime-voice] assistant transcript not stored")

    async def _drop_note(self) -> None:
        note_id, self._note_id, self._note_text = self._note_id, None, None
        await self._delete_item(note_id, "memory note")

    def _record_usage(self, response: object) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return
        for key, value in split_usage(usage).items():
            self._usage[key] += value
        self._usage["responses"] += 1

    # ------------------------------------------------------------------
    # recall_memory tool
    # ------------------------------------------------------------------

    async def _run_tools(self, calls: list[dict]) -> None:
        """Answer each tool call, then ask for the follow-up reply."""
        outputs: list[tuple[str, str]] = []
        self.stats.tool_calls += len(calls)
        for call in calls:
            call_id = call["call_id"]
            logger.info("[realtime-voice] tool_call name=%s call_id=%s", call["name"], call_id[:16])
            if call["name"] != "recall_memory":
                outputs.append((call_id, json.dumps({"error": f"未知工具: {call['name']}"}, ensure_ascii=False)))
                continue
            query = ""
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
                query = str(args.get("query") or "") if isinstance(args, dict) else ""
            except (json.JSONDecodeError, TypeError):
                pass
            entries = await self._get_memories(query)
            output = format_turn_memories(
                [entry["line"] for entry in entries or []], char_budget=1600,
            ) or "没有找到相关记忆。"
            outputs.append((call_id, output))
        async with self._lock:
            self._tool_running = False
            if self._closed:
                return
            for call_id, output in outputs:
                await self._send_provider(function_call_output_event(call_id=call_id, output=output))
            if self._unanswered:
                # The user moved on; their new turn gets the reply, with the
                # tool output already in context.
                await self._maybe_reply()
            elif not self._user_speaking() and not self._responding:
                await self._request_reply(with_note=False)

    async def _get_memories(self, query: str) -> list[dict[str, str]] | None:
        """Get memories: use prefetch cache if available, else fetch live."""
        if self._prefetch_task and not self._prefetch_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._prefetch_task), timeout=self._retrieval_timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("[realtime-voice] prefetch wait timed out")
            except Exception:
                pass
        if self._prefetch_result:
            result = self._prefetch_result
            self._prefetch_result = None
            return result
        try:
            return await asyncio.wait_for(
                self._retrieve_memories(query, None),
                timeout=self._retrieval_timeout_s,
            )
        except TimeoutError:
            logger.warning("[realtime-voice] memory retrieval timed out")
            return None
        except Exception:
            logger.warning("[realtime-voice] memory retrieval failed")
            return None

    # ------------------------------------------------------------------
    # Memory prefetch
    # ------------------------------------------------------------------

    def _start_prefetch(self, query: str) -> bool:
        """Retrieve for this user turn in the background; newest turn wins.

        Small talk retrieves nothing and leaves an earlier turn's retrieval running.
        """
        if is_small_talk(query):
            logger.info("[realtime-voice] memory skipped: small talk len=%d", len(query))
            return False
        self._cancel_prefetch()
        self._prefetch_result = None
        self._prefetch_task = asyncio.create_task(
            self._do_prefetch(query, self._last_assistant_text or None),
        )
        self._prefetch_task.add_done_callback(self._prefetch_done)
        return True

    async def _do_prefetch(self, query: str, previous: str | None) -> None:
        started = time.monotonic()
        entries: list[dict[str, str]] = []
        try:
            entries = await asyncio.wait_for(
                self._retrieve_memories(query, previous),
                timeout=self._retrieval_timeout_s,
            ) or []
            self._prefetch_result = entries
            self.stats.retrieval(time.monotonic() - started)
        except TimeoutError:
            self.stats.retrieval(time.monotonic() - started, timed_out=True)
            logger.info("[realtime-voice] memory retrieval timed out")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("[realtime-voice] prefetch failed")
        async with self._lock:
            self._refresh_slot(entries, started)

    def _refresh_slot(self, entries: list[dict[str, str]], started: float) -> None:
        """Fold one retrieval into the rolling block read by the next reply."""
        if self._slot is None or self._closed:
            return
        # A retrieval that lands before this turn's transcript belongs to it.
        turn = self._turn + (0 if self._unanswered or self._responding else 1)
        changed = self._slot.merge(entries, turn) if entries else self._slot.expire(turn)
        logger.info(
            "[realtime-voice] memory block turn=%d hits=%d lines=%d changed=%s ms=%d",
            turn, len(entries), len(self._slot.lines), changed,
            round((time.monotonic() - started) * 1000),
        )

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

    async def _guard_repeat(self, text: str) -> None:
        """Name a repeated line in the next reply's note so it is avoided.

        The repeat itself is already spoken; this keeps it from a third time.
        """
        key = spoken_key(text)
        repeated = len(key) >= _REPEAT_MIN_CHARS and any(
            SequenceMatcher(None, key, earlier).ratio() >= _REPEAT_RATIO
            for earlier in self._recent_replies
        )
        self._recent_replies.append(key)
        if repeated:
            self.stats.repeats += 1
        if repeated and self._slot is not None and self._slot.avoid(text):
            logger.info("[realtime-voice] repeat detected len=%d turn=%d", len(text), self._turn)

    # ------------------------------------------------------------------
    # Client-initiated actions
    # ------------------------------------------------------------------

    async def _interrupt(self) -> None:
        """Client requests interruption."""
        if self._responding or self._output_active:
            self.stats.client_interrupts += 1
            await self._cut_reply()
            self._output_active = False
            self._playback_started = self._playback_until = 0.0
            await self._send_client({"type": "interrupted"})

    async def _greet(self) -> None:
        """Client requests an opening greeting: the agent speaks first."""
        if self._responding or self._output_active or self._greet_cue_sent or self._unanswered:
            return
        self._greet_cue_sent = True
        await self._send_provider(history_item_event(item_id="greet_cue", role="user", text=self._greet_cue))
        await self._request_reply(with_note=False)

    async def _drop_greet_cue(self) -> None:
        cue_id, self._greet_cue_id = self._greet_cue_id, None
        await self._delete_item(cue_id, "greeting cue")

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
        # Never releases the pending reply: most errors carry no event_id,
        # and a rejected cancel looks like any other. Freeing the reply here
        # deleted its note mid-generation and DashScope closed the call
        # (2026-11-07). _watch_reply_start covers a reply that never starts.
        logger.warning("[realtime-voice] provider error: %s", message or "unknown")
        await self._send_client({
            "type": "warning",
            "message": message or "语音服务返回了一个错误",
        })

    # ------------------------------------------------------------------
    # Transcript collection
    # ------------------------------------------------------------------

    def _remember_id(self, message_id: str | None) -> None:
        if message_id:
            self._message_ids.append(message_id)

    def _collect_transcript(self, role: str, text: str, *, interrupted: bool = False) -> None:
        """Record a transcript pair for batch extraction at call end."""
        self._transcript_pairs.append({
            "role": role,
            "text": text,
            "interrupted": interrupted,
            "ts": time.time(),
        })


def _item_text(item: dict) -> str:
    parts = item.get("content") if isinstance(item.get("content"), list) else []
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key in ("text", "transcript"):
            value = part.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


class _MicMeter:
    """Byte count and level of the client's PCM16, sampled every 8th sample.

    A call where the provider never reports speech is either missing audio
    (bytes=0) or too quiet for server VAD (low peak_dbfs); this tells which.
    """

    _STRIDE = 8
    _FIRST_REPORT_BYTES = SAMPLE_RATE * 2 * 3

    def __init__(self) -> None:
        self.bytes = 0
        self.chunks = 0
        self._peak = 0
        self._sum_squares = 0
        self._samples = 0
        self._reported = False

    def add(self, pcm: bytes) -> bool:
        """Count a chunk. True once, when the first 3 s have arrived."""
        self.bytes += len(pcm)
        self.chunks += 1
        samples = array("h")
        samples.frombytes(pcm)
        picked = samples[:: self._STRIDE]
        if picked:
            self._peak = max(self._peak, max(map(abs, picked)))
            self._sum_squares += sum(value * value for value in picked)
            self._samples += len(picked)
        if not self._reported and self.bytes >= self._FIRST_REPORT_BYTES:
            self._reported = True
            return True
        return False

    def as_dict(self) -> dict:
        """Same numbers as summary(), for the per-call stats row; dBFS null when silent."""
        rms = math.sqrt(self._sum_squares / self._samples) if self._samples else 0.0
        return {"seconds": round(self.bytes / (SAMPLE_RATE * 2), 1), "chunks": self.chunks,
                "peak_dbfs": _dbfs_value(self._peak), "rms_dbfs": _dbfs_value(rms)}

    def summary(self) -> str:
        seconds = self.bytes / (SAMPLE_RATE * 2)
        rms = math.sqrt(self._sum_squares / self._samples) if self._samples else 0.0
        return (
            f"bytes={self.bytes} chunks={self.chunks} seconds={seconds:.1f} "
            f"peak_dbfs={_dbfs(self._peak)} rms_dbfs={_dbfs(rms)}"
        )


def _dbfs_value(value: float) -> float | None:
    return round(20 * math.log10(value / 32768), 1) if value > 0 else None


def _dbfs(value: float) -> str:
    if value <= 0:
        return "-inf"
    return f"{20 * math.log10(value / 32768):.1f}"
