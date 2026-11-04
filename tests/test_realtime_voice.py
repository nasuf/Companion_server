"""Realtime voice tests — server_vad + recall_memory tool architecture."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from app.services.realtime_voice.bridge import VoiceBridge
from app.services.realtime_voice.instructions import (
    DynamicContext,
    PersonaFacts,
    build_instructions,
    format_retrieved_memories,
    format_turn_memories,
    history_turns,
    persona_from_agent,
    voice_for_gender,
)
from app.services.realtime_voice.protocol import (
    RECALL_MEMORY_TOOL,
    decode_client_pcm,
    function_call_output_event,
    session_update_event,
)

_VOICE = Path(__file__).resolve().parents[1] / "app" / "api" / "realtime" / "voice.py"
_CHAT_WS = Path(__file__).resolve().parents[1] / "app" / "api" / "realtime" / "ws.py"


def _persona() -> PersonaFacts:
    return PersonaFacts(
        name="阿宁",
        nickname="宁宁",
        gender="女生",
        age=27,
        occupation="落叶纹理整理师",
        city="杭州",
        personality="安静内敛、感性细腻",
        style="语气平和。",
        background="在转塘长大。",
        values="把日子过慢一点",
        life_overview="晚上整理叶子标本。",
    )


# ---------------------------------------------------------------------------
# protocol.py
# ---------------------------------------------------------------------------

class TestSessionUpdateEvent:
    def test_server_vad_enabled(self):
        event = session_update_event(instructions="你是阿宁。", voice="elegantgentle-female")
        session = event["session"]
        assert session["modalities"] == ["text", "audio"]
        assert session["input_audio_format"] == "pcm16"
        assert session["output_audio_format"] == "pcm16"
        # Server VAD must be enabled
        td = session["turn_detection"]
        assert td["type"] == "server_vad"
        assert "silence_duration_ms" in td
        assert "prefix_padding_ms" in td

    def test_recall_memory_tool_included(self):
        event = session_update_event(instructions="test", voice="test")
        session = event["session"]
        tools = session.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "recall_memory"
        assert session.get("tool_choice") == "auto"

    def test_tool_can_be_disabled(self):
        event = session_update_event(
            instructions="test", voice="test", enable_memory_tool=False,
        )
        session = event["session"]
        assert session.get("tools", []) == []
        assert "tool_choice" not in session

    def test_silence_duration_configurable(self):
        event = session_update_event(
            instructions="test", voice="test",
            silence_duration_ms=800, prefix_padding_ms=300,
        )
        td = event["session"]["turn_detection"]
        assert td["silence_duration_ms"] == 800
        assert td["prefix_padding_ms"] == 300


class TestFunctionCallOutput:
    def test_creates_valid_event(self):
        event = function_call_output_event(call_id="call_123", output="记忆内容")
        assert event["type"] == "conversation.item.create"
        item = event["item"]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call_123"
        assert item["output"] == "记忆内容"


class TestDecodePcm:
    def test_raw_bytes(self):
        pcm = struct.pack("<4h", 100, -100, 200, -200)
        assert decode_client_pcm(pcm) == pcm

    def test_base64(self):
        import base64
        pcm = struct.pack("<2h", 500, -500)
        encoded = base64.b64encode(pcm).decode()
        assert decode_client_pcm(encoded) == pcm

    def test_rejects_empty(self):
        assert decode_client_pcm(b"") is None
        assert decode_client_pcm("") is None
        assert decode_client_pcm(None) is None


# ---------------------------------------------------------------------------
# instructions.py
# ---------------------------------------------------------------------------

class TestPersona:
    def test_voice_for_gender(self):
        assert voice_for_gender("male", female="f", male="m") == "m"
        assert voice_for_gender("女", female="f", male="m") == "f"
        assert voice_for_gender(None, female="f", male="m") == "f"

    def test_persona_from_agent(self):
        agent = SimpleNamespace(
            name="阿宁", nickname="宁宁", gender="female", age=27,
            occupation="整理师", city="杭州", background="转塘长大",
            values={"生活": "慢"}, lifeOverview="整理标本",
            mbti={"EI": 30, "NS": 60, "TF": 70, "JP": 45},
            currentMbti=None,
        )
        persona = persona_from_agent(agent)
        assert persona.name == "阿宁"
        assert persona.gender == "女生"

    def test_build_instructions_with_dynamic_context(self):
        persona = _persona()
        dynamic = DynamicContext(
            time_context="当前时间：2026年11月4日 14时 周二",
            intimacy_stage="升温期",
            ai_mood_text="高兴（淡淡的）",
            patience_zone="normal",
            relation_meta_line="认识 30 天，聊过大约 200 轮",
        )
        text = build_instructions(
            persona,
            [("身份", "用户叫小明")],
            [("身份", "阿宁现居杭州")],
            has_history=True,
            dynamic=dynamic,
        )
        assert "recall_memory" in text
        assert "当前时间" in text
        assert "升温期" in text
        assert "高兴" in text
        assert "认识 30 天" in text
        assert "下面已经有你们聊过的内容" in text

    def test_build_instructions_without_dynamic(self):
        text = build_instructions(
            _persona(), [], [], has_history=False,
        )
        assert "通话刚开始" in text
        assert "recall_memory" in text

    def test_build_instructions_patience_medium(self):
        dynamic = DynamicContext(patience_zone="medium")
        text = build_instructions(
            _persona(), [], [], has_history=False, dynamic=dynamic,
        )
        assert "不满" in text

    def test_format_retrieved_memories(self):
        items = [
            {"source": "user", "id": "u1", "text": "用户养了一只叫米粒的猫"},
            {"source": "ai", "id": "a1", "text": "阿宁现居杭州"},
        ]
        text, ids = format_retrieved_memories(items)
        assert text is not None
        assert "米粒" in text
        assert "杭州" in text
        assert set(ids) == {"u1", "a1"}

    def test_history_turns_limits(self):
        messages = [{"role": "user", "content": f"第{i}句"} for i in range(20)]
        turns = history_turns(messages, limit=3)
        assert len(turns) == 3
        assert [turn["text"] for turn in turns] == ["第17句", "第18句", "第19句"]


# ---------------------------------------------------------------------------
# bridge.py — VoiceBridge with server_vad + tool architecture
# ---------------------------------------------------------------------------

class TestVoiceBridge:
    """Integration tests for the voice bridge with server_vad enabled."""

    @staticmethod
    def _make_bridge(
        *,
        provider_events: list | None = None,
        client_events: list | None = None,
        memory_result: str | None = "关于用户：用户喜欢蓝色",
    ) -> tuple[VoiceBridge, list, list]:
        provider_sent: list = provider_events if provider_events is not None else []
        client_sent: list = client_events if client_events is not None else []

        async def send_provider(event):
            provider_sent.append(event)

        async def send_client(event):
            client_sent.append(event)

        async def retrieve(text):
            return memory_result

        async def save_user(text):
            return "msg_user_001"

        async def save_assistant(text, user_msg_id, interrupted):
            pass

        bridge = VoiceBridge(
            send_provider=send_provider,
            send_client=send_client,
            retrieve_memories=retrieve,
            save_user_line=save_user,
            save_assistant_line=save_assistant,
            retrieval_timeout_s=1.0,
        )
        return bridge, provider_sent, client_sent

    @pytest.mark.asyncio
    async def test_audio_forwarded_to_provider(self):
        """Client PCM is forwarded directly to provider (no local VAD)."""
        bridge, provider_sent, _ = self._make_bridge()
        pcm = struct.pack("<4h", 100, -100, 200, -200)
        await bridge.feed_audio(pcm)
        assert any(e["type"] == "input_audio_buffer.append" for e in provider_sent)

    @pytest.mark.asyncio
    async def test_speech_started_relayed(self):
        """Provider speech_started event relayed to client."""
        bridge, _, client_sent = self._make_bridge()
        await bridge.feed_provider_event({
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 1000,
            "item_id": "item_001",
        })
        assert any(e["type"] == "speech.started" for e in client_sent)

    @pytest.mark.asyncio
    async def test_speech_stopped_relayed(self):
        """Provider speech_stopped event relayed to client."""
        bridge, _, client_sent = self._make_bridge()
        await bridge.feed_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 2000,
            "item_id": "item_001",
        })
        assert any(e["type"] == "speech.stopped" for e in client_sent)

    @pytest.mark.asyncio
    async def test_user_transcript_saved_and_prefetched(self):
        """User transcript triggers save, prefetch and client notification."""
        bridge, _, client_sent = self._make_bridge()
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_001",
            "transcript": "你喜欢什么颜色",
        })
        transcript_events = [e for e in client_sent if e["type"] == "transcript.user"]
        assert len(transcript_events) == 1
        assert transcript_events[0]["text"] == "你喜欢什么颜色"
        assert transcript_events[0]["message_id"] == "msg_user_001"
        # Prefetch should have been started
        assert bridge._prefetch_task is not None

    @pytest.mark.asyncio
    async def test_audio_delta_relayed(self):
        """Provider audio deltas are relayed to client with seq numbers."""
        import base64
        bridge, _, client_sent = self._make_bridge()
        pcm = struct.pack("<2h", 100, -100)
        delta = base64.b64encode(pcm).decode()

        # First delta opens output
        await bridge.feed_provider_event({
            "type": "response.audio.delta",
            "delta": delta,
        })
        started = [e for e in client_sent if e["type"] == "response.started"]
        assert len(started) == 1
        audio = [e for e in client_sent if e["type"] == "audio.delta"]
        assert len(audio) == 1
        assert audio[0]["seq"] == 1

    @pytest.mark.asyncio
    async def test_function_call_recall_memory(self):
        """recall_memory tool call: prefetch result returned to provider."""
        bridge, provider_sent, _ = self._make_bridge(memory_result="关于用户：用户喜欢蓝色")

        # Simulate user transcript to trigger prefetch
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_001",
            "transcript": "你还记得我喜欢什么颜色吗",
        })
        # Wait for prefetch
        if bridge._prefetch_task:
            await bridge._prefetch_task

        # Simulate function_call creation
        await bridge.feed_provider_event({
            "type": "conversation.item.created",
            "item": {
                "id": "fc_001",
                "type": "function_call",
                "call_id": "call_001",
                "name": "recall_memory",
            },
        })
        # Simulate arguments
        await bridge.feed_provider_event({
            "type": "response.function_call_arguments.done",
            "call_id": "call_001",
            "arguments": '{"query": "用户喜欢的颜色"}',
        })

        # Should have sent function_call_output + response.create
        output_events = [
            e for e in provider_sent
            if e.get("type") == "conversation.item.create"
            and isinstance(e.get("item"), dict)
            and e["item"].get("type") == "function_call_output"
        ]
        assert len(output_events) == 1
        assert "蓝色" in output_events[0]["item"]["output"]
        assert any(e["type"] == "response.create" for e in provider_sent)

    @pytest.mark.asyncio
    async def test_response_done_saves_transcript(self):
        """response.done event saves assistant transcript."""
        bridge, _, client_sent = self._make_bridge()

        # Accumulate transcript deltas
        await bridge.feed_provider_event({
            "type": "response.audio_transcript.delta",
            "delta": "蓝色",
        })
        await bridge.feed_provider_event({
            "type": "response.audio_transcript.delta",
            "delta": "挺好看的",
        })
        await bridge.feed_provider_event({
            "type": "response.done",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "audio", "transcript": "蓝色挺好看的"}],
                }],
            },
        })
        transcript_events = [e for e in client_sent if e["type"] == "transcript.assistant"]
        assert len(transcript_events) == 1
        assert transcript_events[0]["text"] == "蓝色挺好看的"
        assert transcript_events[0]["interrupted"] is False

    @pytest.mark.asyncio
    async def test_speech_started_interrupts_output(self):
        """Speech during output marks it as interrupted."""
        import base64
        bridge, _, client_sent = self._make_bridge()

        # Start output
        bridge._responding = True
        bridge._output_active = True
        bridge._assistant_parts = ["你好"]

        # User starts speaking — triggers barge-in
        await bridge.feed_provider_event({
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 1000,
            "item_id": "item_002",
        })

        assert bridge._output_active is False
        assert bridge._responding is False
        assert any(e["type"] == "speech.started" for e in client_sent)

    @pytest.mark.asyncio
    async def test_client_interrupt(self):
        """Client-initiated interrupt cancels provider response."""
        bridge, provider_sent, client_sent = self._make_bridge()
        bridge._responding = True
        bridge._output_active = True

        await bridge.feed_client_text('{"type": "interrupt"}')

        assert any(e["type"] == "response.cancel" for e in provider_sent)
        assert any(e["type"] == "interrupted" for e in client_sent)

    @pytest.mark.asyncio
    async def test_greet_triggers_response(self):
        """Greet event triggers response.create on provider."""
        bridge, provider_sent, _ = self._make_bridge()
        await bridge.feed_client_text('{"type": "greet"}')
        assert any(e["type"] == "response.create" for e in provider_sent)

    @pytest.mark.asyncio
    async def test_greet_blocked_during_response(self):
        """Greet does nothing if a response is already in progress."""
        bridge, provider_sent, _ = self._make_bridge()
        bridge._responding = True
        await bridge.feed_client_text('{"type": "greet"}')
        assert not any(e["type"] == "response.create" for e in provider_sent)

    @pytest.mark.asyncio
    async def test_ping_pong(self):
        bridge, _, client_sent = self._make_bridge()
        await bridge.feed_client_text('{"type": "ping"}')
        assert any(e["type"] == "pong" for e in client_sent)

    @pytest.mark.asyncio
    async def test_transcript_pairs_collected(self):
        """Transcript pairs are collected for end-of-call extraction."""
        bridge, _, _ = self._make_bridge()

        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "你好",
        })
        await bridge.feed_provider_event({
            "type": "response.done",
            "response": {
                "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "audio", "transcript": "你好呀"}]}],
            },
        })

        pairs = bridge.transcript_pairs
        assert len(pairs) == 2
        assert pairs[0]["role"] == "user"
        assert pairs[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        bridge, _, _ = self._make_bridge()
        await bridge.aclose()
        await bridge.aclose()  # second call should not raise
        assert bridge._closed


# ---------------------------------------------------------------------------
# voice.py endpoint — structural checks
# ---------------------------------------------------------------------------

class TestVoiceEndpoint:
    def test_endpoint_does_not_import_vad(self):
        """vad.py is deleted; voice.py must not reference it."""
        source = _VOICE.read_text()
        assert "from app.services.realtime_voice.vad" not in source
        assert "PcmVad" not in source

    def test_endpoint_uses_server_vad_protocol(self):
        """voice.py should configure session with server_vad."""
        source = _VOICE.read_text()
        assert "configure_session" in source
        assert "silence_duration_ms" in source

    def test_endpoint_schedules_end_of_call_extraction(self):
        """voice.py should trigger extraction at call end, not per-utterance."""
        source = _VOICE.read_text()
        assert "schedule_end_of_call_extraction" in source
        assert "process_memory_pipeline" not in source

    def test_endpoint_resolves_the_shared_agent_voice(self):
        """Calls use the agent's library voice, with the official gender voice as fallback."""
        source = _VOICE.read_text()
        assert 'getattr(conv.agent, "ttsVoiceId"' not in source
        assert "ensure_agent_voice(conv.agent)" in source
        assert "fallback_voice=fallback_voice" in source

    def test_default_voices_are_realtime_compatible(self):
        """soft-spoken-gentleman is TTS-only and fails session.update."""
        from app.config import Settings

        fields = Settings.model_fields
        assert fields["stepfun_realtime_voice_male"].default == "wenrounansheng"
        assert fields["stepfun_realtime_voice_female"].default == "elegantgentle-female"

    def test_endpoint_has_idle_timeout(self):
        """voice.py should have idle watchdog."""
        source = _VOICE.read_text()
        assert "idle_watchdog" in source
        assert "_IDLE_TIMEOUT_SECONDS" in source

    def test_text_chat_not_affected(self):
        """Text chat ws.py must not import realtime_voice modules."""
        source = _CHAT_WS.read_text()
        assert "realtime_voice" not in source

    def test_stepfun_key_not_in_deploy_required(self):
        """STEPFUN_API_KEY should not block deployment when unconfigured."""
        deploy = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"
        if deploy.exists():
            required_line = deploy.read_text()
            # The key should NOT be in the mandatory check loop
            assert "STEPFUN_API_KEY" not in required_line.split("for key in")[1].split("; do")[0] if "for key in" in required_line else True


class _FakeVoiceSocket:
    def __init__(self):
        self.scope = {"subprotocols": ["companion.chat.v1", "companion.ticket.x"]}
        self.headers = {}
        self.sent: list[dict] = []
        self.accepted = False
        self.closed_with: int | None = None

    async def accept(self, subprotocol=None):
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000, reason=None):
        self.closed_with = code


class _FakeProviderConnection:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize(
    ("gender", "expected", "lookup"),
    [
        ("male", "wenrounansheng", "voice-tone-male"),
        ("female", "elegantgentle-female", "voice-tone-female"),
        ("female", "elegantgentle-female", None),
    ],
)
async def test_call_uses_agent_voice_with_official_fallback(monkeypatch, gender, expected, lookup):
    """The agent's library voice is tried first; rejection reaches the client."""
    from app.api.realtime import voice as endpoint
    from app.services.realtime_voice.provider import StepfunSetupError

    agent = SimpleNamespace(
        id="agent-1", userId="user-1", status="active", gender=gender,
        ttsVoiceId="qwen-audio-3.0-tts-plus-voice-clone",
    )
    conv = SimpleNamespace(
        id="conv-1", userId="user-1", isDeleted=False, agent=agent, workspaceId=None,
    )
    principal = SimpleNamespace(role="user", user_id="user-1", expires_at=float("inf"))
    seen: dict = {}
    released: list[tuple] = []

    async def find_unique(**_):
        return conv

    async def consume_ticket(*_):
        return principal

    async def ok(*_, **__):
        return True

    async def load_call_context(**_):
        return SimpleNamespace(name="阿宁"), "instructions", []

    async def acquire_call(_):
        return "lock-token"

    async def release_call(conversation_id, token):
        released.append((conversation_id, token))

    async def configure_session(sock, *, voice, fallback_voice, **_):
        seen["voice"] = voice
        seen["fallback"] = fallback_voice
        raise StepfunSetupError(f"voice {voice} is not valid")

    async def ensure_agent_voice(agent):
        if lookup is None:
            raise RuntimeError("database unavailable")
        return lookup

    monkeypatch.setattr(endpoint, "is_redis_healthy", lambda: True)
    monkeypatch.setattr(endpoint, "offered_ticket", lambda _: "ticket")
    monkeypatch.setattr(endpoint, "origin_allowed", lambda _: True)
    monkeypatch.setattr(endpoint, "consume_ticket", consume_ticket)
    monkeypatch.setattr(endpoint.db, "conversation", SimpleNamespace(find_unique=find_unique), raising=False)
    monkeypatch.setattr(endpoint, "get_patience", ok)
    monkeypatch.setattr(endpoint, "get_patience_zone", lambda _: "normal")
    monkeypatch.setattr(endpoint, "_accounts_current", ok)
    monkeypatch.setattr(endpoint.settings, "stepfun_api_key", "sk-test")
    monkeypatch.setattr(endpoint.settings, "stepfun_realtime_voice_male", "wenrounansheng")
    monkeypatch.setattr(endpoint.settings, "stepfun_realtime_voice_female", "elegantgentle-female")
    monkeypatch.setattr(endpoint, "acquire_call", acquire_call)
    monkeypatch.setattr(endpoint, "release_call", release_call)
    monkeypatch.setattr(endpoint, "load_call_context", load_call_context)
    monkeypatch.setattr(endpoint, "connect_stepfun", lambda *_: _FakeProviderConnection())
    monkeypatch.setattr(endpoint, "configure_session", configure_session)
    monkeypatch.setattr(endpoint, "ensure_agent_voice", ensure_agent_voice)
    monkeypatch.setattr(
        endpoint, "get_effective_realtime_model",
        AsyncMock(return_value="stepaudio-2.5-realtime"),
    )

    ws = _FakeVoiceSocket()
    await endpoint.voice_websocket(ws, "conv-1")

    assert ws.accepted
    assert seen["voice"] == (lookup or expected)
    assert seen["fallback"] == expected
    assert ws.sent == [{"type": "error", "code": "provider_unavailable", "message": "实时语音暂时连不上"}]
    assert ws.closed_with == 1011
    assert released == [("conv-1", "lock-token")]


async def test_call_transcripts_are_hidden_from_timeline(monkeypatch):
    """Voice calls are heard, not read: transcripts never reach the chat list."""
    from app.services.realtime_voice import persist

    created: list[dict] = []

    async def create(*, data):
        created.append(data)
        return SimpleNamespace(id=f"m{len(created)}")

    monkeypatch.setattr(persist.db, "message", SimpleNamespace(create=create), raising=False)
    monkeypatch.setattr(persist, "fire_background", lambda coro: coro.close())

    user_id = await persist.save_user_transcript(
        conversation_id="c1", user_id="u1", workspace_id=None, model="m", text="你好",
    )
    await persist.save_assistant_transcript(
        conversation_id="c1", user_id="u1", workspace_id=None, model="m",
        text="嗨", user_message_id=user_id, interrupted=False,
    )

    assert [row["hiddenFromTimeline"] for row in created] == [True, True]


async def test_chat_history_skips_hidden_rows(monkeypatch):
    from unittest.mock import AsyncMock

    from app.api.public import conversations

    find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(conversations, "db", SimpleNamespace(message=SimpleNamespace(find_many=find_many)))
    await conversations.list_messages(
        conversation_id="c1", limit=50, offset=0, include_metadata=True,
        include_achievements=False, include_usage=False,
        conv=SimpleNamespace(userId="u1", agentId="a1"), user={"sub": "u1", "role": "user"},
    )
    assert find_many.call_args.kwargs["where"]["hiddenFromTimeline"] is False


async def test_search_and_rank_skip_hidden_rows(monkeypatch):
    from unittest.mock import AsyncMock

    from app.services.chat import message_search

    find_many = AsyncMock(return_value=[])
    query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr(
        message_search, "db",
        SimpleNamespace(message=SimpleNamespace(find_many=find_many), query_raw=query_raw),
    )
    await message_search.search_messages(conversation_id="c1", q="天气", scope="text", limit=30, offset=0)
    await message_search.message_rank("c1", "m1")

    assert find_many.call_args.kwargs["where"]["hiddenFromTimeline"] is False
    assert "NOT m2.hidden_from_timeline" in query_raw.call_args.args[0]


class _ScriptedProvider:
    """StepfunSocket stand-in that replays provider events per session.update."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[dict] = []
        self.inbox = [{"type": "session.created"}]

    async def send_event(self, event):
        self.sent.append(event)
        if event.get("type") == "session.update":
            self.inbox.append(self.replies.pop(0))

    async def recv_event(self):
        if not self.inbox:
            await asyncio.sleep(3600)
        return self.inbox.pop(0)


_REJECTED = {"type": "error", "error": {"message": "voice voice-tone-x is not valid"}}


async def test_configure_session_retries_a_rejected_voice_once():
    from app.services.realtime_voice.provider import configure_session

    sock = _ScriptedProvider([_REJECTED, {"type": "session.updated"}])
    voice = await configure_session(
        sock, instructions="hi", voice="voice-tone-x", history=[], fallback_voice="wenrounansheng",
    )
    assert voice == "wenrounansheng"
    assert [e["session"]["voice"] for e in sock.sent] == ["voice-tone-x", "wenrounansheng"]


async def test_configure_session_keeps_an_accepted_voice():
    from app.services.realtime_voice.provider import configure_session

    sock = _ScriptedProvider([{"type": "session.updated"}])
    voice = await configure_session(
        sock, instructions="hi", voice="voice-tone-x", history=[], fallback_voice="wenrounansheng",
    )
    assert voice == "voice-tone-x"
    assert len(sock.sent) == 1


@pytest.mark.parametrize("fallback", [None, "voice-tone-x"])
async def test_configure_session_without_a_distinct_fallback_raises(fallback):
    from app.services.realtime_voice.provider import StepfunSetupError, configure_session

    sock = _ScriptedProvider([_REJECTED])
    with pytest.raises(StepfunSetupError, match="not valid"):
        await configure_session(
            sock, instructions="hi", voice="voice-tone-x", history=[], fallback_voice=fallback,
        )


async def test_bridge_sums_provider_usage_across_responses():
    sent: list[dict] = []

    async def send(event):
        sent.append(event)

    async def noop(*_, **__):
        return None

    bridge = VoiceBridge(
        send_provider=send, send_client=send, retrieve_memories=noop,
        save_user_line=noop, save_assistant_line=noop,
    )
    for usage in (
        {"input_tokens": 31, "output_tokens": 3, "input_token_details": {"cached_tokens": 10}},
        {"input_tokens": 50, "output_tokens": 7},
        None,
    ):
        response = {"status": "completed", "output": []}
        if usage is not None:
            response["usage"] = usage
        await bridge.feed_provider_event({"type": "response.done", "response": response})

    assert bridge.usage == {"input": 81, "output": 10, "cached_input": 10, "responses": 2}


async def test_call_usage_is_written_as_one_realtime_row(monkeypatch):
    from app.api.realtime import voice as endpoint
    from app.services.llm import usage_repo

    written: list[dict] = []

    async def persist(**kwargs):
        written.append(kwargs)

    monkeypatch.setattr(usage_repo, "persist_usage", persist)
    await endpoint._record_call_usage(
        {"input": 81, "output": 10, "cached_input": 10, "responses": 2},
        model="stepaudio-2.5-realtime", conversation_id="conv-1", agent_id="a1", user_id="u1",
    )
    await endpoint._record_call_usage(
        {"input": 0, "output": 0, "cached_input": 0, "responses": 0},
        model="stepaudio-2.5-realtime", conversation_id="conv-1", agent_id="a1", user_id="u1",
    )

    assert len(written) == 1
    row = written[0]
    assert row["scope"] == "realtime_voice"
    assert row["summary"]["tokens_by_model"] == {
        "stepaudio-2.5-realtime": {"input": 81, "output": 10, "cached_input": 10},
    }
    assert row["summary"]["call_count"] == 2


def test_realtime_usage_is_its_own_cost_category():
    from app.services.effect import costs

    assert "realtime" in costs.CATEGORIES
    assert costs._SCOPE_CATEGORY["realtime_voice"] == "realtime"


def test_realtime_model_resolves_from_system_config(monkeypatch):
    from app.services import runtime_config

    monkeypatch.setattr(runtime_config, "_current_snapshot", runtime_config.ContextVar("s", default=None))
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    assert runtime_config.resolve_config_sync(agent_id=None).realtime_model == "stepaudio-2.5-realtime"
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realtimeModel": "stepaudio-3-realtime"})
    assert runtime_config.resolve_config_sync(agent_id=None).realtime_model == "stepaudio-3-realtime"


def test_realtime_is_a_stepfun_model_kind():
    from app.api.admin import model_registry

    model_registry._validate_model_metadata("realtime", "per_million_tokens", provider="stepfun")
    with pytest.raises(Exception) as exc:
        model_registry._validate_model_metadata("realtime", "per_10k_characters", provider="stepfun")
    assert getattr(exc.value, "status_code", None) == 400


async def test_finished_call_meters_usage_and_schedules_extraction(monkeypatch):
    """A call that reaches the pumps meters StepFun usage once at hang-up."""
    from app.api.realtime import voice as endpoint

    agent = SimpleNamespace(id="agent-1", userId="user-1", status="active", gender="female")
    conv = SimpleNamespace(id="conv-1", userId="user-1", isDeleted=False, agent=agent, workspaceId=None)
    principal = SimpleNamespace(role="user", user_id="user-1", expires_at=float("inf"))
    metered: list[dict] = []
    extracted: list[dict] = []

    class FakeBridge:
        transcript_pairs = [{"role": "user", "text": "hi"}]
        usage = {"input": 5, "output": 2, "cached_input": 0, "responses": 1}

        async def aclose(self):
            pass

    async def ok(*_, **__):
        return True

    async def none(*_, **__):
        return None

    async def find_unique(**_):
        return conv

    async def principal_for(*_):
        return principal

    async def context(**_):
        return SimpleNamespace(name="阿宁"), "instructions", []

    async def lock(_):
        return "lock-token"

    async def configure(sock, *, voice, **_):
        return voice

    async def record(usage, **kwargs):
        metered.append({"usage": usage, **kwargs})

    monkeypatch.setattr(endpoint, "is_redis_healthy", lambda: True)
    monkeypatch.setattr(endpoint, "offered_ticket", lambda _: "ticket")
    monkeypatch.setattr(endpoint, "origin_allowed", lambda _: True)
    monkeypatch.setattr(endpoint, "consume_ticket", principal_for)
    monkeypatch.setattr(endpoint.db, "conversation", SimpleNamespace(find_unique=find_unique), raising=False)
    monkeypatch.setattr(endpoint, "get_patience", ok)
    monkeypatch.setattr(endpoint, "get_patience_zone", lambda _: "normal")
    monkeypatch.setattr(endpoint, "_accounts_current", ok)
    monkeypatch.setattr(endpoint.settings, "stepfun_api_key", "sk-test")
    monkeypatch.setattr(endpoint, "acquire_call", lock)
    monkeypatch.setattr(endpoint, "release_call", none)
    monkeypatch.setattr(endpoint, "load_call_context", context)
    monkeypatch.setattr(endpoint, "ensure_agent_voice", AsyncMock(return_value="voice-tone-1"))
    monkeypatch.setattr(endpoint, "get_effective_realtime_model", AsyncMock(return_value="stepaudio-2.5-realtime"))
    monkeypatch.setattr(endpoint, "connect_stepfun", lambda *_: _FakeProviderConnection())
    monkeypatch.setattr(endpoint, "configure_session", configure)
    monkeypatch.setattr(endpoint, "_bridge", lambda **_: FakeBridge())
    monkeypatch.setattr(endpoint, "schedule_user_present", lambda _: None)
    monkeypatch.setattr(endpoint, "_run_pumps", none)
    monkeypatch.setattr(endpoint, "_record_call_usage", record)
    monkeypatch.setattr(endpoint, "schedule_end_of_call_extraction", lambda **kw: extracted.append(kw))

    ws = _FakeVoiceSocket()
    await endpoint.voice_websocket(ws, "conv-1")

    assert ws.sent[0]["type"] == "ready" and ws.sent[0]["voice"] == "voice-tone-1"
    assert metered == [{
        "usage": FakeBridge.usage, "model": "stepaudio-2.5-realtime",
        "conversation_id": "conv-1", "agent_id": "agent-1", "user_id": "user-1",
    }]
    assert extracted == [{"user_id": "user-1", "workspace_id": None, "conversation_id": "conv-1"}]


async def test_call_usage_metering_failure_is_swallowed(monkeypatch, caplog):
    from app.api.realtime import voice as endpoint
    from app.services.llm import usage_repo

    monkeypatch.setattr(usage_repo, "persist_usage", AsyncMock(side_effect=RuntimeError("db down")))
    await endpoint._record_call_usage(
        {"input": 1, "output": 1, "cached_input": 0, "responses": 1},
        model="m", conversation_id="conv-1", agent_id="a1", user_id="u1",
    )
    assert "usage metering failed" in caplog.text


async def test_configure_session_seeds_history_and_returns_voice():
    from app.services.realtime_voice.provider import configure_session

    sock = _ScriptedProvider([{"type": "session.updated"}])
    original_send = sock.send_event

    async def send(event):
        await original_send(event)
        if event.get("type") == "conversation.item.create":
            sock.inbox.append({"type": "conversation.item.created", "item": {"id": event["item"]["id"]}})

    sock.send_event = send
    voice = await configure_session(
        sock, instructions="hi", voice="voice-tone-x",
        history=[{"role": "user", "text": "你好"}], fallback_voice="wenrounansheng",
    )
    assert voice == "voice-tone-x"
    assert [e["type"] for e in sock.sent] == ["session.update", "conversation.item.create"]


async def test_effective_realtime_model_reads_runtime_config(monkeypatch):
    from app.services import runtime_config

    monkeypatch.setattr(runtime_config, "ensure_loaded", AsyncMock())
    monkeypatch.setattr(runtime_config, "_current_snapshot", runtime_config.ContextVar("s", default=None))
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realtimeModel": "stepaudio-3-realtime"})
    assert await runtime_config.get_effective_realtime_model() == "stepaudio-3-realtime"
