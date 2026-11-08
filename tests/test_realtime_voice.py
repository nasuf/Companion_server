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
    format_turn_memories,
    history_turns,
    persona_from_agent,
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
    def test_server_vad_finds_turns_but_never_replies_or_cancels_by_itself(self):
        event = session_update_event(instructions="你是阿宁。", voice="Serena")
        session = event["session"]
        assert session["modalities"] == ["text", "audio"]
        assert session["voice"] == "Serena"
        assert session["audio"] == {
            "input": {"format": {"type": "pcm", "sample_rate": 24000}},
            "output": {"format": {"type": "pcm", "sample_rate": 24000}},
        }
        td = session["turn_detection"]
        assert td["type"] == "server_vad"
        # The bridge asks for each reply after retrieval, and decides barge-in.
        assert td["create_response"] is False
        assert td["interrupt_response"] is False
        assert session["input_audio_transcription"] == {"model": "qwen3-asr-flash-realtime"}

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

    def test_vad_and_temperature_are_clamped(self):
        td = session_update_event(instructions="t", voice="v", silence_duration_ms=800, vad_threshold=0.7)["session"]["turn_detection"]
        assert td["silence_duration_ms"] == 800 and td["threshold"] == 0.7
        td = session_update_event(instructions="t", voice="v", silence_duration_ms=50, vad_threshold=3)["session"]["turn_detection"]
        assert td["silence_duration_ms"] == 200 and td["threshold"] == 1.0
        assert session_update_event(instructions="t", voice="v", silence_duration_ms=9000)["session"]["turn_detection"]["silence_duration_ms"] == 6000
        assert "temperature" not in session_update_event(instructions="t", voice="v")["session"]
        assert session_update_event(instructions="t", voice="v", temperature=5)["session"]["temperature"] == 1.99


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

    def test_memory_lines_label_owner_and_cap_each_side(self):
        from app.services.realtime_voice.instructions import memory_lines

        items = [{"source": "user", "id": f"u{i}", "text": f"用户的第{i}件事"} for i in range(7)]
        items += [{"source": "ai", "id": "a1", "text": "阿宁现居杭州"}, "noise", {"source": "user", "text": " "}]
        lines = memory_lines(items)
        assert [entry["id"] for entry in lines] == ["u0", "u1", "u2", "u3", "u4", "a1"]
        assert lines[0] == {"id": "u0", "raw": "用户的第0件事", "line": "关于用户：用户的第0件事"}
        assert lines[-1]["line"] == "关于你自己：阿宁现居杭州"

    def test_history_turns_limits(self):
        messages = [{"role": ("user", "assistant")[i % 2], "content": f"第{i}句"} for i in range(20)]
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
        memory_result: list | None = None,
        instructions: str | None = None,
    ) -> tuple[VoiceBridge, list, list]:
        if memory_result is None:
            memory_result = [{"id": "m1", "raw": "用户喜欢蓝色", "line": "关于用户：用户喜欢蓝色"}]
        provider_sent: list = provider_events if provider_events is not None else []
        client_sent: list = client_events if client_events is not None else []

        async def send_provider(event):
            provider_sent.append(event)

        async def send_client(event):
            client_sent.append(event)

        async def retrieve(text, previous):
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
            instructions=instructions,
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
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "item_001"})
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
        """recall_memory runs once its reply is done, then the follow-up reply."""
        bridge, provider_sent, _ = self._make_bridge()
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_001",
            "transcript": "你还记得我喜欢什么颜色吗",
        })
        await bridge._reply_task
        assert provider_sent[-1]["type"] == "response.create"
        await bridge.feed_provider_event({"type": "response.created"})
        await bridge.feed_provider_event({
            "type": "response.function_call_arguments.done",
            "call_id": "call_001", "name": "recall_memory",
            "arguments": '{"query": "用户喜欢的颜色"}',
        })
        await bridge.feed_provider_event({"type": "response.done", "response": {
            "status": "completed", "output": [{"type": "function_call", "call_id": "call_001"}]}})
        await bridge._tool_task
        output_events = [
            e for e in provider_sent
            if e.get("type") == "conversation.item.create"
            and isinstance(e.get("item"), dict)
            and e["item"].get("type") == "function_call_output"
        ]
        assert len(output_events) == 1
        assert "蓝色" in output_events[0]["item"]["output"]
        assert provider_sent[-1]["type"] == "response.create"
        assert sum(e["type"] == "response.create" for e in provider_sent) == 2

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

    @staticmethod
    async def _agent_speaking(bridge, text="今天店里特别忙一下午做了三个蛋糕"):
        import base64
        chunk = base64.b64encode(b"\x01\x00" * 2400).decode()
        await bridge.feed_provider_event({"type": "response.created"})
        await bridge.feed_provider_event({"type": "response.audio_transcript.delta", "delta": text})
        await bridge.feed_provider_event({"type": "response.audio.delta", "delta": chunk})
        return chunk

    @pytest.mark.asyncio
    async def test_speech_over_the_agent_waits_for_its_transcript(self):
        """Speech over the agent may be its own echo: nothing stops yet."""
        bridge, provider_sent, client_sent = self._make_bridge()
        await self._agent_speaking(bridge)
        client_sent.clear()
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "item_002"})
        assert client_sent == [] and provider_sent == []

    @pytest.mark.asyncio
    async def test_barge_in_cancels_generation_and_drops_its_audio(self):
        """The user's own words cut the reply; the next reply plays normally."""
        saved: list[tuple] = []
        bridge, provider_sent, client_sent = self._make_bridge()

        async def save_assistant(text, user_msg_id, interrupted):
            saved.append((text, interrupted))

        bridge._save_assistant_line = save_assistant
        chunk = await self._agent_speaking(bridge, "我跟你说")
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.delta", "item_id": "u1", "text": "等一下", "stash": "我想问"})

        assert [e["type"] for e in provider_sent] == ["response.cancel"]
        assert client_sent[-1] == {"type": "speech.started"}
        client_sent.clear()
        # Late events of the cancelled reply are swallowed.
        await bridge.feed_provider_event({"type": "response.audio.delta", "delta": chunk})
        await bridge.feed_provider_event({"type": "response.audio_transcript.delta", "delta": "后面的话"})
        await bridge.feed_provider_event({"type": "response.done", "response": {"status": "cancelled"}})
        assert [e["type"] for e in client_sent] == ["response.done"]
        assert saved == [("我跟你说", True)]

        await bridge.feed_provider_event({"type": "response.created"})
        await bridge.feed_provider_event({"type": "response.audio.delta", "delta": chunk})
        assert client_sent[-1]["type"] == "audio.delta"

    @pytest.mark.asyncio
    async def test_the_agents_own_voice_in_the_microphone_is_dropped(self):
        bridge, provider_sent, client_sent = self._make_bridge()
        await self._agent_speaking(bridge)
        client_sent.clear()
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "e1"})
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.delta", "item_id": "e1", "text": "店里特别忙", "stash": "一下午"})
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_stopped", "item_id": "e1"})
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed", "item_id": "e1", "transcript": "店里特别忙一下午做了。"})
        assert client_sent == []
        # Deleted once the reply is done: deleting during generation can
        # close the DashScope session.
        assert provider_sent == []
        await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed"}})
        assert [e["type"] for e in provider_sent] == ["conversation.item.delete"]
        assert provider_sent[0]["item_id"] == "e1"
        assert "echo_dropped=1" in bridge.mic_summary
        # Only the agent's own reply was recorded, not its echo.
        assert [p["role"] for p in bridge.transcript_pairs] == ["assistant"]

    @pytest.mark.asyncio
    async def test_a_backchannel_over_the_agent_gets_no_reply(self):
        bridge, provider_sent, client_sent = self._make_bridge()
        await self._agent_speaking(bridge)
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "b1"})
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed", "item_id": "b1", "transcript": "嗯。"})
        await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed"}})
        assert [e["type"] for e in provider_sent] == ["conversation.item.delete"]
        assert not any(e["type"] == "speech.started" for e in client_sent[1:])
        assert [p["role"] for p in bridge.transcript_pairs] == ["assistant"]

    @pytest.mark.asyncio
    async def test_a_short_answer_right_after_the_agent_is_the_user(self):
        import time as _time
        bridge, provider_sent, client_sent = self._make_bridge()
        await self._agent_speaking(bridge, "你吃了吗")
        await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed"}})
        bridge._playback_until = _time.monotonic() - 0.1  # playback just ended
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "t1"})
        assert not any(e["type"] == "speech.started" for e in client_sent)
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.completed", "item_id": "t1", "transcript": "吃了"})
        await bridge._reply_task
        assert client_sent[-2:][0]["type"] == "speech.started" or any(
            e["type"] == "speech.started" for e in client_sent)
        assert bridge.transcript_pairs[-1] == {**bridge.transcript_pairs[-1], "role": "user", "text": "吃了"}
        assert provider_sent[-1]["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_speech_after_generation_finished_sends_no_cancel(self):
        """Playback-only tail: the client clears it, the provider has nothing to cancel."""
        bridge, provider_sent, client_sent = self._make_bridge()
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started"})
        assert provider_sent == []
        assert client_sent == [{"type": "speech.started"}]

    @pytest.mark.asyncio
    async def test_barge_in_survives_a_failed_cancel(self):
        bridge, _, client_sent = self._make_bridge()

        async def broken(event):
            raise RuntimeError("socket closed")

        await self._agent_speaking(bridge)
        bridge._send_provider = broken
        await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
        await bridge.feed_provider_event({
            "type": "conversation.item.input_audio_transcription.delta", "item_id": "u1", "text": "你等一下", "stash": ""})
        assert client_sent[-1] == {"type": "speech.started"}

    @pytest.mark.asyncio
    async def test_client_interrupt(self):
        """Client-initiated interrupt cancels provider response."""
        bridge, provider_sent, client_sent = self._make_bridge()
        bridge._responding = True
        bridge._reply_seen = True
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
        """Calls use the agent's library voice, the one its voice messages use."""
        source = _VOICE.read_text()
        assert 'getattr(conv.agent, "ttsVoiceId"' not in source
        assert "ensure_agent_voice(conv.agent)" in source

    def test_no_provider_fallback_remains(self):
        """DashScope is the only voice provider: no StepFun setting or fallback."""
        from app.config import Settings

        source = _VOICE.read_text()
        assert "fallback_voice" not in source and "stepfun" not in source.lower()
        assert not [name for name in Settings.model_fields if name.startswith("stepfun")]

    def test_endpoint_has_idle_timeout(self):
        """voice.py should have idle watchdog."""
        source = _VOICE.read_text()
        assert "idle_watchdog" in source
        assert "_IDLE_TIMEOUT_SECONDS" in source

    def test_text_chat_not_affected(self):
        """Text chat ws.py must not import realtime_voice modules."""
        source = _CHAT_WS.read_text()
        assert "realtime_voice" not in source

    def test_deploy_no_longer_ships_stepfun_settings(self):
        deploy = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"
        text = deploy.read_text()
        assert "STEPFUN" not in text
        assert "DASHSCOPE_WORKSPACE_ID=${{ vars.DASHSCOPE_WORKSPACE_ID }}" in text


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


def _endpoint_fixtures(monkeypatch, endpoint, conv):
    principal = SimpleNamespace(role="user", user_id="user-1", expires_at=float("inf"))

    async def ok(*_, **__):
        return True

    async def find_unique(**_):
        return conv

    async def principal_for(*_):
        return principal

    async def context(**_):
        return SimpleNamespace(name="阿宁"), "instructions", [], "cue"

    async def lock(_):
        return "lock-token"

    monkeypatch.setattr(endpoint, "is_redis_healthy", lambda: True)
    monkeypatch.setattr(endpoint, "offered_ticket", lambda _: "ticket")
    monkeypatch.setattr(endpoint, "origin_allowed", lambda _: True)
    monkeypatch.setattr(endpoint, "consume_ticket", principal_for)
    monkeypatch.setattr(endpoint.db, "conversation", SimpleNamespace(find_unique=find_unique), raising=False)
    monkeypatch.setattr(endpoint, "get_patience", ok)
    monkeypatch.setattr(endpoint, "get_patience_zone", lambda _: "normal")
    monkeypatch.setattr(endpoint, "_accounts_current", ok)
    monkeypatch.setattr(endpoint.settings, "dashscope_api_key", "sk-test")
    monkeypatch.setattr(endpoint, "acquire_call", lock)
    monkeypatch.setattr(endpoint, "load_call_context", context)
    monkeypatch.setattr(endpoint, "connect_realtime", lambda *_: _FakeProviderConnection())
    monkeypatch.setattr(
        endpoint, "get_effective_realtime_model",
        AsyncMock(return_value="qwen3.8-omni-flash-realtime"),
    )


@pytest.mark.parametrize("lookup", ["Serena", None])
async def test_call_dials_the_agent_voice_or_reports_an_empty_library(monkeypatch, lookup):
    from app.api.realtime import voice as endpoint
    from app.services.realtime_voice.provider import RealtimeSetupError

    agent = SimpleNamespace(id="agent-1", userId="user-1", status="active", gender="female")
    conv = SimpleNamespace(id="conv-1", userId="user-1", isDeleted=False, agent=agent, workspaceId=None)
    seen: dict = {}
    released: list[tuple] = []
    _endpoint_fixtures(monkeypatch, endpoint, conv)

    async def release_call(conversation_id, token):
        released.append((conversation_id, token))

    async def configure_session(sock, *, voice, **kwargs):
        seen.update(voice=voice, **kwargs)
        raise RealtimeSetupError("session rejected")

    async def ensure_agent_voice(agent):
        if lookup is None:
            raise LookupError("No enabled female voice in the library")
        return lookup

    monkeypatch.setattr(endpoint, "release_call", release_call)
    monkeypatch.setattr(endpoint, "configure_session", configure_session)
    monkeypatch.setattr(endpoint, "ensure_agent_voice", ensure_agent_voice)

    ws = _FakeVoiceSocket()
    await endpoint.voice_websocket(ws, "conv-1")

    assert ws.accepted
    if lookup is None:
        assert "voice" not in seen
        assert ws.sent == [{"type": "error", "code": "not_configured", "message": "音色库没有可用音色"}]
    else:
        assert seen["voice"] == "Serena"
        assert seen["vad_threshold"] == endpoint.settings.dashscope_realtime_vad_threshold
        assert seen["temperature"] == endpoint.settings.dashscope_realtime_temperature
        assert ws.sent == [{"type": "error", "code": "provider_unavailable", "message": "实时语音暂时连不上"}]
    assert ws.closed_with == 1011
    assert released == [("conv-1", "lock-token")]


async def test_call_without_a_key_is_not_configured(monkeypatch):
    from app.api.realtime import voice as endpoint

    agent = SimpleNamespace(id="agent-1", userId="user-1", status="active", gender="female")
    conv = SimpleNamespace(id="conv-1", userId="user-1", isDeleted=False, agent=agent, workspaceId=None)
    _endpoint_fixtures(monkeypatch, endpoint, conv)
    monkeypatch.setattr(endpoint.settings, "dashscope_api_key", " ")
    ws = _FakeVoiceSocket()
    await endpoint.voice_websocket(ws, "conv-1")
    assert ws.sent[0]["code"] == "not_configured" and "DASHSCOPE_API_KEY" in ws.sent[0]["message"]


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
    """RealtimeSocket stand-in that replays provider events per session.update."""

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


async def test_configure_session_sends_the_voice_and_waits_for_the_ack():
    from app.services.realtime_voice.provider import configure_session

    sock = _ScriptedProvider([{"type": "session.updated"}])
    await configure_session(sock, instructions="hi", voice="Serena", history=[])
    assert [e["session"]["voice"] for e in sock.sent] == ["Serena"]


async def test_a_rejected_session_raises():
    from app.services.realtime_voice.provider import RealtimeSetupError, configure_session

    sock = _ScriptedProvider([{"type": "error", "error": {"message": "bad session"}}])
    with pytest.raises(RealtimeSetupError, match="bad session"):
        await configure_session(sock, instructions="hi", voice="Serena", history=[])


async def test_bridge_sums_provider_usage_per_modality():
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
        {"input_tokens": 1087, "output_tokens": 35,
         "input_tokens_details": {"text_tokens": 1000, "audio_tokens": 87},
         "output_tokens_details": {"text_tokens": 9, "audio_tokens": 26}},
        # No details: the tokens count as audio, the dearer kind.
        {"input_tokens": 50, "output_tokens": 7},
        None,
    ):
        response = {"status": "completed", "output": []}
        if usage is not None:
            response["usage"] = usage
        await bridge.feed_provider_event({"type": "response.done", "response": response})

    assert bridge.usage == {
        "input_text": 1000, "input_audio": 137,
        "output_text": 9, "output_audio": 33, "responses": 2,
    }


_CALL_USAGE = {"input_text": 1000, "input_audio": 137, "output_text": 9, "output_audio": 33, "responses": 2}


async def test_call_usage_is_written_as_one_realtime_row(monkeypatch):
    from app.api.realtime import voice as endpoint
    from app.services.llm import usage_repo

    written: list[dict] = []

    async def persist(**kwargs):
        written.append(kwargs)

    monkeypatch.setattr(usage_repo, "persist_usage", persist)
    model = "qwen3.8-omni-flash-realtime"
    await endpoint._record_call_usage(
        _CALL_USAGE, model=model, conversation_id="conv-1", agent_id="a1", user_id="u1",
    )
    await endpoint._record_call_usage(
        {**_CALL_USAGE, "responses": 0}, model=model, conversation_id="conv-1", agent_id="a1", user_id="u1",
    )

    assert len(written) == 1
    row = written[0]
    assert row["scope"] == "realtime_voice"
    # Text and audio tokens are priced separately: the ':audio' bucket.
    assert row["summary"]["tokens_by_model"] == {
        model: {"input": 1000, "output": 9, "cached_input": 0},
        f"{model}:audio": {"input": 137, "output": 33, "cached_input": 0},
    }
    assert row["summary"]["input_tokens"] == 1137 and row["summary"]["output_tokens"] == 42
    assert row["summary"]["call_count"] == 2


def test_billing_prices_text_and_audio_tokens_separately(monkeypatch):
    from app.services.llm import pricing
    from app.services.realtime_voice import billing

    prices = {"m": (1.5, 4.5), "m:audio": (6.0, 12.0)}
    monkeypatch.setattr(pricing, "estimate_cost_cny", lambda name, i, o, cached_input_tokens=0: (
        (i * prices[name][0] + o * prices[name][1]) / 1_000_000))
    split = billing.split_usage({
        "input_tokens": 100, "output_tokens": 40,
        "input_tokens_details": {"text_tokens": 100},
        "output_tokens_details": {"text_tokens": 10, "audio_tokens": 30},
    })
    assert billing.cost_cny("m", split) == round((100 * 1.5 + 10 * 4.5 + 30 * 12.0) / 1_000_000, 6)
    assert billing.is_price_only("m:audio") and not billing.is_price_only("m")
    assert billing.split_usage({"input_tokens": -3, "output_tokens": "x"}) == {
        "input_text": 0, "input_audio": 0, "output_text": 0, "output_audio": 0}


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
    assert runtime_config.resolve_config_sync(agent_id=None).realtime_model == "qwen3.8-omni-flash-realtime"
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realtimeModel": "qwen3.9-omni-flash-realtime"})
    assert runtime_config.resolve_config_sync(agent_id=None).realtime_model == "qwen3.9-omni-flash-realtime"


def test_realtime_is_a_dashscope_model_kind():
    from app.api.admin import model_registry

    model_registry._validate_model_metadata("realtime", "per_million_tokens", provider="dashscope")
    with pytest.raises(Exception) as exc:
        model_registry._validate_model_metadata("realtime", "per_10k_characters", provider="dashscope")
    assert getattr(exc.value, "status_code", None) == 400


async def test_finished_call_meters_usage_and_schedules_extraction(monkeypatch):
    """A call that reaches the pumps meters its usage once at hang-up."""
    from app.api.realtime import voice as endpoint

    agent = SimpleNamespace(id="agent-1", userId="user-1", status="active", gender="female")
    conv = SimpleNamespace(id="conv-1", userId="user-1", isDeleted=False, agent=agent, workspaceId=None)
    metered: list[dict] = []
    extracted: list[dict] = []
    _endpoint_fixtures(monkeypatch, endpoint, conv)

    from app.services.realtime_voice import call_stats

    built: dict = {}

    class FakeBridge:
        transcript_pairs = [{"role": "user", "text": "hi"}]
        message_ids = ["msg-u1", "msg-a1"]
        mic_summary = "bytes=0 chunks=0 seconds=0.0 peak_dbfs=-inf rms_dbfs=-inf speech_turns=0 echo_dropped=0"
        usage = _CALL_USAGE
        stats_summary = {"schema_version": 1}

        def __init__(self, stats):
            self.stats = stats

        async def aclose(self):
            pass

    def make_bridge(**kwargs):
        built.update(kwargs)
        return FakeBridge(kwargs["stats"])

    stored: list = []

    async def store(stats, summary, **kwargs):
        stored.append((stats, summary, kwargs))

    async def none(*_, **__):
        return None

    async def record(usage, **kwargs):
        metered.append({"usage": usage, **kwargs})

    monkeypatch.setattr(endpoint, "release_call", none)
    monkeypatch.setattr(endpoint, "ensure_agent_voice", AsyncMock(return_value="Serena"))
    monkeypatch.setattr(endpoint, "configure_session", none)
    monkeypatch.setattr(endpoint, "_bridge", make_bridge)
    monkeypatch.setattr(call_stats, "persist", store)
    monkeypatch.setattr(endpoint, "schedule_user_present", lambda _: None)
    monkeypatch.setattr(endpoint, "_run_pumps", none)
    monkeypatch.setattr(endpoint, "_record_call_usage", record)
    monkeypatch.setattr(endpoint, "schedule_end_of_call_extraction", lambda **kw: extracted.append(kw))

    ws = _FakeVoiceSocket()
    await endpoint.voice_websocket(ws, "conv-1")

    assert ws.sent[0]["type"] == "ready" and ws.sent[0]["voice"] == "Serena"
    assert ws.sent[0]["model"] == "qwen3.8-omni-flash-realtime"
    assert metered == [{
        "usage": _CALL_USAGE, "model": "qwen3.8-omni-flash-realtime",
        "conversation_id": "conv-1", "agent_id": "agent-1", "user_id": "user-1",
    }]
    assert extracted == [{
        "user_id": "user-1", "workspace_id": None, "conversation_id": "conv-1",
        "agent_id": "agent-1", "message_ids": ["msg-u1", "msg-a1"],
    }]
    # The call's diagnostics carry the environment it ran in, and are stored once at hang-up.
    stats = built["stats"]
    assert (stats.model, stats.vad_threshold, stats.silence_ms) == ("qwen3.8-omni-flash-realtime", 0.5, 600)
    assert stored == [(stats, {"schema_version": 1}, {
        "user_id": "user-1", "workspace_id": None, "conversation_id": "conv-1", "agent_id": "agent-1"})]


async def test_call_usage_metering_failure_is_swallowed(monkeypatch, caplog):
    from app.api.realtime import voice as endpoint
    from app.services.llm import usage_repo

    monkeypatch.setattr(usage_repo, "persist_usage", AsyncMock(side_effect=RuntimeError("db down")))
    await endpoint._record_call_usage(
        _CALL_USAGE,
        model="m", conversation_id="conv-1", agent_id="a1", user_id="u1",
    )
    assert "usage metering failed" in caplog.text


async def test_configure_session_sends_config_then_seeds_history():
    from app.services.realtime_voice.provider import configure_session

    sock = _ScriptedProvider([{"type": "session.updated"}])
    original_send = sock.send_event

    async def send(event):
        await original_send(event)
        if event.get("type") == "conversation.item.create":
            sock.inbox.append({"type": "conversation.item.created", "item": {"id": "srv", "role": "user"}})

    sock.send_event = send
    await configure_session(
        sock, instructions="hi", voice="Serena",
        history=[{"role": "user", "text": "你好"}], temperature=0.7,
    )
    assert [e["type"] for e in sock.sent] == ["session.update", "conversation.item.create"]
    assert sock.sent[0]["session"]["voice"] == "Serena"
    assert sock.sent[0]["session"]["temperature"] == 0.7


async def test_configure_session_reports_a_rejected_session():
    from app.services.realtime_voice.provider import RealtimeSetupError, configure_session

    sock = _ScriptedProvider([{"type": "error", "error": {"message": "bad session"}}])
    with pytest.raises(RealtimeSetupError, match="bad session"):
        await configure_session(sock, instructions="hi", voice="Serena", history=[])


async def test_effective_realtime_model_reads_runtime_config(monkeypatch):
    from app.services import runtime_config

    monkeypatch.setattr(runtime_config, "ensure_loaded", AsyncMock())
    monkeypatch.setattr(runtime_config, "_current_snapshot", runtime_config.ContextVar("s", default=None))
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realtimeModel": "qwen3.9-omni-flash-realtime"})
    assert await runtime_config.get_effective_realtime_model() == "qwen3.9-omni-flash-realtime"
