import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.speech_output import delivery, policy, voices


def prepared_voice():
    return SimpleNamespace(
        transcript="你好", metadata={"id": "attachment", "kind": "audio"},
        attachment=SimpleNamespace(id="attachment", storage_key="voice.wav"),
        user_id="user", conversation_id="conversation",
    )


@pytest.mark.asyncio
async def test_discard_never_deletes_bytes_of_a_bound_attachment(monkeypatch):
    monkeypatch.setattr(delivery.media_repo, "delete_unbound_attachment", AsyncMock(return_value=None))
    deleted = []
    monkeypatch.setattr(delivery.media_storage, "delete_media_file", deleted.append)
    await delivery.discard_prepared_voice_output(prepared_voice())
    assert not deleted


@pytest.mark.asyncio
async def test_discard_removes_only_the_deleted_unbound_attachment(monkeypatch):
    monkeypatch.setattr(delivery.media_repo, "delete_unbound_attachment", AsyncMock(return_value=SimpleNamespace(storage_key="deleted.wav")))
    deleted = []
    monkeypatch.setattr(delivery.media_storage, "delete_media_file", deleted.append)
    await delivery.discard_prepared_voice_output(prepared_voice())
    assert deleted == ["deleted.wav"]


@pytest.mark.asyncio
@pytest.mark.parametrize("updated", [0, 1])
async def test_voice_binding_requires_one_updated_attachment(monkeypatch, updated):
    prepared = prepared_voice()
    prepared.speech = SimpleNamespace(request_id="request")
    monkeypatch.setattr(delivery.media_repo, "db", SimpleNamespace(execute_raw=AsyncMock(return_value=updated)))
    link = AsyncMock()
    monkeypatch.setattr(delivery, "link_tts_usage_to_message", link)
    if updated:
        await delivery.bind_prepared_voice_output(prepared, message_id="message")
        link.assert_awaited_once_with(request_id="request", message_id="message")
    else:
        with pytest.raises(LookupError, match="binding"):
            await delivery.bind_prepared_voice_output(prepared, message_id="message")
        link.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("save failed"), asyncio.CancelledError()])
async def test_short_circuit_save_failure_cleans_voice_and_propagates(monkeypatch, defer, error):
    from app.services.chat import multi_intent

    prepared = prepared_voice()
    prepare = AsyncMock(return_value=prepared)
    discard = AsyncMock()
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    monkeypatch.setattr(delivery, "prepare_voice_output", prepare)
    monkeypatch.setattr(delivery, "discard_prepared_voice_output", discard)
    with pytest.raises(type(error)):
        await multi_intent.short_circuit_reply(
            "你好", "conversation", "agent", "user", AsyncMock(side_effect=error),
            agent=SimpleNamespace(id="agent"), reply_context={"client_supports_voice": True},
            voice_context=policy.VoiceContext.NORMAL_CHAT, defer_turn_finalization=defer,
        )
    discard.assert_awaited_once_with(prepared)
    assert prepare.await_args.kwargs["detect_missing_emotion"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("save failed"), asyncio.CancelledError()])
async def test_proactive_save_failure_cleans_voice(monkeypatch, error):
    from app.services.proactive import emit

    prepared = prepared_voice()
    discard = AsyncMock()
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    monkeypatch.setattr(delivery, "prepare_voice_output", AsyncMock(return_value=prepared))
    monkeypatch.setattr(delivery, "discard_prepared_voice_output", discard)
    monkeypatch.setattr(emit, "snapshot_prompt_render_traces", lambda: [])
    monkeypatch.setattr(emit, "db", SimpleNamespace(
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(id="agent"))),
        message=SimpleNamespace(create=AsyncMock(side_effect=error)),
    ))
    with pytest.raises(type(error)):
        await emit.emit_proactive_message(
            conversation_id="conversation", user_id="user", agent_id="agent",
            workspace_id=None, message="你好", trigger_type="test",
        )
    discard.assert_awaited_once_with(prepared)


@pytest.mark.asyncio
async def test_concurrent_voice_assignment_keeps_existing_admin_voice(monkeypatch):
    database = SimpleNamespace(
        query_raw=AsyncMock(side_effect=[[{"voice_id": "random-voice"}], [{"tts_voice_id": "admin-voice"}]]),
        execute_raw=AsyncMock(return_value=0),
    )
    monkeypatch.setattr(voices, "db", database)
    agent = SimpleNamespace(ttsVoiceId=None)
    assigned = await voices.assign_random_voice(agent_id="agent", gender="female", agent=agent)
    assert assigned == agent.ttsVoiceId == "admin-voice"


@pytest.mark.asyncio
async def test_cancelled_attachment_creation_removes_file_after_metering(monkeypatch):
    speech = SimpleNamespace(audio=b"wav", mime="audio/wav", duration_milliseconds=500, request_id="request")
    metered = AsyncMock()
    deleted = []
    monkeypatch.setattr(delivery, "resolve_agent_tts_model", AsyncMock(return_value=voices.QWEN_AUDIO_TTS_MODEL))
    monkeypatch.setattr(delivery, "get_agent_tts_settings", AsyncMock(return_value=voices.AgentTtsSettings("voice", 1, 1, 50, 0, None, False, 1)))
    monkeypatch.setattr(delivery, "synthesize_speech", AsyncMock(return_value=speech))
    monkeypatch.setattr(delivery, "record_tts_usage", metered)
    monkeypatch.setattr(delivery.media_storage, "save_audio_blob", lambda **kwargs: "voice.wav")
    monkeypatch.setattr(delivery.media_storage, "media_url", lambda key: key)
    monkeypatch.setattr(delivery.media_storage, "delete_media_file", deleted.append)
    monkeypatch.setattr(delivery.media_repo, "create_generated_audio_attachment", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await delivery.prepare_voice_output(
            text="你好", user_id="user", agent=SimpleNamespace(id="agent"),
            conversation_id="conversation", source="chat",
        )
    metered.assert_awaited_once()
    assert deleted == ["voice.wav"]


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_ordinary_chat_save_failure_cleans_prepared_voice(monkeypatch, executor):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from tests.graph_harness_support import configure_chat

    io = configure_chat(monkeypatch)
    monkeypatch.setattr(settings, "chat_executor", executor)
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    prepared = prepared_voice()
    discard = AsyncMock()
    monkeypatch.setattr(delivery, "prepare_voice_output", AsyncMock(return_value=prepared))
    monkeypatch.setattr(delivery, "discard_prepared_voice_output", discard)
    chat._save_replies.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        async for _ in chat.stream_chat_response(
            "c-1", "今晚想聊天", io.agent, "u-1", reply_context={"client_supports_voice": True},
        ):
            pass
    discard.assert_awaited_once_with(prepared)
