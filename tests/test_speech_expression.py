from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.speech_output import delivery, style, voices


def plan(text="我听着呢。", emotion="高兴", intensity=50, **kwargs):
    return style.build_speech_plan(
        text, emotion, intensity,
        instruction=kwargs.get("instruction"),
        enabled=kwargs.get("enabled", True), scale=kwargs.get("scale", 1),
    )


def test_emotion_strength_changes_style_without_overacting():
    weak, mild, medium, strong = [plan(intensity=n) for n in (10, 30, 55, 80)]
    assert weak.instruction == style.DEFAULT_STYLE_INSTRUCTION
    assert "略带开心轻快" in mild.instruction
    assert "带有开心轻快" in medium.instruction
    assert "明显开心轻快" in strong.instruction
    assert weak.text == mild.text == medium.text == "我听着呢。"
    assert strong.text == "[excited]我听着呢。"
    assert all(style.instruction_billable_characters(p.instruction) <= 100 for p in (weak, mild, medium, strong))


@pytest.mark.parametrize("intensity,scale", [(float("nan"), 1), (float("inf"), 1), (50, float("inf")), ("invalid", 1), (-10, 1)])
def test_invalid_emotion_never_crashes_or_adds_controls(intensity, scale):
    assert plan(intensity=intensity, scale=scale).text == "我听着呢。"
    assert plan(intensity=intensity, scale=scale).instruction == style.DEFAULT_STYLE_INSTRUCTION


def test_scale_and_switch_are_honored():
    assert plan(intensity=40, scale=2).text.startswith("[excited]")
    assert plan(intensity=90, scale=0).instruction == style.DEFAULT_STYLE_INSTRUCTION
    assert plan("哈哈，太棒了！", intensity=90, enabled=False).text == "哈哈，太棒了！"


def test_custom_instruction_is_never_truncated():
    custom = "真" * 50
    assert plan(instruction=custom).instruction == custom
    assert plan(instruction="轻松自然").instruction.startswith("轻松自然")


@pytest.mark.parametrize("emotion", ["高兴", "戏谑"])
def test_laughter_requires_explicit_cue(emotion):
    result = plan("哈哈哈，居然被你猜中了，哈哈！", emotion, 50)
    assert result.text == "[giggles]居然被你猜中了，哈哈！"
    assert "[giggles]" not in plan("真开心！", emotion, 90).text
    assert plan("哈哈", emotion, 50).text == "哈哈"


def test_sigh_and_anxiety_do_not_force_trembling():
    assert plan("唉，有点担心明天。", "焦虑", 85).text == "[sighing]有点担心明天。"
    assert plan("有点担心明天。", "焦虑", 85).text == "有点担心明天。"
    assert plan("唉声叹气也没用。", "悲伤", 50).text == "唉声叹气也没用。"


def test_speech_cleanup_preserves_punctuation_but_removes_injected_controls():
    assert plan("[shouting] 你好😊， 等等…… [项目名]", "中性", 0).text == "你好， 等等…… [项目名]"
    with pytest.raises(ValueError, match="speakable"):
        plan("😊[crying]")


@pytest.mark.asyncio
async def test_emotion_detection_only_runs_when_required(monkeypatch):
    from app.services.chat import intent_replies

    detect = AsyncMock(return_value={"emotion": "感激", "intensity": 55})
    monkeypatch.setattr(intent_replies, "ai_reply_emotion", detect)
    assert await style.resolve_voice_emotion("谢谢", None, None, enabled=True, detect_missing=True) == ("感激", 55)
    for enabled, missing in ((False, True), (True, False)):
        await style.resolve_voice_emotion("谢谢", None, None, enabled=enabled, detect_missing=missing)
    await style.resolve_voice_emotion("谢谢", "中性", 0, enabled=True, detect_missing=True)
    detect.assert_awaited_once()


@pytest.mark.asyncio
async def test_emotion_failure_keeps_voice_available(monkeypatch):
    from app.services.chat import intent_replies

    monkeypatch.setattr(intent_replies, "ai_reply_emotion", AsyncMock(side_effect=RuntimeError("offline")))
    assert await style.resolve_voice_emotion("你好", None, None, enabled=True, detect_missing=True) == (None, None)


@pytest.mark.asyncio
async def test_emotion_detection_deadline_cancels_slow_classifier(monkeypatch):
    import asyncio
    from app.services.chat import intent_replies

    cancelled = asyncio.Event()

    async def slow(text):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    timeout = asyncio.timeout
    monkeypatch.setattr(style.asyncio, "timeout", lambda seconds: timeout(0.01))
    monkeypatch.setattr(intent_replies, "ai_reply_emotion", slow)
    assert await style.resolve_voice_emotion("你好", None, None, enabled=True, detect_missing=True) == (None, None)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_incompatible_runtime_model_is_rejected_before_synthesis(monkeypatch):
    from app.services import runtime_config

    monkeypatch.setattr(runtime_config, "get_effective_tts_model", AsyncMock(return_value="qwen-audio-3.0-tts-flash"))
    with pytest.raises(RuntimeError, match="不兼容"):
        await voices.resolve_agent_tts_model()


@pytest.mark.asyncio
@pytest.mark.parametrize("source,detect_missing", [("chat", False), ("chat", True), ("proactive", False)])
async def test_delivery_preserves_transcript_and_shared_plan(monkeypatch, source, detect_missing):
    from app.services.chat import intent_replies

    speech = SimpleNamespace(audio=b"wav", mime="audio/wav", duration_milliseconds=500, request_id="req")
    synthesize = AsyncMock(return_value=speech)
    detect = AsyncMock(return_value={"emotion": "高兴", "intensity": 50})
    monkeypatch.setattr(intent_replies, "ai_reply_emotion", detect)
    monkeypatch.setattr(delivery, "resolve_agent_tts_model", AsyncMock(return_value=voices.QWEN_AUDIO_TTS_MODEL))
    monkeypatch.setattr(delivery, "get_agent_tts_settings", AsyncMock(return_value=voices.AgentTtsSettings("voice", 1, 1, 50, 42, None, True, 1)))
    monkeypatch.setattr(delivery, "synthesize_speech", synthesize)
    monkeypatch.setattr(delivery, "record_tts_usage", AsyncMock())
    monkeypatch.setattr(delivery.media_storage, "save_audio_blob", lambda **kw: "key")
    monkeypatch.setattr(delivery.media_storage, "media_url", lambda key: "/voice.wav")
    monkeypatch.setattr(delivery.media_repo, "create_generated_audio_attachment", AsyncMock(return_value=SimpleNamespace(id="attachment")))
    monkeypatch.setattr(delivery, "attachment_to_metadata", lambda value: {"id": value.id})
    prepared = await delivery.prepare_voice_output(
        text="哈哈，太棒了😊！", user_id="user", agent=SimpleNamespace(id="agent"),
        conversation_id="conversation", source=source, detect_missing_emotion=detect_missing,
    )
    assert prepared.transcript == "哈哈，太棒了😊！"
    should_detect = source != "chat" or detect_missing
    assert detect.await_count == int(should_detect)
    expected = plan("哈哈，太棒了😊！", "高兴" if should_detect else None, 50 if should_detect else None)
    assert synthesize.await_args.kwargs["text"] == expected.text
    assert synthesize.await_args.kwargs["instruction"] == expected.instruction
    assert synthesize.await_args.kwargs["seed"] == 42
    assert synthesize.await_args.kwargs["model"] == voices.QWEN_AUDIO_TTS_MODEL


@pytest.mark.asyncio
async def test_preview_uses_the_same_plan_and_runtime_model(monkeypatch):
    from app.api.admin import tts

    payload = tts.AgentTtsPreviewPayload(
        voice_profile_id="profile", rate=1, pitch=1, volume=50, seed=42,
        emotion_scale=1, text="哈哈，太棒了😊！", emotion="高兴", intensity=50,
    )
    speech = SimpleNamespace(audio=b"wav", mime="audio/wav", duration_milliseconds=500, billable_characters=10, cost_cny=0.01)
    synthesize = AsyncMock(return_value=speech)
    monkeypatch.setattr(tts, "_voice_profile", AsyncMock(return_value={"voice_id": "voice"}))
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[{"user_id": "user"}])))
    monkeypatch.setattr(tts, "resolve_agent_tts_model", AsyncMock(return_value=voices.QWEN_AUDIO_TTS_MODEL))
    monkeypatch.setattr(tts, "synthesize_speech", synthesize)
    monkeypatch.setattr(tts, "record_tts_usage", AsyncMock())
    response = await tts.preview_agent_tts("agent", payload)
    expected = plan(payload.text, payload.emotion, payload.intensity)
    assert synthesize.await_args.kwargs["text"] == expected.text
    assert synthesize.await_args.kwargs["instruction"] == expected.instruction
    assert synthesize.await_args.kwargs["model"] == voices.QWEN_AUDIO_TTS_MODEL
    assert response.body == b"wav"
