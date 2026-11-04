"""Admin-chosen default voices and per-agent voice-message probability."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
import pytest

from app.api.admin import tts
from app.services.speech_output import policy, voices
from app.services.speech_output.client import SpeechSynthesisError, SynthesizedSpeech


def _profile(**overrides):
    return {
        "id": "p1", "provider": "stepfun", "model": "stepaudio-2.5-tts",
        "voice_id": "linjiajiejie", "gender": "female", "source": "system",
        "enabled": True, "is_default": False, "display_name": "邻家姐姐",
        **overrides,
    }


def _speech():
    return SynthesizedSpeech(
        audio=b"RIFFwav", mime="audio/wav", duration_milliseconds=1200,
        model="stepaudio-2.5-tts", voice_id="linjiajiejie", request_id="trace-1",
        raw_characters=10, billable_characters=10, unit_price_cny=5.8, cost_cny=0.0058,
    )


async def test_default_voice_is_read_from_the_library(monkeypatch):
    query = AsyncMock(side_effect=[[{"voice_id": "wenrounansheng"}], [], [{"voice_id": ""}]])
    monkeypatch.setattr(voices, "db", SimpleNamespace(query_raw=query))
    assert await voices.default_voice_for_gender("male") == "wenrounansheng"
    assert query.await_args.args[1:] == ("stepfun", "stepaudio-2.5-tts", "male")
    assert "is_default = true" in query.await_args.args[0]
    assert await voices.default_voice_for_gender("女") is None
    assert query.await_args.args[3] == "female"
    assert await voices.default_voice_for_gender(None) is None


async def test_assignment_prefers_the_default_then_any_enabled_voice(monkeypatch):
    query = AsyncMock(side_effect=[[{"voice_id": "linjiajiejie"}]])
    execute = AsyncMock(return_value=1)
    monkeypatch.setattr(voices, "db", SimpleNamespace(query_raw=query, execute_raw=execute))
    assert await voices.assign_random_voice(agent_id="a1", gender="female") == "linjiajiejie"
    assert query.await_count == 1

    query = AsyncMock(side_effect=[[], [{"voice_id": "jingdiannvsheng"}]])
    monkeypatch.setattr(voices, "db", SimpleNamespace(query_raw=query, execute_raw=execute))
    assert await voices.assign_random_voice(agent_id="a2", gender="female") == "jingdiannvsheng"
    assert "ORDER BY random()" in query.await_args.args[0]


async def test_assignment_fails_loudly_without_any_enabled_voice(monkeypatch):
    query = AsyncMock(side_effect=[[], []])
    execute = AsyncMock()
    monkeypatch.setattr(voices, "db", SimpleNamespace(query_raw=query, execute_raw=execute))
    with pytest.raises(LookupError, match="No enabled male voice"):
        await voices.assign_random_voice(agent_id="a1", gender="male")
    execute.assert_not_awaited()


async def test_agent_probability_overrides_the_global_value(monkeypatch):
    query = AsyncMock(return_value=[{"tts_output_probability": 100}])
    monkeypatch.setattr(policy, "db", SimpleNamespace(query_raw=query))
    monkeypatch.setattr(policy, "effective_tts_probability", AsyncMock(return_value=0))
    assert await policy.should_generate_voice(
        context=policy.VoiceContext.NORMAL_CHAT, client_supports_voice=True, agent_id="a1",
    )
    assert query.await_args.args[1] == "a1"


@pytest.mark.parametrize("rows", [[], [{"tts_output_probability": None}], RuntimeError("db down")])
async def test_agent_without_override_inherits_the_global_value(monkeypatch, rows):
    query = AsyncMock(side_effect=rows) if isinstance(rows, Exception) else AsyncMock(return_value=rows)
    monkeypatch.setattr(policy, "db", SimpleNamespace(query_raw=query))
    monkeypatch.setattr(policy, "effective_tts_probability", AsyncMock(return_value=100))
    assert await policy.agent_tts_probability("a1") is None
    assert await policy.should_generate_voice(
        context=policy.VoiceContext.PROACTIVE_CHAT, client_supports_voice=True, agent_id="a1",
    )


async def test_no_agent_skips_the_lookup(monkeypatch):
    query = AsyncMock()
    monkeypatch.setattr(policy, "db", SimpleNamespace(query_raw=query))
    assert await policy.agent_tts_probability(None) is None
    query.assert_not_awaited()


class _Tx:
    def __init__(self):
        self.execute_raw = AsyncMock(return_value=1)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def test_set_default_clears_the_previous_default_in_one_transaction(monkeypatch):
    tx = _Tx()
    voices_list = [_profile(is_default=True)]
    database = SimpleNamespace(
        query_raw=AsyncMock(side_effect=[[_profile()], voices_list]),
        tx=lambda: tx,
    )
    monkeypatch.setattr(tts, "db", database)

    assert await tts.set_default_voice_profile("p1") == {"voices": voices_list}
    clear, mark = tx.execute_raw.await_args_list
    assert "is_default = false" in clear.args[0]
    assert clear.args[1:] == ("stepfun", "stepaudio-2.5-tts", "female", "p1")
    assert "is_default = true" in mark.args[0] and mark.args[1] == "p1"


async def test_disabled_voice_cannot_become_default(monkeypatch):
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[_profile(enabled=False)])))
    with pytest.raises(HTTPException) as exc:
        await tts.set_default_voice_profile("p1")
    assert exc.value.status_code == 422


@pytest.mark.parametrize("patch", [
    tts.VoiceProfilePatch(enabled=False),
    tts.VoiceProfilePatch(gender="male"),
])
async def test_default_voice_cannot_leave_its_pool(monkeypatch, patch):
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[_profile(is_default=True)])))
    with pytest.raises(HTTPException) as exc:
        await tts.update_voice_profile("p1", patch)
    assert exc.value.status_code == 409


async def test_default_clone_cannot_be_removed(monkeypatch):
    profile = _profile(source="cloned", is_default=True)
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[profile])))
    with pytest.raises(HTTPException) as exc:
        await tts.delete_voice_profile("p1")
    assert exc.value.status_code == 409


async def test_library_preview_uses_the_bare_voice_and_bills_the_admin(monkeypatch):
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[_profile(enabled=False)])))
    monkeypatch.setattr(tts, "resolve_agent_tts_model", AsyncMock(return_value="stepaudio-2.5-tts"))
    synthesize = AsyncMock(return_value=_speech())
    record = AsyncMock()
    monkeypatch.setattr(tts, "synthesize_speech", synthesize)
    monkeypatch.setattr(tts, "record_tts_usage", record)

    response = await tts.preview_voice_profile("p1", tts.VoicePreviewPayload(), admin={"sub": "admin-1"})

    assert response.body == b"RIFFwav"
    assert response.headers["X-TTS-Cost-CNY"] == "0.005800"
    assert synthesize.await_args.kwargs == {
        "text": tts.VOICE_SAMPLE_TEXT, "voice_id": "linjiajiejie", "instruction": None,
        "rate": 0.9, "volume": 50, "model": "stepaudio-2.5-tts",
    }
    assert record.await_args.kwargs["user_id"] == "admin-1"
    assert record.await_args.kwargs["agent_id"] is None
    assert record.await_args.kwargs["source"] == "admin_voice_preview"

    await tts.preview_voice_profile("p1", tts.VoicePreviewPayload(text=" 试试 "), admin={})
    assert synthesize.await_args.kwargs["text"] == "试试"


async def test_library_preview_reports_provider_failure(monkeypatch):
    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[_profile()])))
    monkeypatch.setattr(tts, "resolve_agent_tts_model", AsyncMock(side_effect=SpeechSynthesisError("bad model")))
    with pytest.raises(HTTPException) as exc:
        await tts.preview_voice_profile("p1", tts.VoicePreviewPayload(), admin={})
    assert exc.value.status_code == 502


def _agent_row(probability):
    return {
        "id": "agent-1", "name": "小伴", "gender": "female", "user_id": "u1",
        "tts_voice_id": "linjiajiejie", "tts_rate": 0.9, "tts_volume": 50,
        "tts_instruction": None, "tts_auto_emotion": True, "tts_emotion_scale": 1.0,
        "tts_output_probability": probability,
    }


@pytest.mark.parametrize(("body", "keep", "value", "stored"), [
    ({}, True, None, 30),
    ({"output_probability": None}, False, None, None),
    ({"output_probability": 75}, False, 75, 75),
])
async def test_agent_probability_is_saved_only_when_sent(monkeypatch, body, keep, value, stored):
    database = SimpleNamespace(
        query_raw=AsyncMock(side_effect=[[_profile()], [_agent_row(stored)], [{"id": "p1"}]]),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(tts, "db", database)
    monkeypatch.setattr(tts, "effective_tts_probability", AsyncMock(return_value=20))

    result = await tts.update_agent_tts_config("agent-1", tts.AgentTtsPayload(
        voice_profile_id="p1", rate=0.9, volume=50, emotion_scale=1.0, **body,
    ))

    assert "tts_output_probability = CASE WHEN $8" in database.execute_raw.await_args.args[0]
    assert database.execute_raw.await_args.args[8:] == (keep, value)
    assert result["output_probability"] == stored
    assert result["global_output_probability"] == 20


def test_agent_probability_is_bounded():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        tts.AgentTtsPayload(voice_profile_id="p1", rate=1, volume=50, emotion_scale=1, output_probability=101)
