"""Voice messages: one realtime session reads the reply aloud."""
import asyncio
import base64
import json

import pytest

from app.services.llm import pricing
from app.services.realtime_voice import provider
from app.services.speech_output import client as tts

PCM = b"\x10\x00" * 2400  # 0.1 s


class FakeSession:
    """Scripted DashScope realtime socket: replays ``events`` after response.create."""

    def __init__(self, events, *, hang=False):
        self.events = list(events)
        self.sent: list[dict] = []
        self.hang = hang
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if self.hang:
            await asyncio.sleep(3600)
        if not self.events:
            await asyncio.sleep(3600)
        return json.dumps(self.events.pop(0))


def read_events(spoken, *, pcm=PCM, usage=None, response_id="resp_1"):
    return [
        {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()},
        {"type": "response.audio_transcript.done", "transcript": spoken},
        {"type": "response.done", "response": {"id": response_id, "status": "completed", "usage": usage or {
            "input_tokens": 120, "output_tokens": 40,
            "input_tokens_details": {"text_tokens": 120},
            "output_tokens_details": {"text_tokens": 10, "audio_tokens": 30},
        }}},
    ]


@pytest.fixture
def sessions(monkeypatch):
    opened: list[FakeSession] = []
    script: list[FakeSession] = []

    class Connect:
        def __init__(self, url, key):
            self.url, self.key = url, key

        async def __aenter__(self):
            session = script.pop(0)
            session.url, session.key = self.url, self.key
            opened.append(session)
            return session

        async def __aexit__(self, *exc):
            opened[-1].closed = True

    monkeypatch.setattr(provider, "connect_realtime", Connect)
    monkeypatch.setattr(tts.settings, "dashscope_api_key", "sk-test")
    monkeypatch.setattr(tts.settings, "dashscope_workspace_id", "ws-test")
    monkeypatch.setattr(tts.settings, "tts_max_bytes", 1024 * 1024)
    monkeypatch.setattr(tts.settings, "tts_timeout_s", 5.0)

    async def model():
        return "qwen3.8-omni-flash-realtime"

    monkeypatch.setattr(tts, "get_effective_realtime_model", model)
    prices = {
        "qwen3.8-omni-flash-realtime": (1.5, 4.5),
        "qwen3.8-omni-flash-realtime:audio": (6.0, 12.0),
    }
    monkeypatch.setattr(pricing, "estimate_cost_cny", lambda m, i, o, cached_input_tokens=0: (
        (i * prices[m][0] + o * prices[m][1]) / 1_000_000
    ))
    tts._TRANSPORTS.clear()
    return script, opened


async def test_reads_the_text_in_the_agent_voice_and_meters_per_token(sessions):
    script, opened = sessions
    script.append(FakeSession(read_events("刚下班到家，累瘫了。")))
    result = await tts.synthesize_speech(
        text="刚下班到家，累瘫了。", voice_id="Serena", instruction="像微信语音。", rate=0.9, volume=50,
    )
    (session,) = opened
    assert session.url == "wss://ws-test.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-omni-flash-realtime"
    assert session.key == "sk-test" and session.closed
    update, item, create = session.sent
    assert update["session"]["voice"] == "Serena"
    assert update["session"]["turn_detection"] is None
    assert update["session"]["instructions"].endswith("朗读要求：像微信语音。 语速稍慢。")
    assert "「刚下班到家，累瘫了。」" in item["item"]["content"][0]["text"]
    assert create["type"] == "response.create"
    assert result.request_id == "resp_1" and result.model == "qwen3.8-omni-flash-realtime"
    assert result.duration_milliseconds == 100 and result.mime == "audio/wav"
    assert result.audio[:4] == b"RIFF"
    expected = (120 * 1.5 + 10 * 4.5 + 30 * 12.0) / 1_000_000
    assert result.cost_cny == pytest.approx(expected)
    assert result.unit_price_cny == pytest.approx(round(expected * 10_000 / result.billable_characters, 6))


async def test_a_wrong_reading_is_retried_once_and_both_are_billed(sessions):
    script, opened = sessions
    script.append(FakeSession(read_events("我挺好的，就是有点想你了。")))
    script.append(FakeSession(read_events("你今天怎么样？", response_id="resp_2")))
    result = await tts.synthesize_speech(text="你今天怎么样？", voice_id="Serena", instruction=None)
    assert len(opened) == 2 and result.request_id == "resp_2"
    one = (120 * 1.5 + 10 * 4.5 + 30 * 12.0) / 1_000_000
    assert result.cost_cny == pytest.approx(2 * one)
    assert "朗读要求" not in opened[0].sent[0]["session"]["instructions"]


async def test_two_wrong_readings_fail(sessions):
    script, _ = sessions
    script.extend([FakeSession(read_events("答非所问")), FakeSession(read_events("还是答非所问"))])
    with pytest.raises(tts.SpeechSynthesisError, match="verbatim"):
        await tts.synthesize_speech(text="你今天怎么样？", voice_id="Serena", instruction=None)


@pytest.mark.parametrize(("events", "match"), [
    ([{"type": "error", "error": {"message": "The voice X could not be found."}}], "could not be found"),
    ([{"type": "response.audio.delta", "delta": "!!not base64!!"}], "invalid audio"),
])
async def test_provider_errors_are_reported(sessions, events, match):
    script, _ = sessions
    script.append(FakeSession(events))
    with pytest.raises(tts.SpeechSynthesisError, match=match):
        await tts.synthesize_speech(text="你好", voice_id="X", instruction=None)


async def test_audio_limit_and_timeout(sessions, monkeypatch):
    script, _ = sessions
    monkeypatch.setattr(tts.settings, "tts_max_bytes", 64)
    script.append(FakeSession(read_events("你好")))
    with pytest.raises(tts.SpeechSynthesisError, match="exceeded"):
        await tts.synthesize_speech(text="你好", voice_id="Serena", instruction=None)
    monkeypatch.setattr(tts.settings, "tts_timeout_s", 0.05)
    script.append(FakeSession([], hang=True))
    with pytest.raises(tts.SpeechSynthesisError, match="timed out"):
        await tts.synthesize_speech(text="你好", voice_id="Serena", instruction=None)


async def test_unexpected_failures_are_wrapped_and_release_the_permit(sessions, monkeypatch):
    script, _ = sessions

    class Broken(FakeSession):
        async def recv(self):
            raise OSError("socket reset")

    script.append(Broken([]))
    with pytest.raises(tts.SpeechSynthesisError, match="OSError"):
        await tts.synthesize_speech(text="你好", voice_id="Serena", instruction=None)
    assert tts._transport().semaphore._value == tts.settings.tts_max_concurrency


async def test_input_limits_and_missing_key_fail_before_any_session(sessions, monkeypatch):
    _, opened = sessions
    with pytest.raises(tts.SpeechSynthesisError, match="empty"):
        await tts.synthesize_speech(text="  ", voice_id="Serena", instruction=None)
    with pytest.raises(tts.SpeechSynthesisError, match="input limit"):
        await tts.synthesize_speech(text="字" * 1000, voice_id="Serena", instruction=None)
    monkeypatch.setattr(tts.settings, "dashscope_api_key", " ")
    with pytest.raises(tts.SpeechSynthesisError, match="not configured"):
        await tts.synthesize_speech(text="你好", voice_id="Serena", instruction=None)
    assert opened == []


async def test_cancellation_releases_the_permit(sessions):
    script, _ = sessions
    script.append(FakeSession([], hang=True))
    task = asyncio.create_task(tts.synthesize_speech(text="你好", voice_id="Serena", instruction=None))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tts._transport().semaphore._value == tts.settings.tts_max_concurrency
