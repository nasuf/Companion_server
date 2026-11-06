"""DashScope voice cloning: one clone for the realtime model, deletable."""
from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from app.services.speech_output import voice_enrollment
from app.services.speech_output.client import SpeechSynthesisError


CUSTOMIZATION = "https://ws-test.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/customization"


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(voice_enrollment.settings, "dashscope_api_key", "sk-test")
    monkeypatch.setattr(voice_enrollment.settings, "dashscope_workspace_id", "ws-test")


def dashscope(respond, calls=None):
    def handle(request):
        if calls is not None:
            calls.append(request)
        return respond(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def created(voice="qwen-omni-vc-companion-voice-1", request_id="req-1"):
    body = {"output": {"voice": voice, "target_model": "qwen3.8-omni-flash-realtime"}, "usage": {"count": 1}}
    if request_id:
        body["request_id"] = request_id
    return lambda request: httpx.Response(200, json=body)


async def test_clone_targets_the_realtime_model_with_the_sample_inline():
    calls: list[httpx.Request] = []
    async with dashscope(created(), calls) as client:
        result = await voice_enrollment.create_cloned_voice(audio=b"RIFF-wav", client=client)

    assert result == voice_enrollment.EnrollmentResult("qwen-omni-vc-companion-voice-1", "req-1")
    (request,) = calls
    assert str(request.url) == CUSTOMIZATION
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "qwen-voice-enrollment"
    assert body["parameters"] == {"voice_clone_mode": "normal"}
    assert body["input"]["action"] == "create"
    assert body["input"]["target_model"] == "qwen3.8-omni-flash-realtime"
    assert body["input"]["preferred_name"] == "companion"
    assert body["input"]["audio"]["data"] == "data:audio/wav;base64," + base64.b64encode(b"RIFF-wav").decode()


async def test_shared_endpoint_without_a_workspace(monkeypatch):
    monkeypatch.setattr(voice_enrollment.settings, "dashscope_workspace_id", " ")
    calls: list[httpx.Request] = []
    async with dashscope(created(request_id=None), calls) as client:
        result = await voice_enrollment.create_cloned_voice(audio=b"wav", client=client)
    assert result.request_id is None
    assert str(calls[0].url) == "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"


async def test_clone_requires_a_key():
    voice_enrollment.settings.dashscope_api_key = " "
    with pytest.raises(SpeechSynthesisError, match="not configured"):
        await voice_enrollment.create_cloned_voice(audio=b"wav")


@pytest.mark.parametrize(("respond", "match"), [
    (lambda r: httpx.Response(400, json={"code": "InvalidParameter", "message": "bad audio"}),
     "voice enrollment InvalidParameter: bad audio"),
    (lambda r: httpx.Response(500, content=b"<html>"), "voice enrollment http_500: request failed"),
    (lambda r: httpx.Response(200, json={"output": {}}), "did not return a voice"),
    (lambda r: httpx.Response(200, content=b"not json"), "returned invalid JSON"),
    (lambda r: httpx.Response(200, json=["voice"]), "returned invalid payload"),
])
async def test_provider_failures_are_reported(respond, match):
    async with dashscope(respond) as client:
        with pytest.raises(SpeechSynthesisError, match=match):
            await voice_enrollment.create_cloned_voice(audio=b"wav", client=client)


async def test_delete_removes_the_voice_and_tolerates_one_already_gone():
    calls: list[httpx.Request] = []
    async with dashscope(lambda r: httpx.Response(200, json={"request_id": "r"}), calls) as client:
        await voice_enrollment.delete_cloned_voice("qwen-omni-vc-x", client=client)
    assert json.loads(calls[0].content) == {
        "model": "qwen-voice-enrollment", "input": {"action": "delete", "voice": "qwen-omni-vc-x"},
    }
    gone = lambda r: httpx.Response(400, json={"code": "BadRequest.VoiceNotFound", "message": "Voice not found."})
    async with dashscope(gone) as client:
        await voice_enrollment.delete_cloned_voice("qwen-omni-vc-x", client=client)


@pytest.mark.parametrize("respond", [
    lambda r: httpx.Response(400, json={"code": "InvalidParameter", "message": "bad"}),
    lambda r: httpx.Response(400, content=b"<html>"),
    lambda r: httpx.Response(503, json={"code": "Throttling"}),
])
async def test_delete_failures_are_reported(respond):
    async with dashscope(respond) as client:
        with pytest.raises(SpeechSynthesisError, match="voice deletion"):
            await voice_enrollment.delete_cloned_voice("qwen-omni-vc-x", client=client)


async def test_network_failure_is_reported_and_owned_client_closed(monkeypatch):
    clients: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def handle(request):
        raise httpx.ConnectError("offline")

    def create(**kwargs):
        kwargs.pop("trust_env", None)
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(voice_enrollment.httpx, "AsyncClient", create)
    with pytest.raises(SpeechSynthesisError, match="network request failed"):
        await voice_enrollment.create_cloned_voice(audio=b"wav")
    assert len(clients) == 1 and clients[0].is_closed


class FakeProcess:
    def __init__(self, *, stdout=b"pcm", returncode=0, hang=False):
        self.stdout = stdout
        self.returncode = None if hang else returncode
        self.hang = hang
        self.killed = False

    async def communicate(self, data=None):
        if self.hang and not self.killed:
            await asyncio.sleep(3600)
        return self.stdout, b""

    def kill(self):
        self.killed = True


async def test_decode_returns_ffmpeg_stdout(monkeypatch):
    async def spawn(*args, **kwargs):
        assert args[0] == "ffmpeg" and "s16le" in args
        return FakeProcess(stdout=b"\x01\x00")

    monkeypatch.setattr(voice_enrollment.asyncio, "create_subprocess_exec", spawn)
    assert await voice_enrollment._decode_enrollment_pcm(b"m4a") == b"\x01\x00"


@pytest.mark.parametrize("process", [
    FakeProcess(returncode=1), FakeProcess(stdout=b""),
])
async def test_decode_rejects_ffmpeg_failure(monkeypatch, process):
    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(voice_enrollment.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(ValueError, match="could not be decoded"):
        await voice_enrollment._decode_enrollment_pcm(b"m4a")


async def test_decode_without_ffmpeg(monkeypatch):
    async def spawn(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(voice_enrollment.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(ValueError, match="requires ffmpeg"):
        await voice_enrollment._decode_enrollment_pcm(b"m4a")


async def test_decode_timeout_kills_ffmpeg(monkeypatch):
    process = FakeProcess(hang=True)

    async def spawn(*args, **kwargs):
        return process

    async def expire(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(voice_enrollment.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(voice_enrollment.asyncio, "wait_for", expire)
    with pytest.raises(ValueError, match="timed out"):
        await voice_enrollment._decode_enrollment_pcm(b"m4a")
    assert process.killed


async def test_decode_timeout_tolerates_an_already_exited_process(monkeypatch):
    process = FakeProcess(hang=True)

    def gone():
        raise ProcessLookupError

    process.kill = gone
    process.hang = False

    async def spawn(*args, **kwargs):
        return process

    async def expire(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(voice_enrollment.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(voice_enrollment.asyncio, "wait_for", expire)
    with pytest.raises(ValueError, match="timed out"):
        await voice_enrollment._decode_enrollment_pcm(b"m4a")


async def test_prepare_rejects_undecodable_audio(monkeypatch):
    async def decode(blob):
        return b""

    monkeypatch.setattr(voice_enrollment, "_decode_enrollment_pcm", decode)
    with pytest.raises(ValueError, match="could not be decoded"):
        await voice_enrollment.prepare_enrollment_audio(
            blob=b"m4a", mime="audio/mp4", filename="a.m4a",
        )


def test_provider_error_tolerates_unexpected_bodies():
    for response in (
        httpx.Response(502, json={"error": "bad gateway"}),
        httpx.Response(502, json=[1]),
    ):
        assert "http_502: request failed" in str(voice_enrollment._provider_error("x", response))
