"""StepFun voice cloning: upload the reference WAV, then create the voice."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.services.speech_output import voice_enrollment
from app.services.speech_output.client import SpeechSynthesisError


FILES = "https://stepfun.test/v1/files"
VOICES = "https://stepfun.test/v1/audio/voices"


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(voice_enrollment.settings, "stepfun_api_key", "sk-test")
    monkeypatch.setattr(voice_enrollment.settings, "stepfun_files_endpoint", FILES)
    monkeypatch.setattr(voice_enrollment.settings, "stepfun_voices_endpoint", VOICES)


def stepfun(*, upload=None, clone=None, calls=None):
    upload = upload or (lambda request: httpx.Response(200, json={"id": "file-1"}))
    clone = clone or (lambda request: httpx.Response(
        200, json={"id": "voice-tone-1"}, headers={"x-trace-id": "trace-1"},
    ))

    def handle(request):
        if calls is not None:
            calls.append(request)
        return upload(request) if str(request.url) == FILES else clone(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def test_clone_uploads_the_sample_then_creates_the_voice():
    calls: list[httpx.Request] = []
    async with stepfun(calls=calls) as client:
        result = await voice_enrollment.create_cloned_voice(audio=b"RIFF-wav", client=client)

    assert result == voice_enrollment.EnrollmentResult("voice-tone-1", "trace-1")
    upload, clone = calls
    assert upload.headers["Authorization"] == "Bearer sk-test"
    assert b'name="purpose"\r\n\r\nstorage' in upload.content
    assert b"RIFF-wav" in upload.content
    assert json.loads(clone.content) == {"model": "stepaudio-2.5-tts", "file_id": "file-1"}


async def test_clone_without_trace_header_has_no_request_id():
    async with stepfun(clone=lambda r: httpx.Response(200, json={"id": "voice-tone-2"})) as client:
        result = await voice_enrollment.create_cloned_voice(audio=b"wav", client=client)
    assert result.request_id is None


async def test_clone_requires_a_key():
    voice_enrollment.settings.stepfun_api_key = " "
    with pytest.raises(SpeechSynthesisError, match="not configured"):
        await voice_enrollment.create_cloned_voice(audio=b"wav")


@pytest.mark.parametrize(("upload", "clone", "match"), [
    (lambda r: httpx.Response(413, json={"error": {"type": "file_too_large", "message": "big"}}),
     None, "file upload file_too_large: big"),
    (lambda r: httpx.Response(500, content=b"<html>"), None, "file upload http_500: request failed"),
    (lambda r: httpx.Response(200, json={"object": "file"}), None, "did not return a file id"),
    (lambda r: httpx.Response(200, content=b"not json"), None, "file upload returned invalid JSON"),
    (lambda r: httpx.Response(200, json=["file-1"]), None, "file upload returned invalid payload"),
    (None, lambda r: httpx.Response(400, json={"error": {"type": "file_id_invalid"}}),
     "voice enrollment file_id_invalid: request failed"),
    (None, lambda r: httpx.Response(200, json={"object": "audio.voice"}), "did not return a voice id"),
])
async def test_provider_failures_are_reported(upload, clone, match):
    async with stepfun(upload=upload, clone=clone) as client:
        with pytest.raises(SpeechSynthesisError, match=match):
            await voice_enrollment.create_cloned_voice(audio=b"wav", client=client)


async def test_network_failure_is_reported_and_owned_client_closed(monkeypatch):
    clients: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def handle(request):
        raise httpx.ConnectError("offline")

    def create(**kwargs):
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
