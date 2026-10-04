import asyncio
import base64
import json
import struct

import httpx
import pytest
import pytest_asyncio

from app.services.speech_output import client as tts
from tests.test_speech_output import _wav_bytes


def event(output, **extra):
    return ("data: " + json.dumps({"output": output, **extra}) + "\r\n\r\n").encode()


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        for i in range(0, len(self.data), 113):
            yield self.data[i:i + 113]

    async def aclose(self):
        self.closed = True


@pytest_asyncio.fixture(autouse=True)
async def configured(monkeypatch):
    monkeypatch.setattr(tts.settings, "dashscope_tts_api_key", "test-key")
    monkeypatch.setattr(tts.settings, "dashscope_tts_endpoint", "https://tts.example/api")
    monkeypatch.setattr(tts.settings, "dashscope_tts_max_bytes", 1024 * 1024)
    monkeypatch.setattr(tts, "get_tts_pricing", lambda model: {"unit_price_cny": 0.8})
    yield
    await tts.close_tts_client()


async def synthesize(client=None):
    return await tts.synthesize_speech(text="你好", voice_id="voice", instruction="自然", model="test", client=client)


@pytest.mark.asyncio
async def test_streamed_frames_are_joined_without_extra_download():
    audio = _wav_bytes(seconds=0.5)
    data = b"".join(event({"type": "sentence-synthesis", "audio": {"data": base64.b64encode(part).decode()}}, usage={"characters": 7}, request_id="req") for part in (audio[:44], audio[44:8000], audio[8000:]))
    stream = ByteStream(data + event({"finish_reason": "stop", "audio": {"url": "https://oss.example/out.wav"}}))
    calls = []

    async def handle(request):
        calls.append(request.method)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        speech = await synthesize(client)
    assert calls == ["POST"]
    assert speech.audio == audio
    assert speech.duration_milliseconds == 500
    assert speech.request_id == "req"
    assert speech.billable_characters == 7
    assert speech.cost_cny == pytest.approx(0.00056)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("data,match", [
    (event({"audio": {"url": "https://oss.example/out.wav"}}), "before completion"),
    (b"data: {broken}\n\n", "invalid JSON"),
    (b'data: []\n\n', "invalid payload"),
    (b'data: {"code":"QuotaExceeded","message":"quota"}\n\n', "QuotaExceeded"),
    (event({"audio": {"data": "not base64!"}}), "invalid audio"),
    (event({"finish_reason": "stop", "audio": {"data": "dGVzdA=="}}), "invalid WAV"),
    (event({"finish_reason": "stop"}), "did not include audio"),
    (event({"finish_reason": "stop", "audio": {"url": "http://untrusted.example/out.wav"}}), "unsafe audio URL"),
])
async def test_failed_stream_is_closed_and_never_retried(data, match):
    stream = ByteStream(data)
    calls = []

    async def handle(request):
        calls.append(request.method)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match=match):
            await synthesize(client)
    assert stream.closed
    assert calls == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_audio", [True, False])
async def test_audio_limits_apply_before_full_download(monkeypatch, stream_audio):
    monkeypatch.setattr(tts.settings, "dashscope_tts_max_bytes", 64)
    audio = _wav_bytes()
    stream = ByteStream(event({"audio": {"data": base64.b64encode(audio).decode()}, "finish_reason": "stop"}) if stream_audio else audio)

    async def handle(request):
        if stream_audio or request.method == "GET":
            return httpx.Response(200, headers={"content-type": "text/event-stream" if stream_audio else "audio/wav"}, stream=stream)
        return httpx.Response(200, json={"output": {"finish_reason": "stop", "audio": {"url": "https://oss.example/out.wav"}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match="allowed size"):
            await synthesize(client)
    assert stream.closed


@pytest.mark.asyncio
async def test_header_usage_errors_and_zero_price(monkeypatch):
    audio = _wav_bytes()
    usage = {"characters": "bad"}

    async def handle(request):
        if request.method == "GET":
            return httpx.Response(200, content=audio)
        return httpx.Response(200, json={"output": {"audio": {"url": "https://oss.example/out.wav"}}, "usage": usage})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match="invalid usage"):
            await synthesize(client)
        usage["characters"] = 4
        monkeypatch.setattr(tts, "get_tts_pricing", lambda model: {"unit_price_cny": 0})
        assert (await synthesize(client)).cost_cny == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error,match", [(httpx.ReadTimeout("timeout"), "timed out"), (httpx.ConnectError("offline"), "network request failed")])
async def test_network_failure_releases_concurrency_permit(error, match):
    calls = []

    async def handle(request):
        calls.append(request.method)
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match=match):
            await synthesize(client)
    assert calls == ["POST"]
    assert not tts._tts_transport().semaphore.locked()


@pytest.mark.asyncio
async def test_shared_client_reuses_connections_and_closes(monkeypatch):
    audio = _wav_bytes()

    async def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=event({"finish_reason": "stop", "audio": {"data": base64.b64encode(audio).decode()}}))

    real_client = httpx.AsyncClient
    clients = []

    def create(**kwargs):
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(tts.httpx, "AsyncClient", create)
    await synthesize()
    await synthesize()
    assert len(clients) == 1
    assert not clients[0].is_closed
    await tts.close_tts_client()
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_cancellation_while_queued_does_not_create_client_or_leak_permit():
    transport = tts._tts_transport()
    for _ in range(tts.settings.dashscope_tts_max_concurrency):
        await transport.semaphore.acquire()
    task = asyncio.create_task(synthesize())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.client is None
    assert transport.semaphore.locked()
    transport.semaphore.release()
    async with asyncio.timeout(1):
        await transport.semaphore.acquire()


@pytest.mark.asyncio
async def test_cancellation_during_stream_closes_response_and_releases_permit():
    started = asyncio.Event()

    class WaitingStream(ByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

    stream = WaitingStream(b"")

    async def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        task = asyncio.create_task(synthesize(client))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        transport = tts._tts_transport()
        assert not transport.semaphore.locked()


def test_truncated_wav_is_not_delivered_as_complete():
    with pytest.raises(tts.SpeechSynthesisError, match="truncated WAV"):
        tts.wav_duration_milliseconds(_wav_bytes()[:-100])


@pytest.mark.asyncio
async def test_real_dashscope_length_placeholders_keep_aigc_metadata():
    ordinary = _wav_bytes(seconds=0.5)
    aigc = b"AIGC" + struct.pack("<I", 224) + b"x" * 224
    audio = bytearray(ordinary[:36] + aigc + ordinary[36:])
    data_offset = 36 + len(aigc)
    struct.pack_into("<I", audio, 4, 0x7FFFFFBF)
    struct.pack_into("<I", audio, data_offset + 4, 0x7FFFFFBF - data_offset)

    async def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=event({"finish_reason": "stop", "audio": {"data": base64.b64encode(audio).decode()}}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        speech = await synthesize(client)
    assert speech.duration_milliseconds == 500
    assert struct.unpack_from("<I", speech.audio, 4)[0] == len(audio) - 8
    assert struct.unpack_from("<I", speech.audio, data_offset + 4)[0] == 24000
    assert speech.audio[36:data_offset] == aigc
    assert speech.audio[data_offset + 8:] == ordinary[44:]
