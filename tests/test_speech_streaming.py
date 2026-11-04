import asyncio
import json
import struct

import httpx
import pytest
import pytest_asyncio

from app.services.speech_output import client as tts
from tests.test_speech_output import _wav_bytes


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
    monkeypatch.setattr(tts.settings, "stepfun_api_key", "test-key")
    monkeypatch.setattr(tts.settings, "stepfun_tts_endpoint", "https://tts.example/v1/audio/speech")
    monkeypatch.setattr(tts.settings, "stepfun_tts_max_bytes", 1024 * 1024)
    monkeypatch.setattr(tts, "get_tts_pricing", lambda model: {"unit_price_cny": 5.8})
    yield
    await tts.close_tts_client()


async def synthesize(client=None, text="你好"):
    return await tts.synthesize_speech(text=text, voice_id="voice", instruction="自然", model="test", client=client)


def audio_response(audio, *, stream=None):
    return httpx.Response(
        200,
        headers={"content-type": "audio/wav", "x-trace-id": "trace-1"},
        stream=stream or ByteStream(audio),
    )


@pytest.mark.asyncio
async def test_chunked_wav_is_joined_and_metered_locally():
    audio = _wav_bytes(seconds=0.5)
    stream = ByteStream(audio)
    calls = []

    async def handle(request):
        calls.append(request.method)
        return audio_response(audio, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        speech = await synthesize(client, text="你好 A1。")
    assert calls == ["POST"]
    assert speech.audio == audio
    assert speech.duration_milliseconds == 500
    assert speech.request_id == "trace-1"
    # Two Han characters count once each; " A1。" is four other characters.
    assert speech.billable_characters == 4
    assert speech.cost_cny == pytest.approx(4 * 5.8 / 10_000)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body,match", [
    (400, {"error": {"type": "voice_id_invalid", "message": "no such voice"}}, "voice_id_invalid: no such voice"),
    (429, {"error": {"type": "rate_limited"}}, "rate_limited: request failed"),
    (500, b"<html>oops</html>", "http_500: request failed"),
])
async def test_provider_errors_are_reported_and_never_retried(status, body, match):
    calls = []

    async def handle(request):
        calls.append(request.method)
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match=match):
            await synthesize(client)
    assert calls == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("audio,match", [
    (b"", "did not include audio"),
    (b"not a wav", "invalid WAV"),
])
async def test_bad_audio_is_rejected(audio, match):
    async def handle(request):
        return audio_response(audio)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match=match):
            await synthesize(client)


@pytest.mark.asyncio
async def test_audio_limit_applies_before_full_download(monkeypatch):
    monkeypatch.setattr(tts.settings, "stepfun_tts_max_bytes", 64)
    stream = ByteStream(_wav_bytes())

    async def handle(request):
        return audio_response(b"", stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match="allowed size"):
            await synthesize(client)
    assert stream.closed


@pytest.mark.asyncio
async def test_input_limit_and_missing_key_fail_before_any_request(monkeypatch):
    calls = []

    async def handle(request):
        calls.append(request.method)
        return audio_response(_wav_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(tts.SpeechSynthesisError, match="input limit"):
            await synthesize(client, text="好" * 1000)
        with pytest.raises(tts.SpeechSynthesisError, match="empty"):
            await synthesize(client, text="   ")
        monkeypatch.setattr(tts.settings, "stepfun_api_key", " ")
        with pytest.raises(tts.SpeechSynthesisError, match="not configured"):
            await synthesize(client)
    assert calls == []


@pytest.mark.asyncio
async def test_zero_price_and_settings_fallback(monkeypatch):
    async def handle(request):
        return audio_response(_wav_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(tts, "get_tts_pricing", lambda model: {"unit_price_cny": 0})
        assert (await synthesize(client)).cost_cny == 0
        monkeypatch.setattr(tts, "get_tts_pricing", lambda model: None)
        monkeypatch.setattr(tts.settings, "tts_price_cny_per_10k_chars", 5.8)
        assert (await synthesize(client)).unit_price_cny == 5.8


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
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=audio)

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
    for _ in range(tts.settings.stepfun_tts_max_concurrency):
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
        return audio_response(b"", stream=stream)

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
async def test_list_chunk_and_streaming_placeholder_are_finalized():
    """StepFun prepends a LIST/INFO chunk; a 0xFFFFFFFF data length is fixed."""
    ordinary = _wav_bytes(seconds=0.5)
    info = b"LIST" + struct.pack("<I", 26) + b"INFOISFT" + struct.pack("<I", 14) + b"Lavf58.76.100\x00"
    audio = bytearray(ordinary[:36] + info + ordinary[36:])
    data_offset = 36 + len(info)
    struct.pack_into("<I", audio, 4, 0xFFFFFFFF)
    struct.pack_into("<I", audio, data_offset + 4, 0xFFFFFFFF)

    async def handle(request):
        return audio_response(bytes(audio))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        speech = await synthesize(client)
    assert speech.duration_milliseconds == 500
    assert struct.unpack_from("<I", speech.audio, 4)[0] == len(audio) - 8
    assert struct.unpack_from("<I", speech.audio, data_offset + 4)[0] == 24000
    assert speech.audio[36:data_offset] == info
    assert speech.audio[data_offset + 8:] == ordinary[44:]


def test_incomplete_and_empty_wav_frames_are_rejected():
    import io
    import wave

    def wav(frames: bytes, channels: int = 2) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes(frames)
        return buffer.getvalue()

    with pytest.raises(tts.SpeechSynthesisError, match="incomplete WAV"):
        tts.wav_duration_milliseconds(wav(b"\x00\x00"))
    with pytest.raises(tts.SpeechSynthesisError, match="empty WAV"):
        tts.wav_duration_milliseconds(wav(b"", channels=1))
