import json

import httpx
import pytest

from evals.context_embedding.run_eval import Transport


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,done,passes",
    [("stop", True, True), ("length", True, False), ("stop", False, False)],
)
async def test_stream_completion_and_explicit_zero(monkeypatch, finish, done, passes):
    from app.config import settings

    monkeypatch.setattr(settings, "ark_api_key", "synthetic-test-key")
    monkeypatch.setattr(settings, "ark_base_url", "https://ark.invalid/api/v3")
    events = [
        {"choices": [{"delta": {"content": "你好"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 77,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    data = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
    if done:
        data += "data: [DONE]\n\n"

    def handle(request):
        body = json.loads(request.content)
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=data)

    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        send = Transport(client, "doubao-seed-character-260628", False, calls)
        if passes:
            result = await send([{"role": "user", "content": "你好"}])
            assert result["text"] == "你好"
            assert calls[0]["measurement"]["cached_input_tokens"] == 0
            assert calls[0]["ttft_seconds"] is not None
        else:
            with pytest.raises(RuntimeError):
                await send([{"role": "user", "content": "你好"}])
            assert calls[0]["price"]["cost_cny"] is None
            assert calls[0]["measurement"]["status"] == "request_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "incomplete"])
async def test_search_counts_and_nonstream_latency(monkeypatch, status):
    from app.config import settings

    monkeypatch.setattr(settings, "ark_api_key", "synthetic-test-key")
    monkeypatch.setattr(settings, "ark_base_url", "https://ark.invalid/api/v3")

    def handle(request):
        body = json.loads(request.content)
        assert (
            body["tools"] == [{"type": "web_search", "user_location": {"city": "镇江"}}]
            and body["stream"] is False
        )
        return httpx.Response(
            200,
            json={
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "官网"}],
                    }
                ],
                "usage": {
                    "input_tokens": 30000,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                    "tool_usage": {"web_search": 2},
                },
            },
        )

    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        send = Transport(client, "doubao-seed-character-260628", True, calls)
        if status == "completed":
            await send(
                [{"role": "user", "content": "官网"}],
                needs_web_search=True,
                web_search_location={"city": "镇江"},
            )
            assert calls[0]["price"]["search_cost_cny"] == 0.008
            assert calls[0]["ttft_seconds"] is None
            assert calls[0]["time_to_complete_seconds"] >= 0
        else:
            with pytest.raises(RuntimeError):
                await send(
                    [{"role": "user", "content": "官网"}],
                    needs_web_search=True,
                    web_search_location={"city": "镇江"},
                )
            assert calls[0]["price"]["cost_cny"] is None
