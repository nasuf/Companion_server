"""Bound real search before billing; never call an unpriced/uncapped route."""

from copy import deepcopy
from decimal import Decimal
from unittest.mock import AsyncMock
import json

import httpx
import pytest

from app.services.quality import search, transport, budget

PRICE = {search.MODEL: {"input": 0.8, "output": 2, "cached_input": 0.16}}
MESSAGES = [{"role": "user", "content": "请查杭州今天的天气"}]


def response():
    return dict(
        status="completed",
        usage=dict(
            input_tokens=25000,
            output_tokens=120,
            input_tokens_details={"cached_tokens": 1000},
            tool_usage={"web_search": 1},
            tool_usage_details={"web_search": {"search_engine": 1}},
        ),
        output=[
            dict(type="web_search_call", action={"query": "synthetic"}),
            dict(type="message", content=[dict(type="output_text", text="合成回复")]),
        ],
    )


def test_search_ceiling_reserves_upper_tier_but_keeps_32k_gate():
    bound, amount = transport.price_bound(
        "ark",
        search.MODEL.split("/")[1],
        MESSAGES,
        900,
        prices=PRICE,
        operation="web_search",
    )
    assert bound == 262144 and amount == Decimal(".32937280")
    for model in ("ark/unknown", "dashscope/qwen3.5-flash"):
        with pytest.raises(ValueError):
            search.ceiling(model, 900)
    for location in ({"api_key": "not allowed"}, {"city": []}, "杭州"):
        with pytest.raises(ValueError):
            search.body(search.MODEL.split("/")[1], MESSAGES, 900, location)
    value = response()
    value["usage"]["input_tokens"] = 32768
    assert search.usage(value, 900)["input"] == 32768
    value["usage"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="32k"):
        search.usage(value, 900)


@pytest.mark.parametrize(
    "mode",
    [
        "ok",
        "missing_usage",
        "wrong_source",
        "fanout",
        "over_tier",
        "truncated",
        "http",
        "cancel",
    ],
)
async def test_search_billing_contract(monkeypatch, mode):
    import asyncio

    reserved, settled, uncertain = (
        AsyncMock(return_value="charge"),
        AsyncMock(),
        AsyncMock(),
    )
    monkeypatch.setattr(budget, "reserve", reserved)
    monkeypatch.setattr(budget, "settle", settled)
    monkeypatch.setattr(budget, "uncertain", uncertain)
    value = response()
    if mode == "missing_usage":
        value.pop("usage")
    if mode == "wrong_source":
        value["usage"]["tool_usage_details"]["web_search"] = {"doubao": 1}
    if mode == "fanout":
        value["usage"]["tool_usage"]["web_search"] = 2
    if mode == "over_tier":
        value["usage"]["input_tokens"] = 32769
    if mode == "truncated":
        value["status"] = "incomplete"

    def answer(request):
        assert reserved.await_count == 1
        body = json.loads(request.content)
        assert request.url.path == "/api/v3/responses"
        assert body["max_tool_calls"] == body["tools"][0]["max_keyword"] == 1
        assert body["max_output_tokens"] == 900 and body["store"] is False
        assert body["tools"][0]["sources"] == ["search_engine"]
        assert body["thinking"]["type"] == "disabled"
        if mode == "cancel":
            raise asyncio.CancelledError()
        return httpx.Response(503 if mode == "http" else 200, json=value)

    client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda **_: client)
    run = transport.complete(
        dict(id="job", lease_token="lease", snapshot={"prices": PRICE}),
        "ark",
        search.MODEL.split("/")[1],
        MESSAGES,
        operation="web_search",
        key_override="synthetic",
    )
    if mode == "ok":
        text, used = await run
        assert text == "合成回复" and used["route"] == "responses.web_search"
        assert used["cost_cny"] == pytest.approx(0.0236)
        uncertain.assert_not_awaited()
    else:
        with pytest.raises((ValueError, httpx.HTTPStatusError, asyncio.CancelledError)):
            await run
        uncertain.assert_awaited_once_with("charge")
    assert settled.await_count == (1 if mode in {"ok", "truncated"} else 0)


def test_search_metering_does_not_accept_bool_negative_or_missing_counts():
    for field, value in [
        ("input_tokens", True),
        ("output_tokens", -1),
        ("output_tokens", 1801),
    ]:
        raw = response()
        raw["usage"][field] = value
        with pytest.raises(ValueError):
            search.usage(raw, 900)
    raw = response()
    raw["usage"]["input_tokens_details"]["cached_tokens"] = 25001
    with pytest.raises(ValueError):
        search.usage(raw, 900)
