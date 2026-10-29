"""Budgeted provider transport, independent of user chat/fallback side effects."""

import asyncio
import math
import time
from decimal import Decimal, ROUND_CEILING

import httpx

from app.config import settings
from app.services.llm.providers import get_provider, _model_runtime_options
from app.services.runtime_config import get_pricing
from app.services.quality import budget, search


class NotConfigured(RuntimeError):
    pass


def pinned_price(model, prices=None):
    price = dict(
        (prices.get(model) if prices is not None else get_pricing(model)) or {}
    )
    if not price or any(
        type(price.get(k)) not in (int, float)
        or not math.isfinite(price[k])
        or price[k] <= 0
        for k in ("input", "output")
    ):
        raise NotConfigured("evaluation_model_not_priced")
    if price.get("cached_input") is None:
        price["cached_input"] = price["input"]
    if "cached_input" in price and (
        type(price["cached_input"]) not in (int, float)
        or not math.isfinite(price["cached_input"])
        or not 0 <= price["cached_input"] <= price["input"]
    ):
        raise NotConfigured("evaluation_cache_price_invalid")
    return price


def price_cost(price, inputs, outputs, cached=0):
    return float(
        (
            Decimal(inputs - cached) * Decimal(str(price["input"]))
            + Decimal(cached) * Decimal(str(price.get("cached_input", price["input"])))
            + Decimal(outputs) * Decimal(str(price["output"]))
        )
        / Decimal(1_000_000)
    )


def price_bound(
    provider, model, messages, max_tokens, *, prices=None, operation="chat"
):
    if (
        provider not in {"ark", "dashscope"}
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= 2048
    ):
        raise ValueError("evaluation_route_not_allowed")
    if (
        not messages
        or len(messages) > 80
        or any(
            m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str)
            for m in messages
        )
    ):
        raise ValueError("invalid_eval_messages")
    # UTF-8 bytes conservatively dominate text tokens; overhead is per-message.
    bound = sum(len(m["content"].encode()) + 64 for m in messages) + 128
    if bound > 32768:
        raise ValueError("price_unknown_above_32k_bound")
    key = provider + "/" + model
    price = pinned_price(key, prices)
    if operation == "web_search":
        if price["input"] > 1.2 or price["output"] > 6:
            raise NotConfigured("search_price_above_qualified_ceiling")
        return search.MAX_INPUT, search.ceiling(key, max_tokens)
    if operation != "chat":
        raise ValueError("evaluation_operation_not_allowed")
    ceiling = Decimal(str(price_cost(price, bound, max_tokens))).quantize(
        Decimal(".00000001"), rounding=ROUND_CEILING
    )
    return bound, ceiling


async def complete(
    job,
    provider,
    model,
    messages,
    *,
    max_tokens=900,
    json_mode=False,
    key_override=None,
    operation="chat",
    location=None,
):
    price = pinned_price(provider + "/" + model, job.get("snapshot", {}).get("prices"))
    bound, ceiling = price_bound(
        provider,
        model,
        messages,
        max_tokens,
        prices={provider + "/" + model: price},
        operation=operation,
    )
    spec = get_provider(provider)
    key = key_override or getattr(settings, spec.api_key_setting)
    if not key:
        raise NotConfigured("evaluation_provider_key_missing")
    search_body = (
        search.body(model, messages, max_tokens, location)
        if operation == "web_search"
        else None
    )
    charge_id = await budget.reserve(
        job["id"],
        provider + "/" + model,
        ceiling,
        lease_token=job["lease_token"],
        request=dict(messages=messages, max_tokens=max_tokens, operation=operation),
    )
    try:
        body = {
            **_model_runtime_options(provider, model),
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if provider == "dashscope":
            body.update(enable_thinking=False)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if search_body:
            body = search_body
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=75, trust_env=False) as client:
            response = await client.post(
                getattr(settings, spec.base_url_setting).rstrip("/")
                + ("/responses" if search_body else "/chat/completions"),
                headers={"Authorization": "Bearer " + key},
                json=body,
            )
            response.raise_for_status()
            result = response.json()
        if search_body:
            from app.services.llm.ark_web_search import _extract_output

            used = search.usage(result, max_tokens)
            cost = price_cost(
                price, used["input"], used["output"], used["cached_input"]
            ) + float(search.tool_cost(used))
            await budget.settle(charge_id, cost, used)
            text = _extract_output(result).text
            if result.get("status") != "completed" or not text:
                raise ValueError("evaluation_search_incomplete")
            return text, {
                **used,
                "cost_cny": cost,
                "response_ms": round((time.monotonic() - started) * 1000),
                "route": "responses.web_search",
            }
        usage = result.get("usage") or {}
        if (
            type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
        ):
            raise ValueError("evaluation_usage_missing")
        inputs, outputs = usage["prompt_tokens"], usage["completion_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if (
            any(type(n) is not int for n in (inputs, outputs, cached))
            or not 0 <= cached <= inputs <= bound
            or not 0 <= outputs <= max_tokens
        ):
            raise ValueError("evaluation_usage_out_of_bounds")
        cost = price_cost(price, inputs, outputs, cached)
        await budget.settle(
            charge_id, cost, dict(input=inputs, output=outputs, cached_input=cached)
        )
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop" or not choice["message"].get(
            "content"
        ):
            raise ValueError("evaluation_response_incomplete")
        return choice["message"]["content"], dict(
            input=inputs,
            output=outputs,
            cached_input=cached,
            cost_cny=cost,
            route="chat.completions",
            response_ms=round((time.monotonic() - started) * 1000),
        )
    except BaseException:
        # If the caller is cancelled or metering failed, conservatively retain
        # the full reservation. Never retry the same billed call automatically.
        await asyncio.shield(budget.uncertain(charge_id))
        raise
