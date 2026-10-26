"""Strict route-aware measurements; missing evidence is never priced as free.

Ark rule qualified against 2026-10-26 raw responses: an explicit cached_tokens=0
is a measured miss. A dropped Trace summary may be recovered from its original
usage payload; absence of the ORIGINAL payload does not imply zero.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from statistics import mean
from unittest.mock import patch

ARK_MODEL = "ark/doubao-seed-character-260628"
ARK_PRICES = {"input": 0.8, "cached_input": 0.16, "output": 2.0}
MAX_QUALIFIED_INPUT = 32000
SEARCH_PRICE_CNY = 0.004


def _counter(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def ark_usage(usage, route):
    """Read exact provider fields, distinguishing missing and measured zero."""
    if route not in {"chat.completions", "responses.web_search"}:
        raise ValueError("Unqualified Ark route")
    if not isinstance(usage, dict):
        return {"status": "usage_missing"}
    inp, out, details = (
        ("prompt_tokens", "completion_tokens", "prompt_tokens_details")
        if route == "chat.completions"
        else ("input_tokens", "output_tokens", "input_tokens_details")
    )
    cached = usage.get(details)
    counts = [
        _counter(usage.get(inp)),
        _counter(usage.get(out)),
        _counter(cached.get("cached_tokens")) if isinstance(cached, dict) else None,
    ]
    if any(v is None for v in counts):
        return {
            "status": "usage_incomplete",
            "input_tokens": counts[0],
            "output_tokens": counts[1],
            "cached_input_tokens": counts[2],
        }
    if counts[2] > counts[0]:
        return {
            "status": "usage_invalid",
            "input_tokens": counts[0],
            "output_tokens": counts[1],
        }
    searches = 0
    if route == "responses.web_search":
        tool = usage.get("tool_usage")
        searches = _counter(tool.get("web_search")) if isinstance(tool, dict) else None
        if searches is None:
            return {
                "status": "search_usage_missing",
                "input_tokens": counts[0],
                "output_tokens": counts[1],
                "cached_input_tokens": counts[2],
            }
    return {
        "status": "measured",
        "input_tokens": counts[0],
        "output_tokens": counts[1],
        "cached_input_tokens": counts[2],
        "web_search_calls": searches,
        "cache_evidence": (
            "provider_explicit_zero" if counts[2] == 0 else "provider_explicit_hit"
        ),
    }


def recover_ark_trace_cache(row):
    """Read-only recovery, not a DB backfill or a rule for all providers."""
    if row.get("model_name") not in {ARK_MODEL, ARK_MODEL.split("/", 1)[1]}:
        return None
    top = row.get("prompt_token_details")
    if isinstance(top, dict) and _counter(top.get("cache_read")) is not None:
        return top["cache_read"]
    outputs = row.get("outputs_json") or {}
    try:
        raw = outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]
        details = raw.get("input_token_details") or {}
        for key in ("cache_read", "cached_tokens"):
            if _counter(details.get(key)) is not None:
                return details[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return None


def priced_call(model, route, measurement):
    input_tokens = _counter(measurement.get("input_tokens"))
    if model != ARK_MODEL or (
        input_tokens is not None and input_tokens > MAX_QUALIFIED_INPUT
    ):
        return {"status": "price_unknown", "cost_cny": None}
    if measurement.get("status") != "measured":
        return {"status": measurement.get("status", "usage_missing"), "cost_cny": None}
    if model != ARK_MODEL or measurement["input_tokens"] > MAX_QUALIFIED_INPUT:
        return {"status": "price_unknown", "cost_cny": None}
    if route not in {"chat.completions", "responses.web_search"}:
        return {"status": "price_unknown", "cost_cny": None}
    from app.services.llm.pricing import estimate_cost_cny

    # Isolated evaluation only: use the validated original production rate,
    # never an unavailable DB cache that would turn missing prices into zero.
    with patch("app.services.runtime_config.get_pricing", return_value=ARK_PRICES):
        tokens = estimate_cost_cny(
            model,
            measurement["input_tokens"],
            measurement["output_tokens"],
            measurement["cached_input_tokens"],
        )
    search_cost = measurement["web_search_calls"] * SEARCH_PRICE_CNY
    return {
        "status": "priced",
        "token_cost_cny": tokens,
        "search_cost_cny": search_cost,
        "cost_cny": tokens + search_cost,
        "basis": "normalized_without_free_quota",
    }


def search_bill_increment(calls, prior_calls, free_limit=20000):
    """Quota-adjusted separate estimate; never used to claim optimization gains."""
    if any(_counter(n) is None for n in (calls, prior_calls, free_limit)):
        raise ValueError("Search counts must be nonnegative integers")
    return (
        max(0, prior_calls + calls - free_limit) - max(0, prior_calls - free_limit)
    ) * SEARCH_PRICE_CNY


def stable_prefix(messages):
    """Hash exact first system segment; no whitespace/date normalization."""
    if not messages or messages[0].get("role") != "system":
        return None
    text = messages[0].get("content")
    if not isinstance(text, str):
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def report_routes(calls):
    groups = defaultdict(list)
    for call in calls:
        groups[
            (call["model"], call["route"], call.get("reply_path", "unspecified"))
        ].append(call)
    result = []
    for (model, route, reply_path), rows in sorted(groups.items()):
        valid = [r for r in rows if r["measurement"]["status"] == "measured"]
        inputs = [r["measurement"].get("input_tokens") for r in rows]
        inputs = [x for x in inputs if x is not None]
        priced = [r for r in rows if r["price"]["status"] == "priced"]
        ttfts = [r["ttft_seconds"] for r in rows if r.get("ttft_seconds") is not None]
        completed = [
            r["time_to_complete_seconds"]
            for r in rows
            if r.get("time_to_complete_seconds") is not None
        ]
        total_input = sum(r["measurement"]["input_tokens"] for r in valid)
        result.append(
            {
                "model": model,
                "route": route,
                "reply_path": reply_path,
                "calls": len(rows),
                "measured_calls": len(valid),
                "priced_calls": len(priced),
                "max_input_tokens": max(inputs) if inputs else None,
                "above_24000": sum(n > 24000 for n in inputs),
                "above_32000": sum(n > 32000 for n in inputs),
                "mean_input_tokens": mean(inputs) if inputs else None,
                "measured_input_tokens": total_input,
                "measured_output_tokens": sum(
                    r["measurement"]["output_tokens"] for r in valid
                ),
                "measured_cached_input_tokens": sum(
                    r["measurement"]["cached_input_tokens"] for r in valid
                ),
                "cache_hit_token_rate": (
                    sum(r["measurement"]["cached_input_tokens"] for r in valid)
                    / total_input
                    if total_input
                    else None
                ),
                "cache_denominator_input_tokens": total_input,
                "normalized_cost_cny": (
                    sum(r["price"]["cost_cny"] for r in priced)
                    if len(priced) == len(rows)
                    else None
                ),
                "web_search_calls": sum(
                    r["measurement"]["web_search_calls"] for r in valid
                ),
                "p90_ttft_seconds": percentile(ttfts, 0.9),
                "ttft_samples": len(ttfts),
                "p90_nonstream_complete_seconds": percentile(completed, 0.9),
                "nonstream_samples": len(completed),
                "status": (
                    "qualified"
                    if len(valid) == len(rows) == len(priced)
                    else "not_qualified"
                ),
            }
        )
    return result
