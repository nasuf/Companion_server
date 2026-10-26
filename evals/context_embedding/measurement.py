"""Strict route-aware measurements; missing evidence is never priced as free.

Ark rule qualified against 2026-10-26 raw responses: an explicit cached_tokens=0
is a measured miss. A dropped Trace summary may be recovered from its original
usage payload; absence of the ORIGINAL payload does not imply zero.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from statistics import mean
import unicodedata
from unittest.mock import patch

ARK_MODEL = "ark/doubao-seed-character-260628"
ARK_PRICES = {"input": 0.8, "cached_input": 0.16, "output": 2.0}
MAX_QUALIFIED_INPUT = 32000
SEARCH_PRICE_CNY = 0.004
REPETITION_PROTOCOL = "previous-whole-turn-v1; exact=strip; bigram=unicode-alnum-NFKC-casefold-set-Dice; threshold=0.6"
# Amendment v2 (2026-10-26): the agreed rule is Jaccard. v1 stays reproducible.
REPETITION_PROTOCOL_V2 = "previous-whole-turn-v2; exact=strip; bigram=unicode-alnum-NFKC-casefold-set-Jaccard; threshold=0.6"
_SIMILARITY = {REPETITION_PROTOCOL: "dice", REPETITION_PROTOCOL_V2: "jaccard"}


def previous_reply_repetition(reply, previous, protocol=REPETITION_PROTOCOL):
    """Deterministic evidence, not a judge verdict or a semantic plagiarism claim."""
    if protocol not in _SIMILARITY:
        raise ValueError("Unknown repetition protocol")
    if not isinstance(reply, str) or (previous is not None and not isinstance(previous, str)):
        raise ValueError("Repetition requires reply text and explicit previous reply or null")
    if previous is None or not previous.strip():
        return {"comparable": False, "exact": False, "bigram_similarity": None,
                "high_similarity": False}
    normalize = lambda text: "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                                    if c.isalnum())
    left, right = normalize(reply), normalize(previous)
    a, b = {left[i:i + 2] for i in range(len(left) - 1)}, {right[i:i + 2] for i in range(len(right) - 1)}
    if not (a and b):
        score = 0.0
    elif _SIMILARITY[protocol] == "jaccard":
        score = len(a & b) / len(a | b)
    else:
        score = 2 * len(a & b) / (len(a) + len(b))
    return {"comparable": True, "exact": reply.strip() == previous.strip(),
            "bigram_similarity": score, "high_similarity": score >= 0.6}


def repetition_by_path(runs, protocol=REPETITION_PROTOCOL):
    groups = defaultdict(list)
    for row in runs:
        paths = {(c.get("route"), c.get("reply_path")) for c in row.get("calls", [])}
        if len(paths) != 1 or any(None in path for path in paths):
            raise ValueError("Repetition requires one actual route per reply")
        if "previous_assistant_reply" not in row:
            raise ValueError("Missing previous_assistant_reply repetition evidence")
        groups[next(iter(paths))].append(
            previous_reply_repetition(row["reply"], row["previous_assistant_reply"], protocol))
    result = []
    for (route, path), rows in sorted(groups.items()):
        denominator = sum(r["comparable"] for r in rows)
        result.append({"provider_route": route, "reply_path": path,
                       "reply_count": len(rows), "comparable_replies": denominator,
                       "no_previous_reply": len(rows) - denominator,
                       **{name + "_count": sum(r[name] for r in rows) for name in ("exact", "high_similarity")},
                       **{name + "_rate": sum(r[name] for r in rows) / denominator if denominator else None
                          for name in ("exact", "high_similarity")}})
    return result


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



def message_list_prefix(previous, current):
    """Exact full-wire-message prefix, in UTF-8 bytes, not model tokens.

    Include every role/content/tool field. Never normalize dates or whitespace.
    Equal first system messages alone do not prove a stable request prefix.
    """
    if previous is None:
        return None
    encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")
    left, right = encode(previous), encode(current)
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    equal_messages = 0
    for a, b in zip(previous, current):
        if encode(a) != encode(b):
            break
        equal_messages += 1
    return {"common_prefix_bytes": common, "unit": "UTF-8 bytes",
            "equal_leading_messages": equal_messages,
            "previous_bytes": len(left), "current_bytes": len(right),
            "entire_list_equal": left == right}


def quality_by_path(runs):
    """No unweighted pooled success rate across product paths."""
    groups = defaultdict(list)
    for row in runs:
        paths = {(c.get("route", "unknown"), c.get("reply_path", "unknown"))
                 for c in row.get("calls", [])}
        route, path = next(iter(paths)) if len(paths) == 1 else ("mixed_or_missing", "unknown")
        groups[(row["category"], route, path)].append(row)
    return [{"category": category, "provider_route": route, "reply_path": path,
             "turns": len(rows), "judged": sum(type(r.get("judge", {}).get("pass")) is bool for r in rows),
             "passed": sum(r.get("judge", {}).get("pass") is True for r in rows)}
            for (category, route, path), rows in sorted(groups.items())]

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
        def valid_latency(value):
            return type(value) in (int, float) and math.isfinite(value) and value >= 0
        ttfts = [r["ttft_seconds"] for r in rows if valid_latency(r.get("ttft_seconds"))]
        # Whole request time; what users wait for on non-streaming product paths.
        durations = [r["duration_seconds"] for r in rows if valid_latency(r.get("duration_seconds"))]
        completed = [
            r["time_to_complete_seconds"]
            for r in rows
            if valid_latency(r.get("time_to_complete_seconds"))
        ]
        # First requests (including repeats of an identical case) never qualify.
        eligible = [r for r in valid if r.get("cache_eligible") is True
                    and r.get("turn_index", 0) >= 1 and r.get("replay_id")]
        prefixes = [r["full_message_prefix"] for r in rows
                    if r.get("turn_index", 0) >= 1 and r.get("replay_id")
                    and isinstance(r.get("full_message_prefix"), dict)]
        prefix_bytes = [p["common_prefix_bytes"] for p in prefixes]
        cache_input = sum(r["measurement"]["input_tokens"] for r in eligible)
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
                    sum(r["measurement"]["cached_input_tokens"] for r in eligible)
                    / cache_input
                    if cache_input
                    else None
                ),
                "cache_denominator_input_tokens": cache_input,
                "cache_eligible_calls": len(eligible),
                "cache_excluded_calls": len(rows) - len(eligible),
                "cache_population": "turn 2+ within isolated replay; never repeated-case warmup",
                "full_message_prefix": {
                    "adjacent_pairs": len(prefixes), "unit": "UTF-8 bytes; not model tokens",
                    "min_common_prefix_bytes": min(prefix_bytes) if prefix_bytes else None,
                    "p50_common_prefix_bytes": percentile(prefix_bytes, .5),
                    "p90_common_prefix_bytes": percentile(prefix_bytes, .9),
                    "entire_list_equal_pairs": sum(p["entire_list_equal"] for p in prefixes),
                    "mean_equal_leading_messages": mean(p["equal_leading_messages"] for p in prefixes) if prefixes else None,
                },
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
                "p90_complete_seconds": percentile(durations, 0.9),
                "complete_samples": len(durations),
                "status": (
                    "qualified"
                    if len(valid) == len(rows) == len(priced)
                    else "not_qualified"
                ),
            }
        )
    return result
