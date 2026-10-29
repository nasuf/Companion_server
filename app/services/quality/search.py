"""Evaluation-only, metered Responses search contract (qualified 2026-10-29).

The product's search settings are untouched. Replays use one keyword/round and
the default public-web source/10 results. The two possible model inferences
reserve the full 128k context at the upper tier, without assuming cache hits.
Actual inputs above 32k still FAIL qualification; the upper tier is a budget
guard, not permission to silently extend the approved benchmark price band.

Sources: docs.volcengine.com/docs/ark/{model-list,model-pricing,web-search}.
"""

from decimal import Decimal, ROUND_CEILING

MODEL = "ark/doubao-seed-character-260628"
POLICY = "ark-public-search-one-round-v1"
MAX_INPUT = 2 * 131072
SEARCH_PRICE = Decimal("0.004")  # no free allowance assumed


def ceiling(model, max_tokens):
    if model != MODEL or type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
        raise ValueError("search_route_not_qualified")
    return (
        (Decimal(MAX_INPUT) * Decimal("1.2") + 2 * max_tokens * Decimal("6"))
        / 1_000_000
        + SEARCH_PRICE
    ).quantize(Decimal(".00000001"), rounding=ROUND_CEILING)


def body(model, messages, max_tokens, location=None):
    ceiling("ark/" + model, max_tokens)
    if location is not None and (
        not isinstance(location, dict)
        or not set(location) <= {"type", "country", "region", "city", "timezone"}
        or any(not isinstance(v, str) or len(v) > 100 for v in location.values())
    ):
        raise ValueError("invalid_search_location")
    return dict(
        model=model,
        input=messages,
        stream=False,
        store=False,
        thinking={"type": "disabled"},
        max_output_tokens=max_tokens,
        max_tool_calls=1,
        tool_choice="required",
        tools=[
            dict(
                type="web_search",
                max_keyword=1,
                limit=10,
                sources=["search_engine"],
                **({"user_location": location} if location else {}),
            )
        ],
    )


def usage(payload, max_tokens):
    raw = payload.get("usage") or {}
    used = dict(
        input=raw.get("input_tokens"),
        output=raw.get("output_tokens"),
        cached_input=(raw.get("input_tokens_details") or {}).get("cached_tokens", 0),
        web_search_calls=(raw.get("tool_usage") or {}).get("web_search"),
        operation="web_search",
        search_policy=POLICY,
    )
    validate(used, max_tokens)
    details = (raw.get("tool_usage_details") or {}).get("web_search")
    if details != {"search_engine": used["web_search_calls"]}:
        raise ValueError("search_source_metering_missing_or_unpriced")
    return used


def validate(used, max_tokens):
    if (
        used.get("operation") != "web_search"
        or used.get("search_policy") != POLICY
        or any(
            type(used.get(k)) is not int
            for k in ("input", "output", "cached_input", "web_search_calls")
        )
        or not 0 <= used["cached_input"] <= used["input"] <= MAX_INPUT
        or not 0 <= used["output"] <= 2 * max_tokens
        or used["web_search_calls"] != 1
    ):
        raise ValueError("search_usage_out_of_bounds")
    if used["input"] > 32768:
        raise ValueError("price_unknown_above_32k_search_usage")


def tool_cost(used):
    return SEARCH_PRICE * used["web_search_calls"]
