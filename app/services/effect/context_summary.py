"""Pure context-usage summaries shared by legacy and derived admin views."""

from app.services.chat import context_policy as policy

CACHE_QUALIFIED_MODEL_PREFIXES = ("ark/",)


def summarize(rows, start, end, now):
    def setting(row):
        n, tokens = str(row.get("rounds") or ""), str(row.get("main_tokens") or "")
        return (
            int(n)
            if n.isdigit() and policy.MIN_ROUNDS <= int(n) <= policy.MAX_ROUNDS
            else None,
            int(tokens) if tokens.isdigit() else None,
        )

    groups = {}
    for row in rows:
        n, main_tokens = setting(row)
        buckets = policy.stored(row.get("tokens_by_model"))
        for model, tokens in buckets.items():
            if not isinstance(tokens, dict):
                continue
            key = (n, main_tokens, model)
            group = groups.setdefault(
                key,
                {
                    "rounds": n,
                    "main_history_tokens": main_tokens,
                    "model": model,
                    "turns": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "cache_known_input_tokens": 0,
                    "cache_unknown_turns": 0,
                },
            )
            group["turns"] += 1
            group["input_tokens"] += int(tokens.get("input") or 0)
            group["output_tokens"] += int(tokens.get("output") or 0)
            cached = tokens.get("cached_input")
            # The usage ledger writes 0 for every provider. Only Ark's 0 is a
            # measured miss (raw responses, 2026-10-26); elsewhere it is unknown.
            qualified = model.startswith(CACHE_QUALIFIED_MODEL_PREFIXES)
            if (
                qualified
                and type(cached) is int
                and 0 <= cached <= int(tokens.get("input") or 0)
            ):
                group["cached_input_tokens"] += cached
                group["cache_known_input_tokens"] += int(tokens.get("input") or 0)
            else:
                group["cache_unknown_turns"] += 1
    # Persisted turn cost is a sum across models. Group it by setting separately
    # rather than attributing the whole turn's cost to each model (double counting).
    costs = {}
    for row in rows:
        n, main_tokens = setting(row)
        item = costs.setdefault(
            (n, main_tokens),
            {
                "rounds": n,
                "main_history_tokens": main_tokens,
                "turns": 0,
                "recorded_token_cost_cny": 0,
                "web_search_calls": 0,
            },
        )
        item["turns"] += 1
        item["recorded_token_cost_cny"] += float(row.get("cost_cny") or 0)
        item["web_search_calls"] += int(row.get("web_search_calls") or 0)
    for g in groups.values():
        g["cache_hit_rate"] = (
            g["cached_input_tokens"] / g["cache_known_input_tokens"]
            if g["cache_known_input_tokens"]
            else None
        )
    return {
        "sampled_at": now.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "scope": "chat foreground usage rows; excludes background and proactive",
        "rows": len(rows),
        "max_rows": 10000,
        "groups": list(groups.values()),
        "costs": [
            {**item, "normalized_search_cost_cny": item["web_search_calls"] * 0.004}
            for item in costs.values()
        ],
        "status": "measured" if rows else "empty",
        "notes": [
            "轮数未知：历史未采集、Trace关闭或已过保留期；不回填当前值。",
            '缓存只对已核实"0=未命中"的供应商（方舟）计入；其他供应商记录的0视为未知，不当作未命中。',
            "费用按回合已记录token费用；搜索按0.004元/次标准化，非免费额度账单。",
            "轮数控制模板上下文；主回复历史长度单独分组，联网搜索回合最多 4000 token。",
        ],
    }


def merge(payloads, start, end, sampled_at):
    """Merge additive counters; recompute ratios from the combined denominator."""
    groups, costs = {}, {}
    for payload in payloads:
        for source in payload["groups"]:
            key = (source["rounds"], source["main_history_tokens"], source["model"])
            target = groups.setdefault(
                key, {k: source[k] for k in ("rounds", "main_history_tokens", "model")}
            )
            for field in (
                "turns",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cache_known_input_tokens",
                "cache_unknown_turns",
            ):
                target[field] = target.get(field, 0) + source[field]
        for source in payload["costs"]:
            key = (source["rounds"], source["main_history_tokens"])
            target = costs.setdefault(
                key, {k: source[k] for k in ("rounds", "main_history_tokens")}
            )
            for field in (
                "turns",
                "recorded_token_cost_cny",
                "web_search_calls",
                "normalized_search_cost_cny",
            ):
                target[field] = target.get(field, 0) + source[field]
    for group in groups.values():
        denominator = group["cache_known_input_tokens"]
        group["cache_hit_rate"] = (
            group["cached_input_tokens"] / denominator if denominator else None
        )
    empty = summarize([], start, end, start)
    count = sum(p["rows"] for p in payloads)
    return {
        **empty,
        "sampled_at": sampled_at,
        "rows": count,
        "max_rows": None,
        "status": "measured" if count else "empty",
        "groups": list(groups.values()),
        "costs": list(costs.values()),
    }
