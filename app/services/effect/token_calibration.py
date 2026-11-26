"""Main-reply input estimates against billed prompt tokens (Q01.04).

Each row is one root trace carrying `windows.main.measured`, joined with the
largest successful LLM run under its `generate_reply`. A chat run is compared
with the calibrated count and the old heuristic; an Ark web-search run bills
the sum of its inferences, so it is compared with the search projection
(2 x text + reserve) instead. Rows without a billed count stay "no_billing"
and never count as an exact match.
"""

from __future__ import annotations

import json

SEARCH_RUN = "ArkResponsesAPI"
SEARCH_TIER = 32768
MIN_BILLED = 1000  # the fit was cross-validated on calls of 1000+ billed tokens


def _obj(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _quantiles(errors: list[float]) -> dict | None:
    if not errors:
        return None
    s = sorted(errors)

    def q(p):
        return round(s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))], 4)
    return {"p5": q(0.05), "p50": q(0.5), "p95": q(0.95), "max_abs": round(max(abs(e) for e in s), 4)}


def summarize(rows: list[dict]) -> dict:
    models: dict[str, dict] = {}
    search = {"turns": 0, "billed_over_32k": 0, "projected_over_32k": 0, "trimmed": 0,
              "max_billed": None, "under_projection": 0, "no_projection": 0}
    no_billing = mismatch = 0
    for row in rows:
        measured = _obj(row.get("measured"))
        billed = _count(row.get("prompt_tokens"))
        if measured is None:
            continue
        if billed is None or not row.get("llm_name"):
            no_billing += 1
            continue
        if row["llm_name"] == SEARCH_RUN:
            cap = _obj(row.get("search_cap"))
            search["turns"] += 1
            search["billed_over_32k"] += billed > SEARCH_TIER
            search["max_billed"] = max(search["max_billed"] or 0, billed)
            projected = _count(cap.get("projected_input_tokens")) if cap else None
            if cap is None or projected is None:
                search["no_projection"] += 1
                continue
            search["projected_over_32k"] += projected > SEARCH_TIER
            search["trimmed"] += cap.get("trimmed") is True
            # The projection is a bound: billing above it means the reserve is short.
            search["under_projection"] += billed > projected
            continue
        expected = str(measured.get("model") or "").rsplit("/", 1)[-1]
        if expected != (row.get("model_name") or ""):
            mismatch += 1  # a fallback model answered; its rate is not the measured one
            continue
        group = models.setdefault(expected, {"model": expected, "calibrated": measured.get("calibrated") is True,
                                             "turns": 0, "turns_1000_plus": 0, "_cal": [], "_old": []})
        group["turns"] += 1
        estimate, heuristic = _count(measured.get("input_tokens")), _count(measured.get("heuristic_input_tokens"))
        if billed < MIN_BILLED or estimate is None:
            continue
        group["turns_1000_plus"] += 1
        group["_cal"].append(estimate / billed - 1)
        if heuristic is not None:
            group["_old"].append(heuristic / billed - 1)
    return {
        "models": [{**{k: v for k, v in g.items() if not k.startswith("_")},
                    "calibrated_error": _quantiles(g["_cal"]), "heuristic_error": _quantiles(g["_old"])}
                   for g in models.values()],
        "search": search,
        "no_billing_turns": no_billing,
        "model_mismatch_turns": mismatch,
        "notes": [
            "误差 = 估值 / 计费输入 − 1；只统计计费 ≥1000 token 的回合（拟合口径）。",
            "旧估算即预算仍在使用的 estimate_tokens，未改预算。",
            "联网回合按两次推理之和计费，与投影（2×文本+结果预留）比较；超过投影说明预留不够。",
            "无计费记录（Trace 关闭/未完成）和备选模型作答的回合单独计数，不计入误差。",
        ],
    }
