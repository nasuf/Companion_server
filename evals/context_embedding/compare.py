"""Paired, fail-closed comparison against a fixed original baseline.

The caller supplies the original report explicitly. This does not authorize a
release: intent accuracy, blinded pairwise preference, client E2E and production
acceptance remain separate gates. Bootstrap resamples conversations, not turns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from evals.context_embedding.measurement import percentile, report_routes


def paired_quality(before, after, targets, iterations=5000):
    def indexed(report):
        rows = report["runs"]
        result = {(r["case"], r["sample"], r["turn"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate sample identities")
        return result

    old, new = indexed(before), indexed(after)
    if old.keys() != new.keys() or not old:
        raise ValueError("Paired sample coverage differs or is empty")
    categories = defaultdict(lambda: defaultdict(list))
    critical_regressions = []
    for key, row in old.items():
        other = new[key]
        if (row["category"], row["critical"]) != (other["category"], other["critical"]):
            raise ValueError("Scenario classification changed")
        a, b = row.get("judge", {}).get("pass"), other.get("judge", {}).get("pass")
        if type(a) is not bool or type(b) is not bool:
            raise ValueError("Unjudged samples cannot pass a release gate")
        categories[row["category"]][row["case"]].append(int(b) - int(a))
        if row["critical"] and a and not b:
            critical_regressions.append(key)
    if not targets or not set(targets).issubset(categories):
        raise ValueError("Name existing target categories before comparison")
    result = []
    rng = random.Random(260526)
    for category, cases in sorted(categories.items()):
        changes = [mean(values) for values in cases.values()]
        draws = sorted(
            mean(rng.choices(changes, k=len(changes))) for _ in range(iterations)
        )
        low, high = percentile(draws, 0.025), percentile(draws, 0.975)
        change = mean(changes)
        passed = (
            change > 0 and low >= 0 if category in targets else change >= -0.02 - 1e-12
        )
        result.append(
            {
                "category": category,
                "scenario_count": len(cases),
                "change": change,
                "paired_cluster_95_ci": [low, high],
                "passed": passed,
            }
        )
    return {
        "passed": not critical_regressions and all(r["passed"] for r in result),
        "critical_regressions": critical_regressions,
        "categories": result,
    }


def compare(before, after, targets):
    for key in ("cases_sha256", "model", "repeats", "model_options", "judge_protocol"):
        if key not in before or key not in after:
            raise ValueError(f"Missing evaluation protocol: {key}")
        if before[key] != after[key]:
            raise ValueError(f"Incompatible baseline: {key}")
    quality = paired_quality(before, after, targets)
    old = report_routes([c for r in before["runs"] for c in r["calls"]])
    new = report_routes([c for r in after["runs"] for c in r["calls"]])
    key = lambda r: (r["model"], r["route"], r["reply_path"])
    base = {key(r): r for r in old}
    if set(base) != {key(r) for r in new}:
        raise ValueError("Actual route mix changed; qualify separately before pooling")
    gates = []
    for row in new:
        prior = base[key(row)]
        reasons = []
        if row["status"] != "qualified" or prior["status"] != "qualified":
            reasons.append("usage_or_price_unknown")
        else:
            for metric in ("mean_input_tokens", "normalized_cost_cny"):
                old_value, new_value = prior[metric], row[metric]
                # Both reports cover the same conversation/sample/turn keys.
                # Compare total cost over that fixed population, not per-call
                # cost: a candidate that retries twice must pay for both calls.
                if new_value > old_value * 1.1:
                    reasons.append(metric + "_over_10_percent")
            if row["reply_path"] == "main_llm" and row["route"] == "chat.completions":
                if row["mean_input_tokens"] > 5441 * 1.1:
                    reasons.append("original_5441_token_ceiling")
                if (
                    row["cache_hit_token_rate"] is None
                    or row["cache_hit_token_rate"] < 0.347 - 0.03
                ):
                    reasons.append("original_34_7_percent_cache_floor")
            if (
                row["cache_hit_token_rate"] is None
                or prior["cache_hit_token_rate"] is None
            ):
                reasons.append("cache_rate_unavailable")
            elif row["cache_hit_token_rate"] < prior["cache_hit_token_rate"] - 0.03:
                reasons.append("paired_cache_drop_over_3_points")
            if row["route"] == "responses.web_search":
                # Production is nonstream: completion latency is not a TTFT.
                reasons.append("nonstream_ttft_gate_unproven")
            elif (
                row["ttft_samples"] != row["calls"]
                or prior["ttft_samples"] != prior["calls"]
            ):
                reasons.append("ttft_missing")
            elif row["p90_ttft_seconds"] > prior["p90_ttft_seconds"] * 1.1:
                reasons.append("p90_ttft_over_10_percent")
        gates.append(
            {"route": list(key(row)), "passed": not reasons, "reasons": reasons}
        )
    return {
        "passed": quality["passed"] and all(r["passed"] for r in gates),
        "quality": quality,
        "resource_gates": gates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--targets", required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.targets.split(","),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
