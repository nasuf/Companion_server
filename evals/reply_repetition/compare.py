"""CHAT01.10.2 两组回放的配对比较。

每个用例先在组内取各采样的平均 (复读率、回应率…), 再按用例配对相减。置信区间用按用户聚集
的 bootstrap: 每次有放回地抽用户, 带上他的全部用例 —— 生产里一个高频用户能占三成轮次,
按用例独立抽会把区间算窄。没评到的采样 (评判失败 / 被拒收) 不进均值, 数量单独报。

    python -m evals.reply_repetition.compare --cases <cases.json> --scores <scores.json> \\
        --candidate current --baseline nofix --output <report.json> [--gate]

门禁 (--gate, 供 CHAT01.02 调用): 候选相对基线,
- 正例复读率差值的 95% 上界 ≤ MAX_REPETITION_INCREASE (不能明显更爱复述)
- 全部用例回应率差值的 95% 下界 ≥ -MAX_RESPONDED_DROP (不能明显更常答非所问)
- 两组都跑完的用例 ≥ MIN_COVERAGE (正例与全部分别算), 且每例两组采样数相同
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

BOOTSTRAP = 2000
SEED = 20261103
MAX_REPETITION_INCREASE = 0.05
MAX_RESPONDED_DROP = 0.05
MIN_COVERAGE = 0.95  # 两组都跑完的用例占比: 候选在难例上崩掉、没出回复时不能因为少了这些例反而显得更好
METRICS = ("repeated", "verbatim", "responded")


def case_means(scores: list[dict], arm: str) -> dict[str, dict[str, float | None]]:
    """{case: {metric: 该组各采样的平均}}; 一个都没评到的指标为 None。"""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    extra: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scores:
        if row["arm"] != arm:
            continue
        for metric in METRICS:
            if row[metric] is not None:
                buckets[row["case"]][metric].append(float(row[metric]))
        for metric in ("chars", "bubble_count", "seconds"):
            extra[row["case"]][metric].append(float(row[metric]))
    out = {}
    for case in set(buckets) | set(extra):
        out[case] = {m: (statistics.fmean(v) if (v := buckets[case].get(m)) else None) for m in METRICS}
        out[case].update({m: statistics.fmean(v) for m, v in extra[case].items()})
    return out


def paired(cases: list[dict], candidate: dict, baseline: dict, metric: str,
           rng: random.Random) -> dict | None:
    """候选 − 基线 的平均差和按用户聚集的 bootstrap 95% 区间。"""
    pairs = [(c["user"], candidate[c["id"]][metric] - baseline[c["id"]][metric]) for c in cases
             if c["id"] in candidate and c["id"] in baseline
             and candidate[c["id"]][metric] is not None and baseline[c["id"]][metric] is not None]
    if not pairs:
        return None
    by_user: dict[str, list[float]] = defaultdict(list)
    for user, diff in pairs:
        by_user[user].append(diff)
    users = sorted(by_user)
    draws = []
    for _ in range(BOOTSTRAP):
        picked = [d for u in rng.choices(users, k=len(users)) for d in by_user[u]]
        draws.append(statistics.fmean(picked))
    draws.sort()
    return {"diff": round(statistics.fmean(d for _, d in pairs), 4),
            "ci95": [round(draws[int(0.025 * BOOTSTRAP)], 4), round(draws[int(0.975 * BOOTSTRAP) - 1], 4)],
            "cases": len(pairs), "users": len(users)}


def arm_summary(cases: list[dict], means: dict, scores: list[dict], arm: str) -> dict:
    rows = [r for r in scores if r["arm"] == arm]

    def rate(metric: str, subset: list[dict]) -> float | None:
        values = [means[c["id"]][metric] for c in subset if c["id"] in means and means[c["id"]][metric] is not None]
        return round(statistics.fmean(values), 4) if values else None

    groups = {"positive": [c for c in cases if c["kind"] == "positive"],
              "negative": [c for c in cases if c["kind"] == "negative"]}
    tags = sorted({t for c in cases for t in c["tags"]})
    seconds = sorted(r["seconds"] for r in rows)
    costs = [r["cost_cny"] for r in rows]
    return {
        "samples": len(rows), "cases": len({r["case"] for r in rows}),
        "unjudged_repetition": sum(r["repeated"] is None for r in rows),
        "unjudged_responded": sum(r["responded"] is None for r in rows),
        **{f"{kind}_{m}": rate(m, subset) for kind, subset in groups.items() for m in METRICS},
        "by_tag_repeated": {t: {"positive": rate("repeated", [c for c in groups["positive"] if t in c["tags"]]),
                                "negative": rate("repeated", [c for c in groups["negative"] if t in c["tags"]]),
                                "cases": sum(t in c["tags"] for c in cases)} for t in tags},
        "mean_chars": rate("chars", cases), "mean_bubbles": rate("bubble_count", cases),
        "main_seconds_p50": seconds[len(seconds) // 2] if seconds else None,
        "main_seconds_p90": seconds[int(len(seconds) * 0.9)] if seconds else None,
        "cost_cny": None if any(c is None for c in costs) else round(sum(costs), 4),
        "cost_unknown_samples": sum(c is None for c in costs),
    }


def coverage(cases: list[dict], scores: list[dict], candidate: str, baseline: str) -> dict:
    """每组各例的采样数; 缺失 (回放失败 / 没跑) 与两组采样数不等的用例单独列出。"""
    counts = {arm: defaultdict(int) for arm in (candidate, baseline)}
    for row in scores:
        if row["arm"] in counts:
            counts[row["arm"]][row["case"]] += 1
    out = {}
    for group in ("all", "positive", "negative"):
        subset = [c["id"] for c in cases if group == "all" or c["kind"] == group]
        paired_ids = [i for i in subset if counts[candidate][i] and counts[baseline][i]]
        out[group] = {"cases": len(subset), "paired": len(paired_ids),
                      "ratio": round(len(paired_ids) / len(subset), 4) if subset else None}
    out["missing"] = {arm: sorted(c["id"] for c in cases if not counts[arm][c["id"]]) for arm in counts}
    out["unequal_samples"] = sorted(c["id"] for c in cases
                                    if counts[candidate][c["id"]] != counts[baseline][c["id"]]
                                    and counts[candidate][c["id"]] and counts[baseline][c["id"]])
    return out


def compare(cases: list[dict], scores: list[dict], candidate: str, baseline: str) -> dict:
    rng = random.Random(SEED)
    means = {arm: case_means(scores, arm) for arm in (candidate, baseline)}
    groups = {"all": cases, "positive": [c for c in cases if c["kind"] == "positive"],
              "negative": [c for c in cases if c["kind"] == "negative"]}
    diffs = {f"{group}_{metric}": paired(subset, means[candidate], means[baseline], metric, rng)
             for group, subset in groups.items() for metric in (*METRICS, "chars")}
    return {"candidate": candidate, "baseline": baseline, "bootstrap": BOOTSTRAP, "seed": SEED,
            "arms": {arm: arm_summary(cases, means[arm], scores, arm) for arm in (candidate, baseline)},
            "coverage": coverage(cases, scores, candidate, baseline), "paired_diff": diffs}


def gate(report: dict) -> dict:
    rep = report["paired_diff"].get("positive_repeated")
    resp = report["paired_diff"].get("all_responded")
    cov = report["coverage"]
    checks = {
        "repetition_not_worse": rep is not None and rep["ci95"][1] <= MAX_REPETITION_INCREASE,
        "responded_not_worse": resp is not None and resp["ci95"][0] >= -MAX_RESPONDED_DROP,
        "coverage": all((cov[g]["ratio"] or 0) >= MIN_COVERAGE for g in ("all", "positive")),
        "equal_samples": not cov["unequal_samples"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "thresholds": {"max_repetition_increase": MAX_REPETITION_INCREASE,
                           "max_responded_drop": MAX_RESPONDED_DROP, "min_coverage": MIN_COVERAGE}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report = compare(json.loads(args.cases.read_text()), json.loads(args.scores.read_text()),
                     args.candidate, args.baseline)
    if args.gate:
        report["gate"] = gate(report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    args.output.chmod(0o600)
    print(json.dumps({k: report["paired_diff"][k] for k in ("positive_repeated", "negative_repeated", "all_responded")},
                     ensure_ascii=False))
    raise SystemExit(0 if not args.gate or report["gate"]["passed"] else 1)
