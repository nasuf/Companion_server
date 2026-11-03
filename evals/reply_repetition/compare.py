"""CHAT01.10.2 两组回放的配对比较。

一组可以由几次回放组成 (组名用 + 连接)。每个用例先在组内取各采样的平均 (复读率、回应率…), 再按用例
配对相减。区间由两层噪声的方差相加 (见 paired): 用例层 (按用户聚集的 bootstrap 与逐例采样噪声取大 ——
生产里一个高频用户能占三成轮次, 按用例独立抽会把区间算窄) 与整次回放层 (由 A/A 标定)。没评到的采样
(评判失败 / 被拒收) 不进均值, 数量单独报。

    python -m evals.reply_repetition.compare --cases <cases.json> --scores <scores.json> \\
        --candidate c1+c2+… --baseline b1+b2+… --output <report.json> [--gate]

门禁 (--gate, 供 CHAT01.02 调用): 候选相对基线,
- 正例复读率差值的 95% 上界 ≤ 标定界值 (不能明显更爱复述)
- 全部用例回应率差值的 95% 下界 ≥ -标定界值 (不能明显更常答非所问)
- 两组都跑完的用例 ≥ MIN_COVERAGE (正例与全部分别算), 且每例两组采样数相同
- 两组的回放次数与每例采样数不少于标定时, 且比较时用的就是这份标定
界值来自仓库里的 calibration.json (A/A 标定, 见 calibrate); 没有标定文件时退回先验
MAX_REPETITION_INCREASE / MAX_RESPONDED_DROP 并标 calibrated=False。

标定: python -m evals.reply_repetition.compare --cases ... --scores ... \\
        --calibrate current+current-b+current-c+current-d nofix+nofix-b --runs-per-arm 8 --output calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
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


def in_arm(row_arm: str, arm: str) -> bool:
    """组名可以用 "+" 把几次回放并成一组 (如 current+current-b: 同一份源码跑两遍, 每例 10 次采样)。"""
    return row_arm in arm.split("+")


def case_values(scores: list[dict], arm: str) -> dict[str, dict[str, list[float]]]:
    """{case: {metric: 该组各采样的取值}}; 没评到的采样不放进去。"""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scores:
        if not in_arm(row["arm"], arm):
            continue
        for metric in (*METRICS, "chars", "bubble_count", "seconds"):
            if row[metric] is not None:
                out[row["case"]][metric].append(float(row[metric]))
    return out


def case_means(scores: list[dict], arm: str) -> dict[str, dict[str, float | None]]:
    """{case: {metric: 该组各采样的平均}}; 一个都没评到的指标为 None。"""
    return {case: {m: (statistics.fmean(v) if (v := values.get(m)) else None)
                   for m in (*METRICS, "chars", "bubble_count", "seconds")}
            for case, values in case_values(scores, arm).items()}


def _sum_squares(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return sum((v - mean) ** 2 for v in values)


# 95% 双侧 t 分位数 (自由度 → 值); 整次回放的标准差只由几次 A/A 估出, 自由度很小, 不能当已知量用 1.96
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df: int) -> float:
    """表里没有的自由度取不超过它的最大一档 (分位数更大, 偏保守)。"""
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    return _T975[max(d for d in _T975 if d <= df)] if df <= 30 else 1.96


def paired(cases: list[dict], candidate: dict, baseline: dict, metric: str,
           rng: random.Random, run_se: float | None = None, run_df: int | None = None) -> dict | None:
    """候选 − 基线 的平均差和 95% 区间 (candidate / baseline 为 case_values 的结果)。

    三层噪声, 方差相加 (不是取最宽的那个):
    - 用例层: 按用户聚集的 bootstrap ("换一批用例/用户结果会不会变") 与逐例采样噪声的解析标准误
      (每例两组合并组内方差 × (1/n_候选 + 1/n_基线), "同样的用例再跑一遍会不会变") 取大。A/A 实测
      (2026-11-03): 只有 21 个用户时 bootstrap 低估了后者;
    - 整次回放层 (run_se 由标定给出, 已含采样噪声): 同一份源码的几次回放整体均值相差比逐例采样噪声
      预期的大 (提供方那一侧的状态等)。只把超出采样噪声的那部分加上去, 用 t 分位数 (run_df)。
    半宽 = √((1.96·用例层)² + (t·√max(run_se² − 采样², 0))²)。
    """
    rows = []
    for c in cases:
        x = (candidate.get(c["id"]) or {}).get(metric)
        y = (baseline.get(c["id"]) or {}).get(metric)
        if x and y:
            rows.append((c["user"], statistics.fmean(x) - statistics.fmean(y), x, y))
    if not rows:
        return None
    by_user: dict[str, list[float]] = defaultdict(list)
    for user, diff, _, _ in rows:
        by_user[user].append(diff)
    users = sorted(by_user)
    draws = []
    for _ in range(BOOTSTRAP):
        picked = [d for u in rng.choices(users, k=len(users)) for d in by_user[u]]
        draws.append(statistics.fmean(picked))
    draws.sort()
    boot = (draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1])
    variance = 0.0
    for _, _, x, y in rows:
        df = len(x) + len(y) - 2
        # 每组只有 1 个采样时没有组内自由度: 用合并后的总体方差, 不当作 0
        within = (_sum_squares(x) + _sum_squares(y)) / df if df > 0 else statistics.pvariance(x + y)
        variance += within * (1 / len(x) + 1 / len(y))
    se = math.sqrt(variance) / len(rows)
    se_case = max(se, (boot[1] - boot[0]) / (2 * 1.96))
    run_extra = math.sqrt(max((run_se or 0.0) ** 2 - se ** 2, 0.0))
    half = math.hypot(1.96 * se_case, (t975(run_df) if run_extra else 1.96) * run_extra)
    diff = statistics.fmean(d for _, d, _, _ in rows)
    out = {"diff": round(diff, 4), "ci95": [round(diff - half, 4), round(diff + half, 4)],
           "ci95_bootstrap": [round(boot[0], 4), round(boot[1], 4)], "se_sampling": round(se, 4),
           "cases": len(rows), "users": len(users)}
    if run_se is not None:
        out["se_runs"] = round(run_se, 4)
    return out


def arm_summary(cases: list[dict], means: dict, scores: list[dict], arm: str) -> dict:
    rows = [r for r in scores if in_arm(r["arm"], arm)]

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
        for arm in counts:
            if in_arm(row["arm"], arm):
                counts[arm][row["case"]] += 1
    out = {}
    for group in ("all", "positive", "negative"):
        subset = [c["id"] for c in cases if group == "all" or c["kind"] == group]
        paired_ids = [i for i in subset if counts[candidate][i] and counts[baseline][i]]
        out[group] = {"cases": len(subset), "paired": len(paired_ids),
                      "ratio": round(len(paired_ids) / len(subset), 4) if subset else None}
    out["missing"] = {arm: sorted(c["id"] for c in cases if not counts[arm][c["id"]]) for arm in counts}
    out["min_samples"] = {arm: min((counts[arm][c["id"]] for c in cases if counts[arm][c["id"]]), default=0)
                          for arm in counts}
    out["unequal_samples"] = sorted(c["id"] for c in cases
                                    if counts[candidate][c["id"]] != counts[baseline][c["id"]]
                                    and counts[candidate][c["id"]] and counts[baseline][c["id"]])
    return out


def arm_runs(scores: list[dict], arm: str) -> list[str]:
    """组里实际有数据的回放 (去重)。点名了却没有数据的回放直接报错 —— 不能靠组名凑次数。"""
    parts = list(dict.fromkeys(arm.split("+")))
    present = {row["arm"] for row in scores}
    missing = [p for p in parts if p not in present]
    if missing:
        raise ValueError(f"No scores for run(s) {missing} in arm {arm!r}")
    return parts


GATED = {"positive_repeated": "upper", "all_responded": "lower"}
SUBSETS = {"positive_repeated": ("positive", "repeated"), "all_responded": ("all", "responded")}


def compare(cases: list[dict], scores: list[dict], candidate: str, baseline: str,
            calibration: dict | None = None) -> dict:
    """calibration 给出时, 门禁看的两项把整次回放之间的波动也算进区间 (见 paired 的 run_se)。"""
    rng = random.Random(SEED)
    runs = {arm: len(arm_runs(scores, arm)) for arm in (candidate, baseline)}
    means = {arm: case_means(scores, arm) for arm in (candidate, baseline)}
    values = {arm: case_values(scores, arm) for arm in (candidate, baseline)}
    groups = {"all": cases, "positive": [c for c in cases if c["kind"] == "positive"],
              "negative": [c for c in cases if c["kind"] == "negative"]}
    gated = calibration_metrics(calibration) if calibration else {}
    spread = math.sqrt(1 / runs[candidate] + 1 / runs[baseline])
    diffs = {}
    for group, subset in groups.items():
        for metric in (*METRICS, "chars"):
            key = f"{group}_{metric}"
            entry = gated.get(key)
            diffs[key] = paired(subset, values[candidate], values[baseline], metric, rng,
                                entry["run_sd"] * spread if entry else None, entry["df"] if entry else None)
    return {"candidate": candidate, "baseline": baseline, "bootstrap": BOOTSTRAP, "seed": SEED,
            "runs": runs, "calibration_run_sd": {m: v["run_sd"] for m, v in gated.items()},
            "arms": {arm: arm_summary(cases, means[arm], scores, arm) for arm in (candidate, baseline)},
            "coverage": coverage(cases, scores, candidate, baseline), "paired_diff": diffs}


# A/A 标定规则 (2026-11-03; 修订过程见 calibration.json 的 rule_history):
# 门禁是非劣效检验 —— 区间上界 (或下界) 不越过界值 δ 即通过。同一份源码的几次回放 (A/A) 真实差值为 0。
# 1. 整次回放层: 几组 A/A 合并估计一次回放 (每例 samples_per_run 次采样) 整体均值的标准差 run_sd (df = Σ(k-1))。
# 2. 用例层: 同组相邻两次回放 "1 vs 1" 比较时, 逐例采样标准误与 bootstrap 折算标准误取大, 各对再取最大。
# 3. 每组 R 次回放时, 预期半宽 = √((1.96·用例层/√R)² + (t_df·√max(2·run_sd² − 采样², 0)/√R)²),
#    与 paired 的区间同一算法。门禁同时查两项, 每项 2.5% 误判失败率: δ = 2 × 预期半宽, 向上取到 0.5 个百分点。
# 4. 界值超过 MAX_MARGIN: 噪声大到门禁发现不了有意义的退步, 标定作废 (该加每组回放次数)。
# 界值只对标定时的每组回放次数成立 (门禁不会按次数缩放): 要求两组都有 runs_per_arm 次回放、每例采样数够。
AA_FACTOR = 2  # δ = 2 × 预期半宽
MAX_MARGIN = 0.06
DEFAULT_CALIBRATION = Path(__file__).with_name("calibration.json")


def _ceil_half_point(value: float) -> float:
    return math.ceil(round(value * 200, 6)) / 200


def calibrate(cases: list[dict], scores: list[dict], aa_groups: list[list[str]], runs_per_arm: int) -> dict:
    """aa_groups: 若干组 "同一份源码的几次回放" (每组至少 2 次, 组与组的源码可以不同)。"""
    if not aa_groups or any(len(g) < 2 for g in aa_groups):
        raise ValueError("Each A/A group needs at least two runs")
    runs = [r for g in aa_groups for r in g]
    values = {r: case_values(scores, r) for r in runs}
    out = {}
    for metric, side in GATED.items():
        group, field = SUBSETS[metric]
        subset = [c for c in cases if (group == "all" or c["kind"] == group)
                  and all((values[r].get(c["id"]) or {}).get(field) for r in runs)]
        if not subset:
            raise ValueError(f"No case scored in every A/A run for {metric}")
        run_means = {r: statistics.fmean(statistics.fmean(values[r][c["id"]][field]) for c in subset) for r in runs}
        ss = sum(_sum_squares([run_means[r] for r in g]) for g in aa_groups)
        df = sum(len(g) - 1 for g in aa_groups)
        run_sd = math.sqrt(ss / df)
        rng = random.Random(SEED)
        pairs = [paired(subset, values[g[i]], values[g[i + 1]], field, rng)
                 for g in aa_groups for i in range(len(g) - 1)]
        se1 = max(p["se_sampling"] for p in pairs)
        case1 = max(max(p["se_sampling"], (p["ci95_bootstrap"][1] - p["ci95_bootstrap"][0]) / (2 * 1.96)) for p in pairs)
        extra1 = math.sqrt(max(2 * run_sd ** 2 - se1 ** 2, 0.0))
        half = math.hypot(1.96 * case1, (t975(df) if extra1 else 1.96) * extra1) / math.sqrt(runs_per_arm)
        margin = _ceil_half_point(AA_FACTOR * half)
        if margin > MAX_MARGIN:
            raise ValueError(f"A/A noise too large for {metric} at {runs_per_arm} runs per arm: "
                             f"margin {margin} > {MAX_MARGIN}")
        out[metric] = {"side": side, "cases": len(subset), "run_means": {r: round(m, 4) for r, m in run_means.items()},
                       "aa_groups": aa_groups, "run_sd": round(run_sd, 4), "df": df,
                       "se_sampling_one_run_pair": round(se1, 4), "se_case_one_run_pair": round(case1, 4),
                       "expected_half_width": round(half, 4), "margin": margin}
    return out


def calibration_document(metrics: dict, meta: dict) -> dict:
    """写入仓库的标定文件: 界值 + 来源说明 (gate() 读其中的 metrics 与 runs_per_arm)。"""
    return {"_note": ("CHAT01.10.2 门禁界值, 由 A/A (同一份源码多次回放) 按 compare.calibrate 的规则标定。"
                      "换主模型、评判模型、用例集或回放方式后要重新标定; 以后每次门禁里基线的几次回放"
                      "也是 A/A 数据, 攒够后可重新标定。"),
            **meta, "rule": {"factor": round(AA_FACTOR, 4), "max_margin": MAX_MARGIN, "round_to": 0.005},
            "metrics": metrics}


def calibration_metrics(calibration: dict) -> dict:
    """校验标定内容; 缺项、界值非法或超上限都直接报错, 不悄悄退回先验值。"""
    metrics = calibration.get("metrics", calibration)
    for metric in GATED:
        entry = metrics.get(metric) or {}
        margin, run_sd = entry.get("margin"), entry.get("run_sd")
        if not isinstance(margin, (int, float)) or not 0 < margin <= MAX_MARGIN:
            raise ValueError(f"Invalid calibration margin for {metric}: {margin!r}")
        if not isinstance(run_sd, (int, float)) or run_sd < 0:
            raise ValueError(f"Invalid calibration run_sd for {metric}: {run_sd!r}")
        if not isinstance(entry.get("df"), int) or entry["df"] < 1:
            raise ValueError(f"Invalid calibration df for {metric}: {entry.get('df')!r}")
    return {m: metrics[m] for m in GATED}


def gate(report: dict, calibration: dict | None = None) -> dict:
    """calibration 为仓库里的 calibration.json; None 时用先验界值并标 calibrated=False,
    CHAT01.02 应只认 calibrated=True 的结果。比较时须把同一份标定传给 compare (区间才含整次回放的波动)。"""
    cov = report["coverage"]
    if calibration is None:
        margins = {"positive_repeated": MAX_REPETITION_INCREASE, "all_responded": MAX_RESPONDED_DROP}
        needed_runs, needed_samples, same_calibration = 1, 1, True
    else:
        metrics = calibration_metrics(calibration)
        margins = {m: v["margin"] for m, v in metrics.items()}
        needed_runs = int(calibration.get("runs_per_arm") or 1)
        needed_samples = needed_runs * int(calibration.get("samples_per_run") or 1)
        same_calibration = report.get("calibration_run_sd") == {m: v["run_sd"] for m, v in metrics.items()}
    rep = report["paired_diff"].get("positive_repeated")
    resp = report["paired_diff"].get("all_responded")
    checks = {
        "repetition_not_worse": rep is not None and rep["ci95"][1] <= margins["positive_repeated"],
        "responded_not_worse": resp is not None and resp["ci95"][0] >= -margins["all_responded"],
        "coverage": all((cov[g]["ratio"] or 0) >= MIN_COVERAGE for g in ("all", "positive")),
        "equal_samples": not cov["unequal_samples"],
        # 界值按标定时的每组回放次数与采样数算; 少了噪声更大, 界值不再成立
        "enough_runs": min((report.get("runs") or {}).values(), default=0) >= needed_runs,
        "enough_samples": min(cov.get("min_samples", {}).values(), default=0) >= needed_samples,
        # 比较时用的正是这份标定 (区间里的整次回放波动与界值同源)
        "same_calibration": same_calibration,
    }
    return {"passed": all(checks.values()), "checks": checks, "calibrated": calibration is not None,
            "thresholds": {"max_repetition_increase": margins["positive_repeated"],
                           "max_responded_drop": margins["all_responded"], "min_coverage": MIN_COVERAGE,
                           "runs_per_arm": needed_runs, "samples_per_case": needed_samples}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION if DEFAULT_CALIBRATION.exists() else None,
                        help="门禁界值 (默认仓库里的 calibration.json, 由 A/A 标定)")
    parser.add_argument("--calibrate", nargs="+", metavar="RUN+RUN[+…]",
                        help="标定模式: 若干组同一份源码的回放 (组内用 + 连接) → 写标定文件到 --output")
    parser.add_argument("--runs-per-arm", type=int, default=4, help="标定模式: 门禁每组要几次回放")
    parser.add_argument("--meta", default="{}", help="标定模式: 写进标定文件的来源说明 (JSON)")
    args = parser.parse_args()
    cases, scores = json.loads(args.cases.read_text()), json.loads(args.scores.read_text())
    if not args.calibrate and not (args.candidate and args.baseline):
        parser.error("--candidate and --baseline are required unless --calibrate is given")
    if args.calibrate:
        metrics = calibrate(cases, scores, [g.split("+") for g in args.calibrate], args.runs_per_arm)
        document = calibration_document(metrics, {"runs_per_arm": args.runs_per_arm, **json.loads(args.meta)})
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({m: v["margin"] for m, v in document["metrics"].items()}))
        raise SystemExit(0)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    report = compare(cases, scores, args.candidate, args.baseline, calibration)
    if args.gate:
        report["gate"] = gate(report, calibration)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    args.output.chmod(0o600)
    print(json.dumps({k: report["paired_diff"][k] for k in ("positive_repeated", "negative_repeated", "all_responded")},
                     ensure_ascii=False))
    raise SystemExit(0 if not args.gate or report["gate"]["passed"] else 1)
