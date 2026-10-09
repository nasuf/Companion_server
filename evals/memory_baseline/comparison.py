"""Frozen activation gates; a complete baseline alone never activates anything."""

from collections import defaultdict
import json
from pathlib import Path

from .dataset import split_cases, fingerprint
from .metrics import grade, paired_interval, percentile, summarize

GATES = json.loads(Path(__file__).with_name("gates.json").read_text())


def compare(cases, rows, *, target_groups, samples=1, live_report=None):
    report = summarize(cases, rows, samples=samples)
    if not report["complete"]:
        return {
            "status": "blocked",
            "reason": "incomplete_retrieval_evidence",
            "quality_passed": False,
        }
    if not target_groups or not set(target_groups) <= {c["group"] for c in cases}:
        raise ValueError("Declare the optimization target groups before comparing")
    index = {(r["case"], r["sample"], r["arm"], r["cache"]): r for r in rows}
    membership = split_cases(cases)
    segments = []
    checks = {}
    identities = {
        arm: {r.get("code_tree_sha256") for r in rows if r["arm"] == arm}
        for arm in ("baseline", "candidate")
    }
    checks["source_identity"] = all(
        len(v) == 1 and all(isinstance(x, str) and len(x) == 64 for x in v)
        for v in identities.values()
    )
    checks["different_implementations"] = (
        identities["baseline"] != identities["candidate"]
    )
    for partition in ("development", "holdout"):
        for group in ("overall", *sorted(set(target_groups))):
            clusters = defaultdict(list)
            core_regressions = 0
            hard = 0
            for c in cases:
                if membership[c["id"]] != partition or (
                    group != "overall" and c["group"] != group
                ):
                    continue
                for sample in range(samples):
                    # Cache hit/miss quality must agree; latency gates are separate.
                    for mode in ("cold", "warm"):
                        a, b = (
                            index[c["id"], sample, arm, mode]
                            for arm in ("baseline", "candidate")
                        )
                        ag, bg = grade(c, a), grade(c, b)
                        delta = (
                            bg["stages"]["selected"]["recall"]
                            - ag["stages"]["selected"]["recall"]
                        )
                        clusters[c["family"]].append(delta)
                        core_regressions += int(
                            (c["critical_model"] or c["hard_invariant"]) and delta < 0
                        )
                        hard += int(
                            c["hard_invariant"]
                            and bool(bg["stages"]["selected"]["forbidden"])
                        )
            interval = paired_interval(clusters)
            checks[f"{partition}.{group}.recall"] = interval["delta"] >= (
                GATES["overall_minimum_recall_delta"]
                if group == "overall"
                else GATES["target_group_minimum_recall_delta"]
            )
            checks[f"{partition}.{group}.confidence"] = interval["lower"] >= (
                GATES["overall_minimum_delta_lower_bound"]
                if group == "overall"
                else GATES["target_group_minimum_delta_lower_bound"]
            )
            checks[f"{partition}.{group}.hard_failures"] = hard == 0
            checks[f"{partition}.{group}.core_regressions"] = core_regressions == 0
            segments.append(
                {
                    "partition": partition,
                    "group": group,
                    "recall_interval": interval,
                    "candidate_hard_failures": hard,
                    "core_regressions": core_regressions,
                }
            )
    for mode in ("cold", "warm"):
        a, b = (
            [r["latency_ms"] for r in rows if r["arm"] == arm and r["cache"] == mode]
            for arm in ("baseline", "candidate")
        )
        checks[f"{mode}.p95_latency"] = (
            percentile(b, 0.95)
            <= percentile(a, 0.95) * GATES["maximum_p95_latency_ratio"]
        )
    checks["minimum_scenarios"] = (
        len({c["family"] for c in cases}) >= GATES["minimum_independent_scenarios"]
    )
    checks["live_evidence"] = bool(
        live_report
        and live_report.get("complete")
        and live_report.get("requested_pairs", 0)
        >= GATES["minimum_critical_model_pairs"]
        and live_report.get("retrieval_sha256") == fingerprint(rows)
        and live_report.get("dataset_sha256") == fingerprint(cases)
    )
    # Model cost and answer risks cannot be inferred from a retrieval-only run.
    checks["model_cost"] = False
    checks["model_p95_latency"] = False
    checks["model_risks"] = False
    if checks["live_evidence"]:
        byarm = {s["arm"]: s for s in live_report["segments"]}
        a, b = byarm["baseline"], byarm["candidate"]
        checks["model_cost"] = (
            b["model_cost_cny"]
            <= a["model_cost_cny"] * GATES["maximum_mean_model_cost_ratio"]
        )
        checks["model_p95_latency"] = (
            b["latency_p95_ms"]
            <= a["latency_p95_ms"] * GATES["maximum_p95_latency_ratio"]
        )
        checks["model_risks"] = (
            b["stale_misuse"] <= a["stale_misuse"]
            and b["persona_drift"] <= a["persona_drift"]
            and b["answer_correct"] >= a["answer_correct"]
        )
    return {
        "status": "passed" if all(checks.values()) else "blocked",
        "quality_passed": all(checks.values()),
        "checks": checks,
        "segments": segments,
        "rules": GATES,
        "scope": "algorithm activation only; does not replace application integration/release tests or reviews",
    }
