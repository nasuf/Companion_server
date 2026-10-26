"""A/A generation and same-output judgement repeatability, separated by path.

This measures noise, not candidate quality. It cannot waive a release gate.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def build_gate_policy(calibration):
    """Freeze A/A-derived allowances before any baseline/A/B release replay."""
    if calibration.get("status") != "measured" or not calibration.get("routes"):
        raise ValueError("Measured A/A routes required for preregistration")
    routes = []
    seen = set()
    for row in calibration["routes"]:
        identity = (row["provider_route"], row["reply_path"])
        rate, n = row.get("generation_verdict_flip_rate"), row.get("generation_pairs")
        if identity in seen or any(x in (None, "mixed", "route_changed") for x in identity):
            raise ValueError("A/A routes must be unique known actual paths")
        if type(n) is not int or n < 1 or type(rate) not in (float, int) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Invalid A/A generation flip evidence")
        seen.add(identity)
        routes.append({"provider_route": identity[0], "reply_path": identity[1],
                       "max_other_category_drop": rate, "generation_pairs": n})
    from evals.context_embedding.measurement import REPETITION_PROTOCOL
    fingerprint = hashlib.sha256(json.dumps(calibration, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    return {"version": "claude-review-20261026-v1", "aa_sha256": fingerprint,
            "noise_basis": "actual-path generation verdict flips; judge noise not added",
            "noise_allowances": sorted(routes, key=lambda r: (r["provider_route"], r["reply_path"])),
            "critical": "per case+turn: exactly 3 repeats; candidate failures >=2 and baseline failures =0",
            "main": "strict non-regression", "target": "improvement and cluster CI lower >=0",
            "repetition": REPETITION_PROTOCOL, "state_latency_minimum_per_arm": 36,
            "variant_selection": "default A; B complete gates pass, both repetition rates nonworse on every path with one strict gain, every category within its path AA allowance"}


def policy_sha256(policy):
    return hashlib.sha256(json.dumps(policy, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


AMENDMENT_FIELDS = ("targets", "judge_recovery", "repetition", "latency", "input_tokens",
                    "binding", "variant_selection")


def apply_amendment(policy, amendment):
    """Effective gates = frozen base policy + one recorded, hash-linked amendment.

    Reports keep the base policy they were run under; the amendment never edits
    them and is rejected unless it names that exact base.
    """
    if amendment is None:
        return {**policy, "amendment": None}
    if amendment.get("base_policy_sha256") != policy_sha256(policy):
        raise ValueError("Amendment does not reference the preregistered base policy")
    missing = [f for f in AMENDMENT_FIELDS if f not in amendment]
    if missing:
        raise ValueError(f"Incomplete gate amendment: {missing}")
    from evals.context_embedding.measurement import REPETITION_PROTOCOL_V2
    if amendment["repetition"].get("protocol") != REPETITION_PROTOCOL_V2:
        raise ValueError("Unsupported amended repetition protocol")
    targets = amendment["targets"]
    if not isinstance(targets, dict) or not targets or not all(
            isinstance(v, list) and v and all(isinstance(c, str) for c in v) for v in targets.values()):
        raise ValueError("Amendment targets must name categories per reply path")
    fraction = amendment["judge_recovery"].get("max_excluded_fraction")
    if type(fraction) not in (int, float) or not 0 <= fraction < 0.05:
        raise ValueError("Judge recovery exclusion bound must be explicit and small")
    transport = amendment.get("transport_failure_rule")
    if transport is not None:
        bounds = [transport.get(k) for k in ("max_replaced_fraction", "max_excess_fraction")]
        if not all(type(b) in (int, float) and 0 <= b < 0.05 for b in bounds):
            raise ValueError("Transport failure bounds must be explicit and small")
        transport = {"max_replaced_fraction": bounds[0], "max_excess_fraction": bounds[1]}
    return {**policy, "repetition": amendment["repetition"]["protocol"],
            "amendment": {"version": amendment["version"], "sha256": policy_sha256(amendment),
                          "targets": targets, "max_excluded_fraction": fraction,
                          "unchanged_input_rule": amendment.get("unchanged_input_rule") is True,
                          "transport_failure_rule": transport}}


def latency_observations(report):
    """Preserve call identities so A/A can supplement only the baseline arm."""
    rows = []
    for row in report["runs"]:
        for index, call in enumerate(row.get("calls", [])):
            if call.get("reply_path") != "current_state_shortcircuit":
                continue
            replay = row.get("replay_id")
            if not replay:
                raise ValueError("Latency observation requires independent replay identity")
            rows.append({"identity": [replay, row["case"], row["sample"], row["turn"], index],
                         "model": call["model"], "provider_route": call["route"],
                         "reply_path": call["reply_path"], "ttft_seconds": call.get("ttft_seconds"),
                         "duration_seconds": call.get("duration_seconds")})
    return rows


def calibrate(left, right):
    for field in ("source_files_sha256", "cases_sha256", "prompt_snapshot_sha256",
                  "model", "model_options", "judge_protocol", "cache_protocol",
                  "prefix_protocol", "repeats"):
        if field not in left or left[field] != right.get(field):
            raise ValueError(f"A/A requires identical baseline protocol: {field}")
    if left.get("started_at") == right.get("started_at"):
        raise ValueError("Independent generation runs required")

    def index(report):
        rows = {(r["case"], r["sample"], r["turn"]): r for r in report["runs"]}
        if len(rows) != len(report["runs"]):
            raise ValueError("Duplicate paired identity")
        return rows

    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("A/A sample coverage mismatch")
    groups = defaultdict(list)
    for key, row in a.items():
        other = b[key]
        paths = {tuple((c.get("route"), c.get("reply_path")))
                 for r in (row, other) for c in r.get("calls", [])}
        group = next(iter(paths)) if len(paths) == 1 else ("mixed", "route_changed")
        if row.get("replay_id") == other.get("replay_id"):
            raise ValueError("A/A replays must not share warming namespace")
        verdicts = [r.get("judge", {}).get("pass") for r in (row, other)]
        if any(type(v) is not bool for v in verdicts):
            raise ValueError("A/A contains unjudged generation")
        repeated = []
        for r in (row, other):
            judgments = r.get("judge_repeats", [])
            if len(judgments) < 2 or any(type(j.get("pass")) is not bool for j in judgments):
                raise ValueError("At least two judgments of each identical reply required")
            repeated.extend(j["pass"] != judgments[0]["pass"] for j in judgments[1:])
        groups[group].append({
            "identity": key,
            "text_changed": row["reply"] != other["reply"],
            "generation_verdict_flipped": verdicts[0] != verdicts[1],
            "critical_pass_to_fail": bool(row["critical"] and verdicts[0] and not verdicts[1]),
            "same_output_judge_flips": sum(repeated),
            "same_output_judge_pairs": len(repeated),
        })
    results = []
    for path, rows in sorted(groups.items()):
        pairs = sum(r["same_output_judge_pairs"] for r in rows)
        flips = sum(r["same_output_judge_flips"] for r in rows)
        results.append({
            "provider_route": path[0], "reply_path": path[1],
            "generation_pairs": len(rows),
            "different_text_pairs": sum(r["text_changed"] for r in rows),
            "generation_verdict_flip_rate": sum(r["generation_verdict_flipped"] for r in rows) / len(rows),
            "same_output_judge_pairs": pairs,
            "same_output_judge_flip_rate": flips / pairs,
            "critical_pass_to_fail": [r["identity"] for r in rows if r["critical_pass_to_fail"]],
            "pairs": rows,
        })
    return {"status": "measured", "routes": results,
            "interpretation": "Text variation is not a quality regression; same-output flips measure judge noise. No automatic waiver.",
            "baseline_started_at": [left["started_at"], right["started_at"]],
            "latency_observations": latency_observations(left) + latency_observations(right),
            "latency_protocol": {k: left[k] for k in ("source_files_sha256", "prompt_snapshot_sha256", "model", "model_options", "cache_protocol", "prefix_protocol")},
            "protocol": {k: left[k] for k in ("cases_sha256", "prompt_snapshot_sha256", "model", "judge_protocol", "cache_protocol")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing calibration evidence")
    result = calibrate(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
