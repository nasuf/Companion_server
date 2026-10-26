"""Paired, fail-closed comparison against a fixed original baseline.

The caller supplies the original report explicitly. This does not authorize a
release: intent accuracy, blinded pairwise preference, client E2E and production
acceptance remain separate gates. Bootstrap resamples conversations, not turns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

from evals.context_embedding.measurement import percentile, report_routes, repetition_by_path
from evals.context_embedding.calibrate import apply_amendment, build_gate_policy, latency_observations

# Production calls these paths without streaming (invoke_text): users wait for the
# whole reply, so first-token time would hide a longer or slower answer.
NONSTREAM_REPLY_PATHS = frozenset({"tier", "current_state_shortcircuit"})


def unjudged_identities(*reports):
    return {(r["case"], r["sample"], r["turn"]) for report in reports for r in report["runs"]
            if type(r.get("judge", {}).get("pass")) is not bool}


def judge_exclusions(policy, *reports):
    """Amendment v2: exclude unjudged identities from every arm, bounded and listed."""
    excluded = unjudged_identities(*reports)
    if not excluded:
        return frozenset(), {"excluded": [], "passed": True}
    amendment = policy.get("amendment")
    if not amendment:
        raise ValueError("Unjudged samples cannot pass a release gate")
    rows = {(r["case"], r["sample"], r["turn"]): r for r in reports[0]["runs"]}
    critical = sorted(k for k in excluded if rows.get(k, {}).get("critical"))
    fraction = len(excluded) / len(rows)
    passed = not critical and fraction <= amendment["max_excluded_fraction"]
    return frozenset(excluded), {"excluded": sorted(excluded), "critical_excluded": critical,
                                 "excluded_fraction": fraction, "passed": passed}


def unchanged_paths(before, after):
    """Actual paths whose every first-turn product request is byte-identical.

    The first turn carries the scripted multi-turn history, so history rendering
    is covered. Later turns differ only through the sampled earlier reply; any
    quality, latency or cost difference on such a path cannot come from the
    candidate and is reported, not gated (amendment v3).
    """
    def index(report):
        out = defaultdict(dict)
        for row in report["runs"]:
            if row["turn"] != 0:
                continue
            for call in row.get("calls", []):
                out[(call.get("route"), call.get("reply_path"))][(row["case"], row["sample"])] = (
                    call.get("request_sha256"))
        return out
    def actual(report):
        out = {}
        for row in report["runs"]:
            paths = {(c.get("route"), c.get("reply_path")) for c in row.get("calls", [])}
            out[(row["case"], row["sample"], row["turn"])] = next(iter(paths)) if len(paths) == 1 else None
        return out
    left, right = index(before), index(after)
    found = {path for path in left if left[path] and left[path].keys() == right.get(path, {}).keys()
             and all(h and h == right[path][k] for k, h in left[path].items())}
    # A later turn inherits the candidate's effect through any earlier reply
    # produced on a changed path, so every turn so far must stay on found paths.
    old_paths, new_paths = actual(before), actual(after)
    changed = True
    while changed:
        changed = False
        for (case, sample, turn), path in old_paths.items():
            if path not in found:
                continue
            history = [old_paths.get((case, sample, t)) for t in range(turn + 1)]
            if new_paths.get((case, sample, turn)) != path or any(p not in found for p in history):
                found.discard(path)
                changed = True
    return frozenset(found)


def _targets_for(path, targets, path_targets):
    return set(path_targets.get(path, ())) if path_targets is not None else set(targets or ())


def paired_quality(before, after, targets, iterations=5000, *, policy=None,
                   excluded=frozenset(), path_targets=None, unchanged=frozenset()):
    def indexed(report):
        rows = report["runs"]
        result = {(r["case"], r["sample"], r["turn"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate sample identities")
        return result

    old, new = indexed(before), indexed(after)
    if old.keys() != new.keys() or not old:
        raise ValueError("Paired sample coverage differs or is empty")
    old = {k: v for k, v in old.items() if k not in excluded}
    new = {k: v for k, v in new.items() if k not in excluded}
    categories = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: {'turn_pairs': 0, 'baseline_passed': 0, 'candidate_passed': 0})
    def path(row):
        paths = {(c.get('route'), c.get('reply_path')) for c in row.get('calls', [])}
        if len(paths) != 1 or any(None in pair for pair in paths):
            raise ValueError('Quality requires one known actual route per generated reply')
        return next(iter(paths))
    critical_groups = defaultdict(list)
    for key, row in old.items():
        other = new[key]
        if (row["category"], row["critical"]) != (other["category"], other["critical"]):
            raise ValueError("Scenario classification changed")
        a, b = row.get("judge", {}).get("pass"), other.get("judge", {}).get("pass")
        if type(a) is not bool or type(b) is not bool:
            raise ValueError("Unjudged samples cannot pass a release gate")
        route, reply_path = path(row)
        if path(other) != (route, reply_path):
            raise ValueError('Actual route changed within paired quality sample')
        group = (route, reply_path, row['category'])
        categories[group][row['case']].append(int(b) - int(a))
        counts[group]['turn_pairs'] += 1
        counts[group]['baseline_passed'] += int(a)
        counts[group]['candidate_passed'] += int(b)
        critical_groups[(row["case"], row["turn"])].append(
            (row["sample"], row["critical"], a, b, (route, reply_path) in unchanged))
    critical_results = []
    for (case, turn), rows in sorted(critical_groups.items()):
        if len({r[1] for r in rows}) != 1:
            raise ValueError("Critical classification changed between repeats")
        if not rows[0][1]:
            continue
        # judge_exclusions fails the gate before any critical row is dropped.
        if len(rows) != 3 or len({r[0] for r in rows}) != 3:
            raise ValueError("Critical scenarios require exactly three repeats")
        old_failures = sum(not r[2] for r in rows)
        new_failures = sum(not r[3] for r in rows)
        identical = all(r[4] for r in rows)
        critical_results.append({"case": case, "turn": turn, "repeats": 3,
                                 "baseline_failures": old_failures, "candidate_failures": new_failures,
                                 "request_identical": identical,
                                 "regression": not identical and old_failures == 0 and new_failures >= 2})
    critical_regressions = [r for r in critical_results if r["regression"]]
    if path_targets is not None:
        for path, names in path_targets.items():
            judged = {k[2] for k in categories if k[1] == path}
            if not names or not set(names).issubset(judged):
                raise ValueError(f"Preregistered targets missing on actual path {path}: {names}")
    elif not targets or not set(targets).issubset(k[2] for k in categories):
        raise ValueError("Name existing target categories before comparison")
    result = []
    rng = random.Random(260526)
    for (route, reply_path, category), cases in sorted(categories.items()):
        changes = [mean(values) for values in cases.values()]
        draws = sorted(
            mean(rng.choices(changes, k=len(changes))) for _ in range(iterations)
        )
        low, high = percentile(draws, 0.025), percentile(draws, 0.975)
        change = mean(changes)
        # Main history mode remains unchanged: do not demand synthetic gains.
        criterion = ('request_identical_reported_only' if (route, reply_path) in unchanged else
                     'non_regression' if reply_path == 'main_llm' else
                     'improvement_with_nonnegative_ci'
                     if category in _targets_for(reply_path, targets, path_targets) else
                     'drop_within_preregistered_aa_noise')
        allowance = 0
        if criterion == 'drop_within_preregistered_aa_noise':
            matching = [r for r in (policy or {}).get('noise_allowances', [])
                        if (r['provider_route'], r['reply_path']) == (route, reply_path)]
            if len(matching) != 1:
                raise ValueError(f'Missing preregistered actual-path A/A allowance: {route}/{reply_path}')
            allowance = matching[0]['max_other_category_drop']
        passed = (True if criterion == 'request_identical_reported_only' else
                  change >= 0 if criterion == 'non_regression' else
                  change > 0 and low >= 0 if criterion == 'improvement_with_nonnegative_ci' else
                  change >= -allowance - 1e-12)
        result.append(
            {
                "category": category, "provider_route": route, "reply_path": reply_path,
                "criterion": criterion,
                "max_allowed_drop": allowance,
                "scenario_count": len(cases),
                **counts[(route, reply_path, category)],
                "change": change,
                "paired_cluster_95_ci": [low, high],
                "passed": passed,
            }
        )
    return {
        "passed": not critical_regressions and all(r["passed"] for r in result),
        "critical_regressions": critical_regressions,
        "critical_scenarios": critical_results,
        "categories": result,
    }


def paired_repetition(before, after, protocol=None, unchanged=frozenset()):
    kwargs = {"protocol": protocol} if protocol else {}
    left, right = repetition_by_path(before['runs'], **kwargs), repetition_by_path(after['runs'], **kwargs)
    key = lambda r: (r['provider_route'], r['reply_path'])
    old = {key(r): r for r in left}
    if set(old) != {key(r) for r in right}:
        raise ValueError('Repetition route population changed')
    gates = []
    for row in right:
        prior = old[key(row)]
        if (row['reply_count'], row['comparable_replies']) != (prior['reply_count'], prior['comparable_replies']):
            raise ValueError('Previous reply coverage changed within repetition comparison')
        # No previous AI turn is an explicit ineligible population, not a zero rate.
        identical = key(row) in unchanged
        passed = (identical or row['comparable_replies'] == 0 or all(
            row[m + '_rate'] <= prior[m + '_rate'] for m in ('exact', 'high_similarity')))
        gates.append({'provider_route': row['provider_route'], 'reply_path': row['reply_path'],
                      'baseline': prior, 'candidate': row, 'request_identical': identical,
                      'passed': passed})
    return {'passed': all(r['passed'] for r in gates), 'routes': gates}


def _state_latency(report, calibration=None):
    rows = latency_observations(report)
    supplement = 'not_available' if calibration else 'not_requested'
    if calibration and calibration.get('latency_observations'):
        protocol = calibration.get('latency_protocol', {})
        if not protocol or any(report.get(k) != v for k, v in protocol.items()):
            # Do not borrow unmatched A/A calls, but a fresh arm with its own
            # 36 qualified calls does not need a latency supplement at all.
            supplement = 'excluded_protocol_mismatch'
        else:
            rows += calibration['latency_observations']
            supplement = 'included_matching_aa'
    unique = {}
    for row in rows:
        identity = tuple(row['identity'])
        if identity in unique and row != unique[identity]:
            raise ValueError('Conflicting duplicate latency observation')
        unique[identity] = row
    return list(unique.values()), supplement


# Request settings already in force before the fingerprint recorded them
# (temporal_awareness/run_incident_eval.py at 03d71d2b). Frozen: a later change
# to JUDGE_REQUEST must not make an old fingerprint look compatible.
LEGACY_JUDGE_REQUEST = {"max_tokens": 1200, "response_format": "json_object", "enable_thinking": False,
                        "payload": "instruction + json({case, reply}); single user message"}


def judge_compatible(calibrated, current):
    """The fingerprint later gained explicit request fields (max_tokens etc.).

    An older fingerprint still matches when every field it has is equal and
    each added field equals the request settings that were already in force.
    """
    if not isinstance(calibrated, dict) or not isinstance(current, dict):
        return False
    if any(current.get(k) != v for k, v in calibrated.items()):
        return False
    return all(k in LEGACY_JUDGE_REQUEST and LEGACY_JUDGE_REQUEST[k] == v
               for k, v in current.items() if k not in calibrated)


def check_binding(report, arm, manifest):
    """Amendment v2: a report is only usable as the arm the manifest registered."""
    reasons = []
    # Unjudged rows are handled by the recorded exclusion rule, not by refusal.
    if report.get("status") not in ("baseline_recorded", "judge_incomplete"):
        reasons.append("status")
    if report.get("measurement_qualified") is not True:
        reasons.append("measurement_qualified")
    if report.get("case_count") != manifest["cases"]:
        reasons.append("case_count")
    if len(report.get("runs", [])) != manifest["expected_calls_per_arm"]:
        reasons.append("run_count")
    if report.get("layout_variant") != arm:
        reasons.append("layout_variant")
    registered = manifest["prompt_snapshot_sha256"].get(arm)
    if not registered or report.get("prompt_snapshot_sha256") != registered:
        reasons.append("prompt_snapshot_sha256")
    if reasons:
        raise ValueError(f"Report does not match registered arm {arm}: {reasons}")


ROOT = Path(__file__).resolve().parents[2]


def load_bound_manifest(amendment, path):
    """The manifest an amendment names, read from that exact file, with its hash."""
    named = (amendment or {}).get("applies_to_manifest")
    if not named:
        raise ValueError("Amendment does not name its replay manifest")
    if path is None or Path(path).resolve() != (ROOT / named).resolve():
        raise ValueError(f"Amendment applies to {named}, not {path}")
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def transport_gate(policy, before, after):
    """Re-run replacements are allowed only as pre-registered, and bounded."""
    rule = (policy.get("amendment") or {}).get("transport_failure_rule")
    counts = [len(r.get("replaced_replays") or []) for r in (before, after)]
    totals = [len({(row["case"], row["sample"]) for row in r["runs"]}) for r in (before, after)]
    result = {"replaced": counts, "replays": totals, "rule": rule}
    if rule is None:
        result["passed"] = not any(counts)
        return result
    result["passed"] = (all(c <= rule["max_replaced_fraction"] * t for c, t in zip(counts, totals))
                        and counts[1] - counts[0] <= rule["max_excess_fraction"] * totals[1])
    return result


def _effective_policy(calibration, amendment, *reports):
    base = build_gate_policy(calibration)
    if any(report.get('gate_policy') != base for report in reports):
        raise ValueError('Missing or changed preregistered gate_policy')
    return apply_amendment(base, amendment)


def _latency_gate(row, prior, before, after, calibration, policy, key):
    """Returns (reasons, metric, population) for one actual path."""
    if row["route"] == "responses.web_search":
        # User-approved 2026-10-26: production nonstream route, completion latency.
        if row['nonstream_samples'] != row['calls'] or prior['nonstream_samples'] != prior['calls']:
            return ['nonstream_completion_missing'], 'p90_nonstream_complete_seconds', None
        if row['p90_nonstream_complete_seconds'] > prior['p90_nonstream_complete_seconds'] * 1.1:
            return ['p90_nonstream_completion_over_10_percent'], 'p90_nonstream_complete_seconds', None
        return [], 'p90_nonstream_complete_seconds', None
    amended = policy.get("amendment") is not None
    field = 'duration_seconds' if amended and row['reply_path'] in NONSTREAM_REPLY_PATHS else 'ttft_seconds'
    metric = 'p90_complete_seconds' if field == 'duration_seconds' else 'p90_ttft_seconds'
    if row['reply_path'] == 'current_state_shortcircuit':
        baseline_population, aa_status = _state_latency(before, calibration if field == 'ttft_seconds' else None)
        candidate_population, _ = _state_latency(after)
        populations = [[r for r in values if (r['model'], r['provider_route'], r['reply_path']) == key]
                       for values in (baseline_population, candidate_population)]
        valid = [[r[field] for r in values if type(r.get(field)) in (int, float)
                  and math.isfinite(r[field]) and r[field] >= 0] for values in populations]
        reasons = []
        if any(len(values) != len(samples) for values, samples in zip(populations, valid)):
            reasons.append(field.replace('_seconds', '') + '_missing')
        if any(len(samples) < policy['state_latency_minimum_per_arm'] for samples in valid):
            reasons.append('state_latency_samples_below_36_per_arm')
        elif percentile(valid[1], .9) > percentile(valid[0], .9) * 1.1:
            reasons.append(metric + '_over_10_percent')
        population = {'baseline_samples': len(valid[0]), 'candidate_samples': len(valid[1]),
                      'baseline_p90_seconds': percentile(valid[0], .9),
                      'candidate_p90_seconds': percentile(valid[1], .9),
                      'aa_supplement_status': aa_status, 'unit': 'seconds', 'metric': field}
        return reasons, metric, population
    samples = 'complete_samples' if field == 'duration_seconds' else 'ttft_samples'
    if row[samples] != row["calls"] or prior[samples] != prior["calls"]:
        return [field.replace('_seconds', '') + '_missing'], metric, None
    if row[metric] > prior[metric] * 1.1:
        return [metric + '_over_10_percent'], metric, None
    return [], metric, None


def compare(before, after, targets=None, calibration=None, *, amendment=None, manifest=None, arms=None):
    if before.get('state_repeats') != after.get('state_repeats'):
        raise ValueError('Incompatible state oversampling protocol')
    for key in ("cases_sha256", "model", "repeats", "model_options", "judge_protocol"):
        if key not in before or key not in after:
            raise ValueError(f"Missing evaluation protocol: {key}")
        if before[key] != after[key]:
            raise ValueError(f"Incompatible baseline: {key}")
    # A/A allowances are fixed before running; observed failures remain in reports.
    if not calibration or calibration.get('status') != 'measured':
        raise ValueError('A/A calibration required before candidate release gates')
    for field in ('cases_sha256', 'prompt_snapshot_sha256', 'model', 'cache_protocol'):
        if calibration.get('protocol', {}).get(field) != before.get(field):
            raise ValueError(f'A/A calibration mismatch: {field}')
    if not judge_compatible(calibration.get('protocol', {}).get('judge_protocol'), before.get('judge_protocol')):
        raise ValueError('A/A calibration mismatch: judge_protocol')
    for field in ('cache_protocol', 'prefix_protocol'):
        if not before.get(field) or before[field] != after.get(field):
            raise ValueError(f'Incompatible measurement protocol: {field}')
    policy = _effective_policy(calibration, amendment, before, after)
    path_targets = None
    if amendment is not None:
        if manifest is None or not arms or len(arms) != 2:
            raise ValueError('Amended gates require the replay manifest and both arm names')
        for report, arm in zip((before, after), arms):
            check_binding(report, arm, manifest)
        path_targets = policy['amendment']['targets']
        if targets and set(targets) != {c for names in path_targets.values() for c in names}:
            raise ValueError('Targets differ from the preregistered amendment')
    observed = {(c.get('route'), c.get('reply_path')) for r in before['runs'] for c in r['calls']}
    calibrated = {(r['provider_route'], r['reply_path']) for r in policy['noise_allowances']}
    if not observed.issubset(calibrated):
        raise ValueError('Missing preregistered actual-path A/A allowance')
    transport = transport_gate(policy, before, after)
    excluded, exclusion = judge_exclusions(policy, before, after)
    unchanged = (unchanged_paths(before, after)
                 if (policy.get('amendment') or {}).get('unchanged_input_rule') else frozenset())
    if exclusion.get('critical_excluded'):
        quality = {'passed': False, 'critical_regressions': [], 'critical_scenarios': [],
                   'categories': [], 'reason': 'critical_identity_unjudged'}
    else:
        quality = paired_quality(before, after, targets, policy=policy, excluded=excluded,
                                 path_targets=path_targets, unchanged=unchanged)
    repetition = paired_repetition(before, after, policy['repetition'] if amendment else None, unchanged)
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
        latency_metric, population = None, None
        if row["status"] != "qualified" or prior["status"] != "qualified":
            reasons.append("usage_or_price_unknown")
        elif (row["route"], row["reply_path"]) in unchanged:
            # Byte-identical requests: resource differences are provider noise.
            pass
        else:
            # Totals over the identical paired population: extra calls must pay.
            for metric in ("measured_input_tokens", "normalized_cost_cny"):
                if row[metric] > prior[metric] * 1.1:
                    reasons.append(metric + "_over_10_percent")
            if row["reply_path"] == "main_llm" and row["route"] == "chat.completions":
                if row["mean_input_tokens"] > 5441 * 1.1:
                    reasons.append("original_5441_token_ceiling")
                if (
                    row["cache_hit_token_rate"] is None
                    or row["cache_hit_token_rate"] < 0.347 - 0.03
                ):
                    reasons.append("original_34_7_percent_cache_floor")
            if row['reply_path'] == 'main_llm':
                if row['cache_hit_token_rate'] is None or prior['cache_hit_token_rate'] is None:
                    reasons.append('cache_rate_unavailable')
                elif row['cache_hit_token_rate'] < prior['cache_hit_token_rate'] - 0.03:
                    reasons.append('paired_cache_drop_over_3_points')
            latency_reasons, latency_metric, population = _latency_gate(
                row, prior, before, after, calibration, policy, key(row))
            reasons += latency_reasons
        gate = {"route": list(key(row)), "passed": not reasons, "reasons": reasons,
                "latency_metric": latency_metric,
                "request_identical": (row["route"], row["reply_path"]) in unchanged}
        if population:
            gate['latency_population'] = population
        gates.append(gate)
    return {
        "passed": (quality["passed"] and repetition['passed'] and exclusion['passed']
                   and transport['passed'] and all(r["passed"] for r in gates)),
        "quality": quality,
        "transport_failures": transport,
        "repetition": repetition,
        "judge_exclusions": exclusion,
        "request_identical_paths": sorted(list(p) for p in unchanged),
        "gate_policy": policy,
        "aa_noise": [{k:v for k,v in r.items() if k != 'pairs'} for r in calibration['routes']],
        "quality_population": "actual path and category; no unweighted overall quality score",
        "resource_gates": gates,
        "limitations": (amendment or {}).get("latency", {}).get("limitation"),
    }


def select_variant(before, variant_a, variant_b, targets, calibration, *, amendment=None, manifest=None):
    """Default A. B is never chosen just because its aggregate score is higher."""
    kwargs = {"amendment": amendment, "manifest": manifest}
    a = compare(before, variant_a, targets, calibration, arms=("baseline", "A"), **kwargs)
    b = compare(before, variant_b, targets, calibration, arms=("baseline", "B"), **kwargs)
    policy = _effective_policy(calibration, amendment, before, variant_a, variant_b)
    excluded, exclusion = judge_exclusions(policy, before, variant_a, variant_b)
    path_targets = policy['amendment']['targets'] if amendment else None
    between = ({'categories': [], 'critical_regressions': ['critical_identity_unjudged']}
               if exclusion.get('critical_excluded') else
               paired_quality(variant_a, variant_b, targets, policy=policy,
                              excluded=excluded, path_targets=path_targets))
    repetition = paired_repetition(variant_a, variant_b, policy['repetition'] if amendment else None)
    allowances = {(r['provider_route'], r['reply_path']): r['max_other_category_drop'] for r in policy['noise_allowances']}
    categories_nonworse = all(r['change'] >= -allowances[(r['provider_route'], r['reply_path'])] - 1e-12
                              for r in between['categories'])
    improved = any(r['candidate']['comparable_replies'] and any(
        r['candidate'][m + '_rate'] < r['baseline'][m + '_rate'] for m in ('exact', 'high_similarity'))
        for r in repetition['routes'])
    no_critical_a_to_b = not between['critical_regressions']
    choose_b = (b['passed'] and repetition['passed'] and improved and categories_nonworse
                and no_critical_a_to_b and exclusion['passed'])
    selected = 'B' if choose_b else 'A'
    return {'selected': selected, 'selected_passed': b['passed'] if choose_b else a['passed'],
            'baseline_to_a': a, 'baseline_to_b': b,
            'a_to_b_repetition': repetition, 'a_to_b_categories': between['categories'],
            'a_to_b_critical_regressions': between['critical_regressions'],
            'b_repetition_improved': improved, 'b_other_categories_within_aa_noise': categories_nonworse,
            'judge_exclusions': exclusion}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--variant-a", type=Path)
    parser.add_argument("--variant-b", type=Path)
    parser.add_argument("--targets", default="")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidate-arm", default="A")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text()) if path else None
    targets = [t for t in args.targets.split(",") if t]
    amendment = load(args.amendment)
    manifest, manifest_sha256 = (load_bound_manifest(amendment, args.manifest) if amendment
                                 else (load(args.manifest), None))
    if args.variant_a or args.variant_b:
        result = select_variant(load(args.baseline), load(args.variant_a), load(args.variant_b), targets,
                                load(args.calibration), amendment=amendment, manifest=manifest)
        passed = result["selected_passed"]
    else:
        result = compare(load(args.baseline), load(args.candidate), targets, load(args.calibration),
                         amendment=amendment, manifest=manifest,
                         arms=("baseline", args.candidate_arm) if amendment else None)
        passed = result["passed"]
    result["manifest_sha256"] = manifest_sha256
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 2)
