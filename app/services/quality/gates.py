"""Complete matched samples only; an absent baseline never means pass.

Critical cases run several repetitions, so they are compared per case by
majority vote: a case regresses when the baseline majority passed and the
current majority does not. Identical code re-run night to night flips 5-9 of
125 critical cases (2026-11-14..18), so up to floor(n x critical_noise_rate)
regressed cases are tolerated; a real break usually takes out several cases
of one suite and still exceeds it. Small case sets keep zero tolerance.

A sample marked `unknown` (a confirmed case whose exact source input is gone
or cannot be priced) is listed apart and left out of both sides; more than
quality_max_unknown_rate of them still makes the run incomplete.
"""

from collections import defaultdict
from math import ceil, floor
from app.config import settings


def _majority(rows, keys):
    votes = defaultdict(list)
    for k in keys:
        votes[k[0]].append(bool(rows[k]["pass"]))
    return {case: sum(v) * 2 > len(v) for case, v in votes.items()}


def compare(current, baseline, noise_rate=None):
    noise_rate = settings.quality_noise_rate if noise_rate is None else noise_rate

    def keyed(rows):
        return {(r["case_id"], r["repetition"]): r for r in rows}

    c, b = keyed(current), keyed(baseline)
    unknown = {k for k, r in c.items() if r.get("unknown")}
    unknown_cases = sorted({k[0] for k in unknown})
    if c and len(unknown) > len(c) * settings.quality_max_unknown_rate:
        return dict(status="incomplete", reason="too_many_unknown_samples",
                    unknown_samples=len(unknown), unknown_cases=unknown_cases[:50])
    unknown |= {k for k, r in b.items() if r.get("unknown")}
    c = {k: r for k, r in c.items() if k not in unknown}
    b = {k: r for k, r in b.items() if k not in unknown}
    if not c or any(r["pass"] is None for r in c.values()):
        return dict(status="incomplete", reason="missing_or_unjudged_samples")
    if not b:
        return dict(status="baseline_recorded", reason="no_matching_baseline",
                    unknown_samples=len(unknown), unknown_cases=unknown_cases[:50])
    if set(c) != set(b) or any(r["pass"] is None for r in b.values()):
        return dict(status="incomplete", reason="baseline_case_coverage_mismatch")
    critical = [k for k in c if c[k]["critical"]]
    other = [k for k in c if not c[k]["critical"]]
    critical_change = sum(not c[k]["pass"] for k in critical) - sum(
        not b[k]["pass"] for k in critical
    )
    other_change = sum(not c[k]["pass"] for k in other) - sum(
        not b[k]["pass"] for k in other
    )
    noise = max(settings.quality_noise_min_cases, ceil(len(other) * noise_rate))
    cur, base = _majority(c, critical), _majority(b, critical)
    regressed = sorted(k for k in cur if base[k] and not cur[k])
    improved = sum(1 for k in cur if not base[k] and cur[k])
    critical_noise = floor(len(cur) * settings.quality_critical_noise_rate)
    return dict(
        status="failed" if len(regressed) > critical_noise or other_change > noise else "passed",
        critical_regressed_cases=len(regressed),
        critical_improved_cases=improved,
        critical_noise_cases=critical_noise,
        critical_regressed=regressed[:50],
        critical_failure_delta=critical_change,
        other_failure_delta=other_change,
        noise_cases=noise,
        critical_n=len(critical),
        other_n=len(other),
        noise_rate=noise_rate,
        unknown_samples=len(unknown),
        unknown_cases=unknown_cases[:50],
    )
