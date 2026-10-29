"""Complete matched samples only; an absent baseline never means pass."""

from math import ceil
from app.config import settings


def compare(current, baseline, noise_rate=None):
    noise_rate = settings.quality_noise_rate if noise_rate is None else noise_rate

    def keyed(rows):
        return {(r["case_id"], r["repetition"]): r for r in rows}

    c, b = keyed(current), keyed(baseline)
    if not c or any(r["pass"] is None for r in c.values()):
        return dict(status="incomplete", reason="missing_or_unjudged_samples")
    if not b:
        return dict(status="baseline_recorded", reason="no_matching_baseline")
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
    return dict(
        status="failed" if critical_change > 0 or other_change > noise else "passed",
        critical_failure_delta=critical_change,
        other_failure_delta=other_change,
        noise_cases=noise,
        critical_n=len(critical),
        other_n=len(other),
        noise_rate=noise_rate,
    )
