"""Stage metrics and paired, scenario-clustered bootstrap; missing is not PASS."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean

from .dataset import SEED, fingerprint


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def grade(case: dict, row: dict) -> dict:
    """None means a stage was not executed, rather than an empty/successful stage."""
    stages = ("stored", "candidates", "selected", "used")
    expected, forbidden = set(case["expected"]), set(case["forbidden"])
    out = {}
    for stage in stages:
        ids = row.get(stage)
        if ids is None:
            out[stage] = {"status": "not_run", "precision": None, "recall": None}
            continue
        if (
            not isinstance(ids, list)
            or not all(isinstance(k, str) for k in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("Stage identities must be unique side-qualified strings")
        if not set(ids) <= {m["key"] for m in case["memories"]}:
            raise ValueError("Unknown memory identity")
        present = set(ids)
        out[stage] = {
            "status": "measured",
            "precision": (
                len(present & expected) / len(present)
                if present
                else (1.0 if not expected else 0.0)
            ),
            "recall": len(present & expected) / len(expected) if expected else 1.0,
            "missing": sorted(expected - present),
            "forbidden": sorted(present & forbidden),
        }
    stored, candidates, selected, used = (row.get(k) for k in stages)
    # Attribution follows the first observed loss; do not invent an extraction
    # result for a fixture-seeded retrieval experiment.
    losses = {
        "recording_miss": None,
        "retrieval_miss": None,
        "selection_miss": None,
        "answer_unused": None,
    }
    for name, upstream, downstream in [
        ("recording_miss", list(expected), stored),
        (
            "retrieval_miss",
            row.get("fixture_ids") if row.get("fixture_seeded") else stored,
            candidates,
        ),
        ("selection_miss", candidates, selected),
        ("answer_unused", selected, used),
    ]:
        if upstream is not None and downstream is not None:
            losses[name] = sorted((expected & set(upstream)) - set(downstream))
    return {
        "stages": out,
        "losses": losses,
        "stale_misuse": row.get("stale_misuse"),
        "persona_drift": row.get("persona_drift"),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def paired_interval(
    differences: dict[str, list[float]], *, seed=SEED, draws=2000
) -> dict:
    """Resample families, each containing every repeated paired model sample.

    Sampling repeated calls as independent observations artificially narrows CI.
    First average within each family, then bootstrap these independent clusters.
    """
    if not differences or any(
        not v or not all(finite(x) for x in v) for v in differences.values()
    ):
        raise ValueError("Complete finite paired clusters required")
    values = [mean(differences[k]) for k in sorted(differences)]
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {
        "delta": mean(values),
        "lower": percentile(samples, 0.025),
        "upper": percentile(samples, 0.975),
        "clusters": len(values),
        "paired_observations": sum(map(len, differences.values())),
        "draws": draws,
        "seed": seed,
    }


def summarize(
    cases: list[dict],
    rows: list[dict],
    *,
    samples: int = 1,
    arms=("baseline", "candidate"),
) -> dict:
    if (
        type(samples) is not int
        or samples < 1
        or len(arms) != 2
        or len(set(arms)) != 2
        or not cases
    ):
        raise ValueError(
            "Positive sample count, two distinct arms and nonempty cases required"
        )
    by_id = {c["id"]: c for c in cases}
    expected = {
        (c["id"], sample, arm, cache)
        for c in cases
        for sample in range(samples)
        for arm in arms
        for cache in ("cold", "warm")
    }
    indexed, errors = {}, []
    for row in rows:
        if type(row.get("sample")) is not int:
            errors.append("invalid_sample_identity")
            continue
        key = row.get("case"), row.get("sample"), row.get("arm"), row.get("cache")
        if key not in expected or key in indexed:
            errors.append("unknown_or_duplicate_observation")
            continue
        indexed[key] = row
        if row.get("error") or row.get("network_violations"):
            errors.append("failed_observation")
        if not finite(row.get("latency_ms")) or row["latency_ms"] < 0:
            errors.append("missing_or_invalid_latency")
        if row.get("selected") is None or row.get("candidates") is None:
            errors.append("missing_retrieval_stage")
    if set(indexed) != expected:
        errors.append("incomplete_observations")
    grouped = defaultdict(list)
    hard_failures = []
    for key, row in indexed.items():
        case = by_id[key[0]]
        g = grade(case, row)
        grouped[(case["group"], row["arm"], row["cache"])].append((row, g))
        if case["hard_invariant"] and g["stages"]["selected"].get("forbidden"):
            hard_failures.append(
                {"case": case["id"], "arm": row["arm"], "cache": row["cache"]}
            )
    segments = []
    for (group, arm, cache), values in sorted(grouped.items()):
        precisions = [
            g["stages"]["selected"]["precision"]
            for _, g in values
            if g["stages"]["selected"]["precision"] is not None
        ]
        recalls = [
            g["stages"]["selected"]["recall"]
            for _, g in values
            if g["stages"]["selected"]["recall"] is not None
        ]
        segments.append(
            {
                "group": group,
                "arm": arm,
                "cache": cache,
                "observations": len(values),
                "precision": mean(precisions) if precisions else None,
                "recall": mean(recalls) if recalls else None,
                "latency_p50_ms": percentile(
                    [r["latency_ms"] for r, _ in values if finite(r.get("latency_ms"))],
                    0.5,
                ),
                "latency_p95_ms": percentile(
                    [r["latency_ms"] for r, _ in values if finite(r.get("latency_ms"))],
                    0.95,
                ),
                "db_calls_mean": (
                    mean(r["db_calls"] for r, _ in values)
                    if all(finite(r.get("db_calls")) for r, _ in values)
                    else None
                ),
                "recording": (
                    "not_run"
                    if all(r.get("stored") is None for r, _ in values)
                    else "measured"
                ),
                "answer": (
                    "not_run"
                    if all(r.get("used") is None for r, _ in values)
                    else "measured"
                ),
            }
        )
    paired = defaultdict(list)
    for case in cases:
        for sample in range(samples):
            for cache in ("cold", "warm"):
                a = indexed.get((case["id"], sample, arms[0], cache))
                b = indexed.get((case["id"], sample, arms[1], cache))
                if a is None or b is None:
                    continue
                if a.get("conditions") != b.get("conditions") or not a.get(
                    "conditions"
                ):
                    errors.append("unmatched_conditions")
                    continue
                if a.get("selected") is None or b.get("selected") is None:
                    continue
                ag, bg = grade(case, a), grade(case, b)
                paired[case["family"]].append(
                    bg["stages"]["selected"]["recall"]
                    - ag["stages"]["selected"]["recall"]
                )
    return {
        "schema_version": 1,
        "complete": not errors,
        "errors": sorted(set(errors)),
        "instrumentation_passed": not errors,
        "algorithm_quality_passed": None,
        "hard_failures": hard_failures,
        "segments": segments,
        "recall_interval": paired_interval(paired) if paired and not errors else None,
        "model_quality": "not_run",
        "dataset_sha256": fingerprint(cases),
        "limitations": [
            "A baseline records existing quality failures; it does not certify a new algorithm.",
            "Retrieval fixture seeding does not measure extraction, final prompt use or reply quality.",
        ],
    }
