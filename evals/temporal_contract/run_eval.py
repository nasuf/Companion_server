"""Offline contract and timing comparison. No models, DB or production traffic."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

DATA = Path(__file__).parent


def load_cases():
    data = (DATA / "cases.jsonl").read_bytes()
    manifest = json.loads((DATA / "manifest.json").read_text())
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Frozen dataset hash mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != manifest["case_count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Frozen dataset inventory mismatch")
    return rows, manifest


def evaluate(case, resolve):
    query = resolve(case["message"], anchor=case["anchor"],
                    anchor_source="persisted_message", timezone_name=case["timezone"])
    actual = query.diagnostics()
    projection = {"status": actual["status"], "reason": actual["reason"],
                  "operation": actual.get("operation", {}).get("type")}
    if query.expressions:
        first = query.expressions[0]
        projection.update(start_date=first.start.date().isoformat(),
                          end_date=first.end.date().isoformat(), start_hour=first.start.hour)
    errors = {k: {"expected": v, "actual": projection.get(k)}
              for k, v in case["expected"].items() if projection.get(k) != v}
    return {"id": case["id"], "family": case["family"], "ok": not errors,
            "date_contract_ok": not (errors.keys() - {"operation"}), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DATA.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be 1..100")
    root = args.source_root.resolve()
    if (root / ".env").exists():
        parser.error("Use an isolated checkout without .env")
    if args.output.exists():
        parser.error("Use a new output path; preserve previous results")
    cases, manifest = load_cases()
    sys.path.insert(0, str(root))
    from evals.memory_baseline.safety import configure_isolation, loopback_network_fence
    configure_isolation("postgresql://test:test@127.0.0.1:1/companion_memory_eval_temporal",
                        "redis://127.0.0.1:1/15")
    with loopback_network_fence() as violations:
        from app.services.temporal_query import resolve_temporal_query
        results = [evaluate(c, resolve_temporal_query) for c in cases]
        durations = []
        for _ in range(args.repeats):
            for c in cases:
                started = time.perf_counter_ns()
                resolve_temporal_query(c["message"], anchor=c["anchor"],
                    anchor_source="persisted_message", timezone_name=c["timezone"]).diagnostics()
                durations.append((time.perf_counter_ns() - started) / 1_000_000)
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [root / "app/services/temporal_query.py",
                              root / "app/services/temporal_operation.py",
                              root / "app/services/schedule_domain/time_parser.py"] if p.exists()}
    totals = Counter(r["family"] for r in results)
    passed = Counter(r["family"] for r in results if r["ok"])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    report = {"dataset": manifest, "source_root": str(root), "source_hashes": source_hashes,
              "commit": revision.stdout.strip() if revision.returncode == 0 else None,
              "ok": all(r["ok"] for r in results) and not violations,
              "passed": sum(r["ok"] for r in results), "total": len(results),
              "date_contract_passed": sum(r["date_contract_ok"] for r in results),
              "groups": {g: {"passed": passed[g], "total": n} for g, n in totals.items()},
              "runs": results, "network_violations": violations,
              "timing": {"unit": "ms", "samples": len(durations),
                         "p50": statistics.median(durations),
                         "p95": sorted(durations)[int(.95 * (len(durations)-1))],
                         "max": max(durations), "platform": platform.platform(),
                         "python": platform.python_version(), "scope": "warm pure query parsing plus diagnostics"},
              "limitations": ["48 grammar families with five boundary anchors; not independent model holdout.",
                              "Does not measure full-chat P95, DB retrieval or model response accuracy."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("ok", "passed", "total", "timing")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
