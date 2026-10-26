"""Merge one arm's chunked replays (interleaved with another arm) into one report.

Interleaving arms in small chunks keeps provider drift out of arm comparisons.
Chunks must share every protocol field; anything else is refused, not averaged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.context_embedding.measurement import quality_by_path, report_routes

SHARED = ("schema_version", "model", "model_options", "judge_protocol", "repeats", "state_repeats",
          "cases_sha256", "prompt_snapshot_sha256", "source_files_sha256", "gate_policy",
          "cache_protocol", "prefix_protocol", "layout_variant")
_RANK = {"baseline_recorded": 0, "judge_incomplete": 1, "measurement_not_qualified": 2}


def failed_replays(report):
    """(case, sample) replays with a product call whose usage is unknown."""
    return sorted({(r["case"], r["sample"]) for r in report["runs"]
                   if any((c.get("measurement") or {}).get("status") == "request_failed" for c in r["calls"])})


def _replace(runs, reruns, first):
    """Swap each failed replay for its one pre-registered re-run, keeping the record."""
    failed, replaced = set(failed_replays({"runs": runs})), []
    for rerun, sample in reruns:
        for field in set(SHARED) - {"repeats", "state_repeats"}:
            if rerun.get(field) != first.get(field):
                raise ValueError(f"Re-run protocol differs: {field}")
        cases = {r["case"] for r in rerun["runs"]}
        if len(cases) != 1 or {r["sample"] for r in rerun["runs"]} != {0}:
            raise ValueError("A re-run holds exactly one replay")
        key = (cases.pop(), sample)
        if key not in failed:
            raise ValueError(f"Re-run for a replay without a failed call: {key}")
        failed.discard(key)
        old = [r for r in runs if (r["case"], r["sample"]) == key]
        replaced.append({"case": key[0], "sample": sample,
                         "failed_calls": [{"turn": r["turn"], "reply_path": c.get("reply_path"),
                                           "error_type": c.get("error_type")}
                                          for r in old for c in r["calls"]
                                          if (c.get("measurement") or {}).get("status") == "request_failed"],
                         "rerun_started_at": rerun["started_at"], "rerun_status": rerun.get("status")})
        runs = [r for r in runs if (r["case"], r["sample"]) != key]
        runs += [{**r, "sample": sample, "rerun_of_failed_replay": True} for r in rerun["runs"]]
    return runs, replaced


def merge(chunks, reruns=()):
    if not chunks:
        raise ValueError("No chunks to merge")
    first = chunks[0]
    for chunk in chunks[1:]:
        for field in SHARED:
            if chunk.get(field) != first.get(field):
                raise ValueError(f"Chunk protocol differs: {field}")
    statuses = [c.get("status") for c in chunks]
    if any(s not in _RANK for s in statuses):
        raise ValueError(f"Incomplete chunk: {statuses}")
    runs = [row for chunk in chunks for row in chunk["runs"]]
    runs, replaced = _replace(runs, reruns, first)
    identities = [(r["case"], r["sample"], r["turn"]) for r in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate identities across chunks")
    routes = report_routes([call for row in runs for call in row["calls"]])
    merged = {field: first.get(field) for field in SHARED}
    merged.update(
        runs=runs,
        case_count=len({r["case"] for r in runs}),
        completed_turns=len(runs),
        started_at=min(c["started_at"] for c in chunks),
        ended_at=max(c["ended_at"] for c in chunks),
        routes=routes,
        quality_by_actual_path=quality_by_path(runs),
        measurement_qualified=all(r["status"] == "qualified" for r in routes),
        unjudged_turns=sum(type(r.get("judge", {}).get("pass")) is not bool for r in runs),
        status=max(statuses, key=lambda status: _RANK[status]),
        chunks=[{"started_at": c["started_at"], "ended_at": c["ended_at"],
                 "cases": sorted({r["case"] for r in c["runs"]}), "status": c["status"]} for c in chunks],
        replaced_replays=replaced,
        replay_count=len({(r["case"], r["sample"]) for r in runs}),
        limitations=first.get("limitations", []) + ["merged from interleaved chunks; see chunks"],
    )
    if not merged["measurement_qualified"]:
        merged["status"] = "measurement_not_qualified"
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun", action="append", default=[], metavar="SAMPLE=PATH",
                        help="pre-registered re-run of a failed replay, mapped to its original sample")
    parser.add_argument("chunks", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing merged evidence")
    reruns = [(json.loads(Path(path).read_text()), int(sample))
              for sample, path in (item.split("=", 1) for item in args.rerun)]
    report = merge([json.loads(p.read_text()) for p in args.chunks], reruns)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "case_count", "completed_turns", "unjudged_turns")}))
