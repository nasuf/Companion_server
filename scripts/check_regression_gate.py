"""Fail closed on failed, stale, incomplete or under-covered regression evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_regression_gate import ROOT, evidence_hashes


def ratio(covered: int, total: int) -> float | None:
    return 100 * covered / total if total else None


def metric(summary: dict) -> dict:
    for covered, total in (("covered_lines", "num_statements"), ("covered_branches", "num_branches")):
        if (type(summary[covered]) is not int or type(summary[total]) is not int
                or not 0 <= summary[covered] <= summary[total]):
            raise ValueError("Invalid coverage counts")
    return {"lines": ratio(summary["covered_lines"], summary["num_statements"]),
            "branches": ratio(summary["covered_branches"], summary["num_branches"]),
            "line_denominator": summary["num_statements"], "branch_denominator": summary["num_branches"]}


def changed_lines(diff: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif path and line.startswith("@@ "):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if not match:
                raise ValueError("Unrecognized diff hunk")
            start, count = int(match[1]), int(match[2] or 1)
            result.setdefault(path, set()).update(range(start, start + count))
    return result


def assess(run: dict, coverage: dict, junit: ET.Element, policy: dict, *,
           current_sha: str, current_hashes: dict, inventory: list[str], diff: str) -> dict:
    errors: list[str] = []
    if run["commit"] != current_sha:
        errors.append("Report commit differs from checkout")
    if run["hashes_before"] != run["hashes_after"] or run["hashes_after"] != current_hashes:
        errors.append("Sources/tests/policy changed during or after measurement")
    if run["pytest_exit_code"] != 0 or run["network_violations"]:
        errors.append("Pytest failed or attempted a non-loopback connection")
    if run["test_files"] != inventory:
        errors.append("Test file inventory differs from checkout")
    collected = run["collected_nodeids"]
    if not collected or len(collected) != len(set(collected)):
        errors.append("Missing/duplicate collected tests")
    collected_files = {node.split("::")[0] for node in collected}
    if collected_files != set(inventory):
        errors.append("Some test files were not collected: " + str(sorted(set(inventory) - collected_files)))
    cases = list(junit.iter("testcase"))
    if len(cases) != len(collected):
        errors.append("JUnit result count differs from collected tests")
    identities = set()
    for node in collected:
        prefix, separator, parameters = node.partition("[")
        file, *parts = prefix.split("::")
        parts[-1] += separator + parameters
        identities.add(".".join([file[:-3].replace("/", "."), *parts[:-1]]) + "::" + parts[-1])
    reported = [case.get("classname", "") + "::" + case.get("name", "") for case in cases]
    if identities != set(reported) or len(reported) != len(set(reported)):
        errors.append("JUnit identities do not match collection")
    failures = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    if failures or any(int(s.get("errors", "0")) or int(s.get("failures", "0")) for s in junit.iter("testsuite")):
        errors.append("JUnit contains failures/errors")
    skips = []
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None:
            continue
        identity = case.attrib["classname"] + "::" + case.attrib["name"]
        message = skipped.get("message", "")
        allowed = next((entry for entry in policy["allowed_skips"]
                        if entry["id"] == identity and entry["message"] == message
                        and run["platform"] in entry["platforms"]), None)
        skips.append({"id": identity, "message": message, "allowed": bool(allowed)})
        if not allowed:
            errors.append("Unexpected skipped/xfail test: " + identity)
    if not coverage["meta"]["branch_coverage"]:
        errors.append("Branch coverage was not measured")
    runtime_sources = {path for path in current_hashes if path.startswith(("app/", "jobs/")) and path.endswith(".py")}
    if not runtime_sources.issubset(coverage["files"]):
        errors.append("Coverage omitted runtime files")
    metrics = {"global": metric(coverage["totals"])}

    def enforce(label, measured, floors):
        for kind in ("lines", "branches"):
            value = measured[kind]
            if value is None or value < floors[kind]:
                errors.append(f"{label} {kind}: {value} below {floors[kind]}%")

    enforce("global", metrics["global"], policy["global"])
    for path, floors in policy["critical"].items():
        if path not in coverage["files"]:
            errors.append("Missing critical file: " + path)
            continue
        metrics[path] = metric(coverage["files"][path]["summary"])
        enforce(path, metrics[path], floors)
    changed_total = changed_hit = branch_total = branch_hit = 0
    for path, lines in changed_lines(diff).items():
        if not path.endswith(".py") or not lines:
            continue
        if path not in coverage["files"]:
            errors.append("Changed runtime file absent from coverage: " + path)
            continue
        file = coverage["files"][path]
        executed, missing = set(file["executed_lines"]), set(file["missing_lines"])
        executable = lines & (executed | missing)
        changed_total += len(executable)
        changed_hit += len(executable & executed)
        hit = {tuple(edge) for edge in file["executed_branches"] if edge[0] in lines}
        missed = {tuple(edge) for edge in file["missing_branches"] if edge[0] in lines}
        branch_total += len(hit | missed)
        branch_hit += len(hit)
    metrics["changed"] = {"lines": ratio(changed_hit, changed_total), "branches": ratio(branch_hit, branch_total),
                          "line_denominator": changed_total, "branch_denominator": branch_total}
    for kind, total in (("lines", changed_total), ("branches", branch_total)):
        if total and metrics["changed"][kind] < policy["changed"][kind]:
            errors.append(f"changed {kind} below {policy['changed'][kind]}%")
    return {"ok": not errors, "commit": current_sha, "errors": errors, "metrics": metrics,
            "tests": {"collected": len(collected), "passed": len(cases) - failures - len(skips),
                      "failed": failures, "skipped": len(skips), "files": len(inventory)}, "skips": skips}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", required=True, help="Existing Git commit used for changed-line coverage")
    args = parser.parse_args()
    report = {"ok": False, "errors": []}
    try:
        run = json.loads((args.output / "run.json").read_text())
        coverage = json.loads((args.output / "coverage.json").read_text())
        junit = ET.parse(args.output / "junit.xml").getroot()
        policy = json.loads((ROOT / "tests/quality/coverage-policy.json").read_text())
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        # Resolve first: a missing/shallow baseline must not silently yield 0 changed lines.
        base = subprocess.check_output(["git", "rev-parse", "--verify", args.base + "^{commit}"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--unified=0", base, "--", "app", "jobs"], cwd=ROOT, text=True)
        report = assess(run, coverage, junit, policy, current_sha=sha, current_hashes=evidence_hashes(),
                        inventory=sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("test_*.py")), diff=diff)
        report.update(base_commit=base, measured_at=run["finished_at"], tools=run["tools"])
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, subprocess.CalledProcessError) as exc:
        report["errors"].append(f"Invalid/incomplete regression evidence: {type(exc).__name__}: {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = [f"Regression gate: {'PASS' if report['ok'] else 'FAIL'}", "", json.dumps(report, indent=2)]
    (args.output / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
