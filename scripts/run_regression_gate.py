"""All collected tests, branch coverage and explicit skip accounting.

Requires disposable loopback resources. Never reads the project's .env or
inherits provider credentials. Run with the constrained dev dependencies.

CI splits the files across --shards jobs, each with its own disposable
databases, then --merge combines their evidence; check_regression_gate.py
judges the merged result exactly as it judges a single full run.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DURATIONS = ROOT / "tests/quality/test-durations.json"
SHARD_KEYS = ("commit", "platform", "tools", "test_files", "hashes_before", "hashes_after")


def inventory(root: Path = ROOT) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in (root / "tests").rglob("test_*.py"))


def shard_files(files: list[str], seconds: dict[str, float], shard: int, shards: int) -> list[str]:
    """Longest file first onto the least-loaded shard, so every shard computes the
    same partition. Untimed (new) files count as the median. Timings only affect
    balance: the merge still requires the shards to cover every file exactly once."""
    if not 0 <= shard < shards:
        raise ValueError("--shard must be in [0, --shards)")
    known = sorted(seconds[f] for f in files if f in seconds)
    default = known[len(known) // 2] if known else 1.0
    loads, owned = [0.0] * shards, [[] for _ in range(shards)]
    for f in sorted(files, key=lambda f: (-seconds.get(f, default), f)):
        i = min(range(shards), key=lambda i: (loads[i], i))
        loads[i] += seconds.get(f, default)
        owned[i].append(f)
    return sorted(owned[shard])


def file_seconds(junit: ET.Element) -> dict[str, float]:
    totals: dict[str, float] = {}
    for case in junit.iter("testcase"):
        module = case.get("classname", "").split(".")
        path = "/".join(module[:2]) + ".py"  # tests.test_x[.Class] -> tests/test_x.py
        totals[path] = totals.get(path, 0.0) + float(case.get("time") or 0)
    return totals


def update_durations(junit_path: Path) -> None:
    seconds = file_seconds(ET.parse(junit_path).getroot())
    current = set(inventory())
    DURATIONS.write_text(json.dumps({
        "schema_version": 1,
        "note": "Per-file seconds from a full CI run; only balances shards. Refresh: "
                "python scripts/run_regression_gate.py --update-durations <junit.xml>",
        "seconds": {k: round(v, 1) for k, v in sorted(seconds.items()) if k in current},
    }, indent=1, ensure_ascii=False) + "\n")


def merge(shard_root: Path, output: Path) -> int:
    """Combine shard evidence; any missing, duplicated or inconsistent shard fails."""
    # Each downloaded artifact keeps its own subdirectory holding shard-<n>/.
    runs = [(d, json.loads((d / "run.json").read_text()))
            for d in sorted({p.parent for p in shard_root.glob("**/shard-*/run.json")})]
    if not runs:
        raise ValueError("No shard evidence found")
    first = runs[0][1]
    if (sorted(r["shard"] for _, r in runs) != list(range(first["shards"]))
            or any(r["shards"] != first["shards"] for _, r in runs)):
        raise ValueError("Shard evidence is missing or duplicated")
    for key in SHARD_KEYS:
        if any(r[key] != first[key] for _, r in runs):
            raise ValueError(f"Shards disagree on {key}")
    if sorted(f for _, r in runs for f in r["shard_files"]) != first["test_files"]:
        raise ValueError("Shards do not cover every test file exactly once")
    suites = ET.Element("testsuites")
    for d, _ in runs:
        root = ET.parse(d / "junit.xml").getroot()
        suites.extend([root] if root.tag == "testsuite" else list(root))
    ET.ElementTree(suites).write(output / "junit.xml", encoding="utf-8", xml_declaration=True)
    import coverage
    cov = coverage.Coverage(data_file=str(output / "coverage.data"))
    cov.combine([str(d / "coverage.data") for d, _ in runs], strict=True, keep=True)
    cov.save()
    cov.json_report(outfile=str(output / "coverage.json"))
    cov.html_report(directory=str(output / "coverage-html"))
    report = {key: first[key] for key in ("schema_version", *SHARD_KEYS)}
    report.update(
        shards=first["shards"],
        pytest_exit_code=max(r["pytest_exit_code"] for _, r in runs),
        network_violations=[v for _, r in runs for v in r["network_violations"]],
        started_at=min(r["started_at"] for _, r in runs),
        finished_at=max(r["finished_at"] for _, r in runs),
        collected_nodeids=[n for _, r in runs for n in r["collected_nodeids"]],
    )
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["pytest_exit_code"] or bool(report["network_violations"])


def evidence_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = [p for folder in ("app", "jobs", "tests", "scripts", "prisma", "evals")
             for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json", ".jsonl", ".sql", ".prisma", ".sh"}]
    paths += list((root / ".github/workflows").glob("*.yml"))
    paths += [root / name for name in ("pyproject.toml", "runtime-constraints.txt")]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


class CollectionEvidence:
    def __init__(self):
        self.nodeids: list[str] = []

    def pytest_collection_modifyitems(self, items):
        self.nodeids = [item.nodeid for item in items]


def resource_url(name: str, scheme: str, databases: set[str]) -> str:
    value = os.environ.get(name, "")
    parsed = urlsplit(value)
    if (parsed.scheme != scheme or parsed.hostname not in {"127.0.0.1", "localhost"}
            or not parsed.port or parsed.path not in databases or parsed.query or parsed.fragment):
        raise ValueError(f"{name} must explicitly identify a disposable loopback resource")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", type=Path, help="Directory of shard outputs to combine into --output")
    parser.add_argument("--update-durations", type=Path, metavar="JUNIT")
    args = parser.parse_args()
    if args.update_durations:
        update_durations(args.update_durations.resolve())
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; stale evidence must never be reused")
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if args.merge:
        return merge(args.merge.resolve(), output)
    files = inventory()
    seconds = json.loads(DURATIONS.read_text())["seconds"] if DURATIONS.is_file() else {}
    selected = shard_files(files, seconds, args.shard, args.shards)
    if not selected:
        raise ValueError("Empty shard; use fewer shards")
    sys.path.insert(0, str(ROOT))
    # Validate before importing any application or pytest modules.
    if (ROOT / ".env").exists():
        raise ValueError("Run in a clean test checkout without .env; child processes must not reload local credentials")
    database = resource_url("PROACTIVE_E2E_DATABASE_URL", "postgresql", {"/companion_proactive_e2e"})
    memory = resource_url("MEMORY_EVAL_TEST_DATABASE_URL", "postgresql", {"/companion_memory_eval_ci", "/companion_memory_eval_gate"})
    resource_url("ERASURE_E2E_DATABASE_URL", "postgresql", {"/companion_erasure_e2e"})
    resource_url("ERASURE_E2E_REDIS_URL", "redis", {"/12"})
    for name in ("G03_TEST_DATABASE_URL", "TTS_TEST_DATABASE_URL"):
        resource_url(name, "postgresql", {"/postgres"})
    for name in ("RUNTIME_JOB_TEST_REDIS_URL", "PROACTIVE_E2E_REDIS_URL", "MEMORY_EVAL_TEST_REDIS_URL"):
        resource_url(name, "redis", {"/13", "/14", "/15"})
    names = ("PROACTIVE_E2E_DATABASE_URL", "MEMORY_EVAL_TEST_DATABASE_URL", "G03_TEST_DATABASE_URL",
             "TTS_TEST_DATABASE_URL", "RUNTIME_JOB_TEST_REDIS_URL", "PROACTIVE_E2E_REDIS_URL", "MEMORY_EVAL_TEST_REDIS_URL",
             "ERASURE_E2E_DATABASE_URL", "ERASURE_E2E_REDIS_URL")
    resources = {key: os.environ[key] for key in names}
    started = datetime.now(UTC).isoformat()
    before = evidence_hashes()
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ["BROWSER_ARTIFACT_DIR"] = str(output / "browser")
    # Not a dotfile: artifact upload skips hidden files, and the merge needs it.
    os.environ["COVERAGE_FILE"] = str(output / "coverage.data")
    from pydantic_settings.sources import DotEnvSettingsSource
    read_dotenv = DotEnvSettingsSource._read_env_files
    from evals.memory_baseline.safety import configure_isolation, loopback_network_fence
    configure_isolation(memory, resources["MEMORY_EVAL_TEST_REDIS_URL"])
    os.environ.update(resources)
    os.environ.update(DATABASE_URL=database, DIRECT_DATABASE_URL=database, MIGRATION_DATABASE_URL=database)
    # Tests deliberately construct Settings(_env_file=tmp_path) to exercise
    # rollout persistence. Preserve those explicit fixtures, never the repo .env.
    def read_fixture_dotenv(source):
        paths = source.env_file if isinstance(source.env_file, (list, tuple)) else [source.env_file]
        if any(p is not None and Path(p).resolve() == ROOT / ".env" for p in paths):
            return {}
        return read_dotenv(source)
    DotEnvSettingsSource._read_env_files = read_fixture_dotenv
    import pytest
    collection = CollectionEvidence()
    with loopback_network_fence() as violations:
        # Whole-suite runs keep the plain "tests" target; a shard names its files.
        targets = ["tests"] if args.shards == 1 else selected
        reports = [f"--cov-report=json:{output / 'coverage.json'}",
                   f"--cov-report=html:{output / 'coverage-html'}"] if args.shards == 1 else []
        result = pytest.main([
            *targets, "-q", "--tb=short", "--strict-markers", "-o", "addopts=",
            f"--junitxml={output / 'junit.xml'}",
            "--cov=app", "--cov=jobs", "--cov-branch", "--cov-report=", *reports,
        ], plugins=[collection])
    report = {"schema_version": 1, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "pytest_exit_code": int(result), "network_violations": violations,
              "started_at": started, "finished_at": datetime.now(UTC).isoformat(), "platform": sys.platform,
              "tools": {name: version(name) for name in ("pytest", "pytest-cov", "coverage", "prisma", "redis", "playwright")},
              "test_files": files, "collected_nodeids": collection.nodeids,
              "shard": args.shard, "shards": args.shards, "shard_files": selected,
              "hashes_before": before, "hashes_after": evidence_hashes()}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(result) or bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
