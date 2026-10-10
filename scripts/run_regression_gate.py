"""All collected tests, branch coverage and explicit skip accounting.

Requires disposable loopback resources. Never reads the project's .env or
inherits provider credentials. Run with the constrained dev dependencies.
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


ROOT = Path(__file__).resolve().parents[1]


def evidence_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = [p for folder in ("app", "jobs", "tests", "scripts", "prisma")
             for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json", ".sql", ".prisma"}]
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
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; stale evidence must never be reused")
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Validate before importing any application or pytest modules.
    if (ROOT / ".env").exists():
        raise ValueError("Run in a clean test checkout without .env; child processes must not reload local credentials")
    database = resource_url("PROACTIVE_E2E_DATABASE_URL", "postgresql", {"/companion_proactive_e2e"})
    memory = resource_url("MEMORY_EVAL_TEST_DATABASE_URL", "postgresql", {"/companion_memory_eval_ci", "/companion_memory_eval_gate"})
    for name in ("G03_TEST_DATABASE_URL", "TTS_TEST_DATABASE_URL"):
        resource_url(name, "postgresql", {"/postgres"})
    for name in ("RUNTIME_JOB_TEST_REDIS_URL", "PROACTIVE_E2E_REDIS_URL", "MEMORY_EVAL_TEST_REDIS_URL"):
        resource_url(name, "redis", {"/13", "/14", "/15"})
    names = ("PROACTIVE_E2E_DATABASE_URL", "MEMORY_EVAL_TEST_DATABASE_URL", "G03_TEST_DATABASE_URL",
             "TTS_TEST_DATABASE_URL", "RUNTIME_JOB_TEST_REDIS_URL", "PROACTIVE_E2E_REDIS_URL", "MEMORY_EVAL_TEST_REDIS_URL")
    resources = {key: os.environ[key] for key in names}
    started = datetime.now(UTC).isoformat()
    before = evidence_hashes()
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ["BROWSER_ARTIFACT_DIR"] = str(output / "browser")
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
        result = pytest.main([
            "tests", "-q", "--tb=short", "--strict-markers", "-o", "addopts=",
            f"--junitxml={output / 'junit.xml'}",
            "--cov=app", "--cov=jobs", "--cov-branch", "--cov-report=",
            f"--cov-report=json:{output / 'coverage.json'}",
            f"--cov-report=html:{output / 'coverage-html'}",
        ], plugins=[collection])
    inventory = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("test_*.py"))
    report = {"schema_version": 1, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "pytest_exit_code": int(result), "network_violations": violations,
              "started_at": started, "finished_at": datetime.now(UTC).isoformat(), "platform": sys.platform,
              "tools": {name: version(name) for name in ("pytest", "pytest-cov", "coverage", "prisma", "redis", "playwright")},
              "test_files": inventory, "collected_nodeids": collection.nodeids,
              "hashes_before": before, "hashes_after": evidence_hashes()}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(result) or bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
