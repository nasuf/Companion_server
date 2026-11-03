"""The release gate itself must reject missing evidence and false greens."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.check_regression_gate import assess, changed_lines, metric
from scripts.run_regression_gate import resource_url


@pytest.fixture
def evidence():
    summary = {"covered_lines": 2, "num_statements": 2, "covered_branches": 2, "num_branches": 2}
    path = "app/example.py"
    hashes = {path: "hash", "tests/test_example.py": "test-hash"}
    run = {"commit": "sha", "hashes_before": hashes, "hashes_after": hashes,
           "pytest_exit_code": 0, "network_violations": [], "platform": "linux",
           "test_files": ["tests/test_example.py"], "collected_nodeids": ["tests/test_example.py::test_ok"]}
    coverage = {"meta": {"branch_coverage": True}, "totals": summary,
                "files": {path: {"summary": summary, "executed_lines": [1, 2], "missing_lines": [],
                                 "executed_branches": [[1, 2], [1, -1]], "missing_branches": []}}}
    policy = {"global": {"lines": 77, "branches": 64}, "critical": {path: {"lines": 90, "branches": 80}},
              "changed": {"lines": 90, "branches": 80}, "allowed_skips": []}
    junit = ET.fromstring('<testsuites><testsuite tests="1"><testcase classname="tests.test_example" name="test_ok"/></testsuite></testsuites>')
    return {"run": run, "coverage": coverage, "junit": junit, "policy": policy,
            "current_sha": "sha", "current_hashes": hashes, "inventory": run["test_files"],
            "diff": "+++ b/app/example.py\n@@ -0,0 +1,2 @@\n+if x:\n+    y()"}


def test_complete_evidence_passes(evidence):
    result = assess(**evidence)
    assert result["ok"] and result["tests"]["passed"] == 1
    assert result["metrics"]["changed"]["line_denominator"] == 2


def test_parameter_colons_do_not_become_junit_class_names(evidence):
    evidence["run"]["collected_nodeids"] = ["tests/test_example.py::TestURL::test_ok[http://[::1]/]"]
    case = next(evidence["junit"].iter("testcase"))
    case.set("classname", "tests.test_example.TestURL")
    case.set("name", "test_ok[http://[::1]/]")
    assert assess(**evidence)["ok"]


@pytest.mark.parametrize("failure", ["stale_sha", "mutation", "new_test", "no_tests", "duplicate_tests", "failed_run",
                                         "network", "unexpected_skip", "junit_failure", "no_branches", "omitted_file",
                                         "low_lines", "low_branches", "uncovered_change", "missing_result"])
def test_false_green_is_rejected(evidence, failure):
    e = deepcopy(evidence)
    if failure == "stale_sha": e["current_sha"] = "new-sha"
    elif failure == "mutation": e["run"]["hashes_before"] = {"old": "old"}
    elif failure == "new_test": e["inventory"] = [*e["inventory"], "tests/test_new.py"]
    elif failure == "no_tests": e["run"]["collected_nodeids"] = []
    elif failure == "duplicate_tests": e["run"]["collected_nodeids"] *= 2
    elif failure == "failed_run": e["run"]["pytest_exit_code"] = 1
    elif failure == "network": e["run"]["network_violations"] = ["non-loopback"]
    elif failure == "unexpected_skip": ET.SubElement(next(e["junit"].iter("testcase")), "skipped", message="DB missing")
    elif failure == "junit_failure": ET.SubElement(next(e["junit"].iter("testcase")), "failure")
    elif failure == "no_branches": e["coverage"]["meta"]["branch_coverage"] = False
    elif failure == "omitted_file": e["coverage"]["files"] = {}
    elif failure == "low_lines": e["coverage"]["totals"]["covered_lines"] = 1
    elif failure == "low_branches": e["coverage"]["totals"]["covered_branches"] = 1
    elif failure == "missing_result": e["junit"].clear()
    elif failure == "uncovered_change":
        file = e["coverage"]["files"]["app/example.py"]
        file["executed_lines"] = [1]; file["missing_lines"] = [2]
    assert not assess(**e)["ok"]


def test_changed_branch_edges_are_checked(evidence):
    file = evidence["coverage"]["files"]["app/example.py"]
    file["executed_branches"] = [[1, 2]]; file["missing_branches"] = [[1, -1]]
    result = assess(**evidence)
    assert "changed branches below 80%" in result["errors"]


def test_empty_changed_denominator_is_not_reported_as_100(evidence):
    evidence["diff"] = ""
    assert assess(**evidence)["metrics"]["changed"]["lines"] is None


def test_skip_allowance_is_exact_and_platform_specific(evidence):
    case = next(evidence["junit"].iter("testcase"))
    ET.SubElement(case, "skipped", message="Linux /proc only")
    evidence["policy"]["allowed_skips"] = [{"id": "tests.test_example::test_ok", "message": "Linux /proc only", "platforms": ["darwin"]}]
    assert not assess(**evidence)["ok"]
    evidence["run"]["platform"] = "darwin"
    result = assess(**evidence)
    assert result["ok"] and result["tests"]["passed"] == 0 and result["tests"]["skipped"] == 1


def test_diff_parsing_ignores_deleted_lines_and_counts_added_ranges():
    assert changed_lines("+++ b/app/a.py\n@@ -8,2 +9,0 @@\n@@ -20 +20,3 @@\n") == {"app/a.py": {20, 21, 22}}


@pytest.mark.parametrize("value", [float("nan"), -1, 3, True])
def test_invalid_coverage_counts_cannot_pass(value):
    with pytest.raises(ValueError):
        metric({"covered_lines": value, "num_statements": 2, "covered_branches": 1, "num_branches": 2})


@pytest.mark.parametrize("url", ["", "postgresql://host:5432/companion_proactive_e2e",
    "postgresql://127.0.0.1/companion_proactive_e2e", "postgresql://127.0.0.1:5432/prod",
    "postgresql://127.0.0.1:5432/companion_proactive_e2e?host=prod", "redis://127.0.0.1:6379/14"])
def test_resource_preflight_rejects_implicit_or_non_fixture_urls(monkeypatch, url):
    monkeypatch.setenv("PROACTIVE_E2E_DATABASE_URL", url)
    with pytest.raises(ValueError): resource_url("PROACTIVE_E2E_DATABASE_URL", "postgresql", {"/companion_proactive_e2e"})


def test_missing_artifacts_cli_exits_nonzero_and_writes_diagnostics(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/check_regression_gate.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path), "--base", "HEAD"], capture_output=True, text=True)
    assert result.returncode == 1
    assert not json.loads((tmp_path / "gate.json").read_text())["ok"]


def test_shards_partition_every_file_once_and_balance():
    from scripts.run_regression_gate import shard_files
    files = [f"tests/test_{i}.py" for i in range(40)]
    seconds = {f: float(i % 7 + 1) for i, f in enumerate(files[:-3])}  # last 3 untimed
    parts = [shard_files(files, seconds, i, 4) for i in range(4)]
    assert sorted(f for p in parts for f in p) == sorted(files)
    assert parts == [shard_files(files, seconds, i, 4) for i in range(4)]  # deterministic
    loads = [sum(seconds.get(f, 4.0) for f in p) for p in parts]
    assert max(loads) - min(loads) <= 7
    with pytest.raises(ValueError):
        shard_files(files, seconds, 4, 4)


def _shard(root, index, shards, files, nodeids, *, hashes=None, exit_code=0):
    import coverage
    from scripts.run_regression_gate import SHARD_KEYS
    d = root / f"backend-shard-{index}" / f"shard-{index}"
    d.mkdir(parents=True)
    run = {key: "same" for key in SHARD_KEYS}
    run.update(schema_version=1, test_files=["tests/test_a.py", "tests/test_b.py"],
               hashes_after=hashes or {"x": "1"}, hashes_before=hashes or {"x": "1"},
               shard=index, shards=shards, shard_files=files, collected_nodeids=nodeids,
               pytest_exit_code=exit_code, network_violations=[],
               started_at=f"2026-11-03T0{index}:00", finished_at=f"2026-11-03T1{index}:00")
    (d / "run.json").write_text(json.dumps(run))
    cases = "".join(f'<testcase classname="{n.split("::")[0][:-3].replace("/", ".")}" name="{n.split("::")[1]}"/>'
                    for n in nodeids)
    (d / "junit.xml").write_text(f'<testsuites><testsuite tests="{len(nodeids)}">{cases}</testsuite></testsuites>')
    data = coverage.CoverageData(basename=str(d / "coverage.data"))
    data.add_arcs({str(root.parent / "mod.py"): [(-1, 1), (1, 2), (2, 3)] if index else [(-1, 1), (1, 2), (2, -1)]})
    data.write()


def test_merge_combines_shards_into_single_run_evidence(tmp_path, monkeypatch):
    from scripts.run_regression_gate import merge
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x = 1\nif x:\n    y = 2\n")
    shards, out = tmp_path / "shards", tmp_path / "out"
    out.mkdir()
    _shard(shards, 0, 2, ["tests/test_a.py"], ["tests/test_a.py::test_one"])
    _shard(shards, 1, 2, ["tests/test_b.py"], ["tests/test_b.py::test_two"])
    assert merge(shards, out) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["collected_nodeids"] == ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    assert run["started_at"].endswith("00:00") and run["finished_at"].endswith("11:00")
    assert len(list(ET.parse(out / "junit.xml").getroot().iter("testcase"))) == 2
    covered = json.loads((out / "coverage.json").read_text())
    assert covered["meta"]["branch_coverage"] and covered["totals"]["covered_branches"] == 2


@pytest.mark.parametrize("defect", ["missing_shard", "duplicate_shard", "overlap", "hash_mismatch", "failed_shard"])
def test_merge_rejects_incomplete_or_inconsistent_shards(tmp_path, monkeypatch, defect):
    from scripts.run_regression_gate import merge
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x = 1\nif x:\n    y = 2\n")
    shards, out = tmp_path / "shards", tmp_path / "out"
    out.mkdir()
    _shard(shards, 0, 2, ["tests/test_a.py"] + (["tests/test_b.py"] if defect == "overlap" else []),
           ["tests/test_a.py::test_one"])
    if defect != "missing_shard":
        index = 0 if defect == "duplicate_shard" else 1
        _shard(shards / "dup" if defect == "duplicate_shard" else shards, index, 2, ["tests/test_b.py"],
               ["tests/test_b.py::test_two"], hashes={"x": "2"} if defect == "hash_mismatch" else None,
               exit_code=1 if defect == "failed_shard" else 0)
    if defect == "failed_shard":
        assert merge(shards, out) == 1
    else:
        with pytest.raises(ValueError):
            merge(shards, out)
