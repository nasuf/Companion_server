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
