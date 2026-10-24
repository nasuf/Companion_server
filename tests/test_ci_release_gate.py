from pathlib import Path
import shlex

import pytest
import yaml

from scripts.require_ci_success import qualified_run


def candidate(**kwargs):
    return {"id": 1, "head_sha": "sha", "head_branch": "main", "event": "push", "run_number": 10,
            "run_attempt": 1, "status": "completed", "conclusion": "success", **kwargs}


def test_exact_success_qualifies():
    assert qualified_run([candidate()], "sha")["id"] == 1


@pytest.mark.parametrize("change", [{"head_sha": "old"}, {"head_branch": "topic"}, {"event": "pull_request"},
    {"status": "in_progress"}, {"conclusion": "failure"}, {"conclusion": "cancelled"}, {"conclusion": "skipped"}])
def test_wrong_or_unfinished_evidence_never_qualifies(change):
    with pytest.raises(ValueError): qualified_run([candidate(**change)], "sha")


def test_failed_rerun_cannot_reuse_an_older_success():
    with pytest.raises(ValueError):
        qualified_run([candidate(), candidate(run_attempt=2, conclusion="failure")], "sha")


def test_manual_and_automatic_deploys_share_guard_before_any_ssh():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/deploy.yml").read_text()
    assert workflow.index("python3 scripts/require_ci_success.py") < workflow.index("uses: appleboy/ssh-action")
    assert "if: github.event_name == 'workflow_dispatch'" not in workflow.split("- name: Require exact successful CI")[1].split("- name:")[0]
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in workflow
    assert "github.event.workflow_run.event == 'push'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow


def test_backend_inventory_is_not_a_file_allowlist():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    assert "python scripts/run_regression_gate.py" in workflow
    assert "python scripts/check_regression_gate.py" in workflow
    assert "pytest \\\n" not in workflow


def test_chat_network_checks_prepare_dependencies_before_execution():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["chat-safety-integration"]["steps"]
    for image in ("redis:7-alpine", "nginx:1.24-alpine"):
        prepared = next(i for i, step in enumerate(steps) if f"docker pull {image}" in step.get("run", ""))
        checks = [i for i, step in enumerate(steps) if "scripts/test_chat_safety_e2e.py" in step.get("run", "")]
        assert checks and all(prepared < i for i in checks)


def test_delayed_network_diagnostics_are_retained_on_failure():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["chat-safety-integration"]["steps"]
    command = next(step["run"] for step in steps if "--delay" in step.get("run", ""))
    args = shlex.split(command)
    output = Path(args[args.index("--output") + 1].replace("$RUNNER_TEMP", "/tmp/runner"))
    upload = next(step for step in steps if step.get("with", {}).get("name") == "chat-safety-e2e")
    artifact_root = Path(upload["with"]["path"].replace("${{ runner.temp }}", "/tmp/runner"))
    assert output.is_relative_to(artifact_root)
    assert upload["if"] == "always()"
