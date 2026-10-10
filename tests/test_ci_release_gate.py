from pathlib import Path

import pytest

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
