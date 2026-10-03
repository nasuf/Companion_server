"""Check the failure gate against actual deployment ordering, not prod services."""
from pathlib import Path
import os
import subprocess

import yaml


def test_server_archive_failure_blocks_sync_and_production_stop():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    sync = next(s for s in steps if s["name"] == "Sync project to VPS")
    script = sync["run"]
    assert script.index("< scripts/release_archive.py") < script.index("rsync -azvcO")
    assert script.index("set -euo pipefail") < script.index("python3 - server")
    assert "--source /app/companion-server" in script
    assert "--root /app/companion-release-archives/server" in script
    assert '"set -eu;' in script
    deploy = next(s for s in steps if s["name"] == "Deploy server stack on VPS")
    assert steps.index(sync) < steps.index(deploy)
    assert "cat > .env" in deploy["with"]["script"]
    assert 'echo "==> Stopping server before migrations"' in deploy["with"]["script"]
    maintenance = (root / "scripts/deploy_host_maintenance.sh").read_text()
    assert "release_archive.py prune --root /app/companion-release-archives/server" in maintenance


def test_archive_guard_remains_part_of_ci():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    quality = next(s for s in workflow["jobs"]["backend-quality"]["steps"]
                   if s["name"] == "Run backend quality tests")
    assert "tests/test_release_archive.py" in quality["run"]
    assert "tests/test_release_archive_workflow.py" in quality["run"]


def test_failed_remote_archive_never_runs_rsync():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    sync = next(s for s in workflow["jobs"]["deploy"]["steps"] if s["name"] == "Sync project to VPS")
    code = sync["run"].split("# Run the candidate's standalone script", 1)[1].partition("\n")[2]
    failure = '''set -euo pipefail
ssh() { return 23; }
rsync() { printf 'PRODUCTION_SYNC_REACHED'; }
'''
    result = subprocess.run(["bash"], input=failure+code, cwd=root,
        env={"PATH": os.environ["PATH"], "VPS_USER": "synthetic", "VPS_HOST": "localhost",
             "VPS_PORT": "22", "RELEASE_SHA": "synthetic"},
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 23
    assert "PRODUCTION_SYNC_REACHED" not in result.stdout
