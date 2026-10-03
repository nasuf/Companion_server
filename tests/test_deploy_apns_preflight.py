"""Execute the deployment preflight shell with a fake Docker boundary."""
from pathlib import Path
import os
import re
import subprocess
import textwrap

import pytest
import yaml


@pytest.mark.parametrize("exists,owner,check_exit,expected_success", [
    ("0", "true", "0", True),
    ("1", "true", "0", True),
    ("1", "true", "23", False),
    ("1", "false", "0", False),
])
def test_image_failure_blocks_production_stop_and_retention_is_owned(
        tmp_path, exists, owner, check_exit, expected_success):
    root = Path(__file__).resolve().parents[1]
    block = (root / "scripts/deploy_apns_preflight.sh").read_text()
    log = tmp_path / "commands"
    # No real Docker, filesystem cleanup, production settings or SSH commands.
    fake = r'''
set -euo pipefail
docker() {
  printf '%s\n' "$*" >> "$COMMAND_LOG"
  case "$1" in
    inspect)
      case "$*" in
        *rollback-retention*) printf '%s\n' "$RETENTION_OWNER" ;;
        *) printf '%s\n' 'sha256:previous' ;;
      esac ;;
    container) test "$RETENTION_EXISTS" = 1 ;;
    image) printf '%s\n' 'sha256:candidate' ;;
    run) return "$CHECK_EXIT" ;;
    rm|create|compose) return 0 ;;
    *) return 99 ;;
  esac
}
DOCKER=docker
'''
    environment = {"PATH": os.environ["PATH"], "COMMAND_LOG": str(log),
                   "RETENTION_EXISTS": exists, "RETENTION_OWNER": owner, "CHECK_EXIT": check_exit}
    result = subprocess.run(["bash"], input=fake + textwrap.dedent(block)
        + '\nprintf "PRODUCTION_STOP_ALLOWED\\n"\n', env=environment,
        text=True, capture_output=True, timeout=10)
    assert (result.returncode == 0) is expected_success
    assert ("PRODUCTION_STOP_ALLOWED" in result.stdout) is expected_success
    commands = log.read_text()
    if owner == "false":
        assert "\nrm " not in commands and "\ncreate " not in commands
        assert "compose " not in commands
    else:
        assert "--entrypoint /bin/true sha256:previous" in commands
        assert "--network none --read-only" in commands
        assert "--entrypoint python sha256:candidate -m scripts.check_apns_log_redaction" in commands


def test_workflow_preflight_precedes_production_stop_and_fits_expression_limit():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"]
                if s["name"] == "Deploy server stack on VPS")
    script = step["with"]["script"]
    assert script.index("source scripts/deploy_apns_preflight.sh") < script.index(
        'echo "==> Stopping server before migrations"')
    assert "CANDIDATE_IMAGE" in script
    # GitHub expands interpolated strings to format(...) expressions. Count
    # escaping and argument overhead conservatively, with 500 chars of margin
    # below the platform's 21000-char limit (the previous inline block failed).
    expressions = re.findall(r"\$\{\{(.*?)\}\}", script, re.S)
    literal = re.sub(r"\$\{\{.*?\}\}", "{0}", script, flags=re.S)
    escaped = literal.replace("'", "''").replace("{", "{{").replace("}", "}}")
    estimated_length = len(escaped) + sum(map(len, expressions)) + len(expressions) * 6 + 64
    assert estimated_length < 20500
