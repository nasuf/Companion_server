"""Fixed graph startup and private rollback environment compatibility."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import Settings
from scripts.chat_graph_deploy_config import apply_rollout

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_startup_environment(monkeypatch):
    for key in ("CHAT_EXECUTOR", "CHAT_GRAPH_CONVERSATION_ALLOWLIST", "CHAT_GRAPH_ALL_CONVERSATIONS", "JWT_SECRET", "WEB_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)


def setup_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("JWT_SECRET=synthetic-secret\nWEB_CONCURRENCY=2\n# kept comment\n")
    return env


@pytest.mark.parametrize("old_config", [None, "missing", "{", '{"executor":"legacy","conversation_ids":[]}', "x" * 17000])
def test_retired_host_file_cannot_select_executor(tmp_path, old_config):
    env = setup_env(tmp_path)
    config = tmp_path / "old-rollout.json"
    if old_config not in (None, "missing"):
        config.write_text(old_config)
    before = config.read_bytes() if config.exists() else None
    result = apply_rollout(env, config if old_config is not None else None)
    assert result == {"executor": "langgraph", "cohort_size": 0, "all_conversations": True, "checkpoint_enabled": False}
    assert (config.read_bytes() if config.exists() else None) == before
    assert env.stat().st_mode & 0o777 == 0o600
    settings = Settings(_env_file=env)
    assert settings.jwt_secret == "synthetic-secret" and settings.web_concurrency == 2
    assert "chat_executor" not in Settings.model_fields
    assert "chat_graph_all_conversations" not in Settings.model_fields
    assert "chat_graph_conversation_allowlist" not in Settings.model_fields


def test_fixed_flags_keep_qualified_pre_retirement_images_on_graph(tmp_path):
    # Exact pre-retirement field types/defaults. Image rollback itself is covered
    # by G03.05; this checks the environment contract on each deploy.
    class OlderImageSettings(BaseSettings):
        chat_executor: str = "legacy"
        chat_graph_all_conversations: bool = False
        chat_graph_conversation_allowlist: str = ""
        model_config = SettingsConfigDict(extra="ignore")
    env = setup_env(tmp_path)
    with env.open("a") as output:
        output.write('CHAT_EXECUTOR=legacy\nexport CHAT_EXECUTOR=legacy\nCHAT_GRAPH_CONVERSATION_ALLOWLIST=other\nCHAT_GRAPH_ALL_CONVERSATIONS=false\n')
    apply_rollout(env)
    first = env.read_bytes()
    apply_rollout(env)
    assert env.read_bytes() == first
    old = OlderImageSettings(_env_file=env)
    assert old.chat_executor == "langgraph" and old.chat_graph_all_conversations
    assert old.chat_graph_conversation_allowlist == ""
    assert env.read_text().count("CHAT_EXECUTOR=") == 1
    assert "# kept comment" in env.read_text()


def test_failed_replace_preserves_environment_and_cleans_temp(tmp_path, monkeypatch):
    env = setup_env(tmp_path)
    before = env.read_bytes()
    def fail(*args):
        raise OSError("synthetic failure")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        apply_rollout(env)
    assert env.read_bytes() == before
    assert not list(tmp_path.glob(".graph-env-*"))


def test_write_failure_blocks_deployment_before_stop_without_secrets(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"] if s["name"] == "Deploy server stack on VPS")
    script = step["with"]["script"]
    gate = "python3 scripts/chat_graph_deploy_config.py"
    assert script.index(gate) < script.index("source scripts/deploy_apns_preflight.sh")
    assert script.index(gate) < script.index('echo "==> Stopping server before migrations"')
    # Execute the actual command, replacing only the environment pathname.
    command = script[script.index(gate):script.index("python3 scripts/redis_alert_deploy_config.py")].strip()
    command = command.replace("--env-file .env", f"--env-file '{tmp_path / 'missing.env'}'")
    result = subprocess.run(["bash"], cwd=ROOT, text=True, capture_output=True, timeout=10,
                            input="set -euo pipefail\n" + command + "\necho SERVER_STOP_REACHED\n")
    assert result.returncode == 2
    assert "SERVER_STOP_REACHED" not in result.stdout
    assert "missing.env" not in result.stderr


def test_older_cli_argument_is_accepted_but_never_read(tmp_path):
    env = setup_env(tmp_path)
    obsolete = tmp_path / "old-rollout.json"
    obsolete.symlink_to(tmp_path / "does-not-exist")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/chat_graph_deploy_config.py"),
                             "--env-file", str(env), "--config-file", str(obsolete)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0
    assert '"executor": "langgraph"' in result.stdout
    assert "synthetic-secret" not in result.stdout + result.stderr
