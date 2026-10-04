"""Exercise rollout configuration and its production restart failure gate."""

import json
import os
from pathlib import Path
import subprocess
from uuid import UUID

import pytest
import yaml

from scripts.chat_graph_deploy_config import apply_rollout
from app.config import Settings
from app.services.chat.graph_executor import select_executor


ROOT = Path(__file__).resolve().parents[1]
CONVERSATION = str(UUID(int=1))
OTHER_CONVERSATION = str(UUID(int=2))


@pytest.fixture(autouse=True)
def isolate_startup_environment(monkeypatch):
    for key in ("CHAT_EXECUTOR", "CHAT_GRAPH_CONVERSATION_ALLOWLIST", "CHAT_GRAPH_ALL_CONVERSATIONS", "JWT_SECRET", "WEB_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)


def setup_files(tmp_path, config):
    env = tmp_path / ".env"
    env.write_text("JWT_SECRET=synthetic-secret\nWEB_CONCURRENCY=2\n")
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps(config))
    return env, path


def test_missing_configuration_defaults_to_legacy(tmp_path):
    env, path = setup_files(tmp_path, {})
    path.unlink()
    result = apply_rollout(env, path)
    settings = Settings(_env_file=env)
    assert result == {"executor": "legacy", "cohort_size": 0, "all_conversations": False, "checkpoint_enabled": False}
    assert settings.chat_executor == "legacy"
    assert settings.chat_graph_conversation_allowlist == ""
    assert env.stat().st_mode & 0o777 == 0o600


def test_only_exact_cohort_uses_graph_and_rollback_is_persistent(tmp_path, monkeypatch):
    env, path = setup_files(tmp_path, {"executor": "langgraph", "conversation_ids": [CONVERSATION]})
    apply_rollout(env, path)
    apply_rollout(env, path)
    settings = Settings(_env_file=env)
    monkeypatch.setattr("app.config.settings", settings)
    assert select_executor(CONVERSATION) == "langgraph"
    assert select_executor(OTHER_CONVERSATION) == "legacy"
    assert settings.jwt_secret == "synthetic-secret"
    assert settings.web_concurrency == 2
    assert env.read_text().count("CHAT_EXECUTOR=") == 1
    path.write_text(json.dumps({"executor": "legacy", "conversation_ids": []}))
    apply_rollout(env, path)
    monkeypatch.setattr("app.config.settings", Settings(_env_file=env))
    assert select_executor(CONVERSATION) == "legacy"
    assert select_executor(OTHER_CONVERSATION) == "legacy"
    # A later deployment regenerates .env, then applies the retained host file.
    env.write_text("JWT_SECRET=synthetic-secret\nWEB_CONCURRENCY=2\n")
    apply_rollout(env, path)
    assert Settings(_env_file=env).chat_executor == "legacy"


def test_explicit_full_rollout_survives_deploy_and_can_return_to_cohort_or_legacy(tmp_path, monkeypatch):
    env, path = setup_files(tmp_path, {
        "executor": "langgraph", "conversation_ids": [], "all_conversations": True,
    })
    for _ in range(2):
        # Deployment regenerates .env before applying the persistent private file.
        env.write_text("JWT_SECRET=synthetic-secret\nWEB_CONCURRENCY=2\n")
        result = apply_rollout(env, path)
        assert result["all_conversations"] is True
        settings = Settings(_env_file=env)
        monkeypatch.setattr("app.config.settings", settings)
        assert select_executor(CONVERSATION) == select_executor(OTHER_CONVERSATION) == "langgraph"
        assert settings.chat_graph_conversation_allowlist == ""
        assert settings.jwt_secret == "synthetic-secret" and settings.web_concurrency == 2
    assert env.read_text().count("CHAT_GRAPH_ALL_CONVERSATIONS=") == 1
    path.write_text(json.dumps({"executor": "langgraph", "conversation_ids": [CONVERSATION]}))
    apply_rollout(env, path)
    monkeypatch.setattr("app.config.settings", Settings(_env_file=env))
    assert select_executor(CONVERSATION) == "langgraph"
    assert select_executor(OTHER_CONVERSATION) == "legacy"
    path.write_text(json.dumps({"executor": "legacy", "conversation_ids": []}))
    apply_rollout(env, path)
    monkeypatch.setattr("app.config.settings", Settings(_env_file=env))
    assert select_executor(CONVERSATION) == select_executor(OTHER_CONVERSATION) == "legacy"
    assert not Settings(_env_file=env).chat_graph_all_conversations


@pytest.mark.parametrize("config", [
    {}, [], {"executor": "unknown", "conversation_ids": []},
    {"executor": "langgraph", "conversation_ids": []},
    {"executor": "legacy", "conversation_ids": [CONVERSATION]},
    {"executor": "langgraph", "conversation_ids": "*"},
    {"executor": "langgraph", "conversation_ids": ["*"]},
    {"executor": "langgraph", "conversation_ids": [None]},
    {"executor": "langgraph", "conversation_ids": [CONVERSATION + "\nJWT_SECRET=x"]},
    {"executor": "langgraph", "conversation_ids": [CONVERSATION, CONVERSATION]},
    {"executor": "langgraph", "conversation_ids": [str(UUID(int=i)) for i in range(101)]},
    {"executor": "legacy", "conversation_ids": [], "checkpoint_enabled": True},
    {"executor": "legacy", "conversation_ids": [], "all_conversations": True},
    {"executor": "langgraph", "conversation_ids": [CONVERSATION], "all_conversations": True},
    {"executor": "langgraph", "conversation_ids": [], "all_conversations": "true"},
    {"executor": "langgraph", "conversation_ids": [], "all_conversations": 1},
    {"executor": "langgraph", "conversation_ids": [], "all_conversations": None},
    {"executor": "langgraph", "conversation_ids": [], "all_conversations": False},
])
def test_invalid_cohort_leaves_environment_unchanged(tmp_path, config):
    env, path = setup_files(tmp_path, config)
    before = env.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        apply_rollout(env, path)
    assert env.read_bytes() == before


def test_failed_replace_preserves_environment_and_removes_temporary_file(tmp_path, monkeypatch):
    env, path = setup_files(tmp_path, {"executor": "legacy", "conversation_ids": []})
    before = env.read_bytes()
    def fail(*args):
        raise OSError("synthetic failure")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        apply_rollout(env, path)
    assert env.read_bytes() == before
    assert not list(tmp_path.glob(".graph-env-*"))


def test_invalid_host_configuration_blocks_actual_deployment_gate(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"]
                if s["name"] == "Deploy server stack on VPS")
    script = step["with"]["script"]
    gate = "python3 scripts/chat_graph_deploy_config.py"
    assert script.index(gate) < script.index("source scripts/deploy_apns_preflight.sh")
    assert script.index(gate) < script.index('echo "==> Stopping server before migrations"')
    command = script[script.index(gate):].split("\n\n", 1)[0]
    env, path = setup_files(tmp_path, {"executor": "langgraph", "conversation_ids": ["*"]})
    command = command.replace("--env-file .env", f"--env-file '{env}'")
    command = command.replace("/app/companion-secrets/chat-graph-rollout.json", str(path))
    result = subprocess.run(["bash"], cwd=ROOT, text=True, capture_output=True, timeout=10,
                            input="set -euo pipefail\n" + command + "\necho SERVER_STOP_REACHED\n")
    assert result.returncode == 2
    assert "SERVER_STOP_REACHED" not in result.stdout
    assert "synthetic-secret" not in result.stdout + result.stderr
    assert "*" not in result.stdout + result.stderr


@pytest.mark.parametrize("content", ["{", "x" * 16385,
    '{"executor":"langgraph","executor":"legacy","conversation_ids":[]}'])
def test_unreadable_configuration_cli_does_not_leak_contents(tmp_path, content):
    env, path = setup_files(tmp_path, {})
    path.write_text(content)
    result = subprocess.run(["python3", str(ROOT / "scripts/chat_graph_deploy_config.py"),
                             "--env-file", str(env), "--config-file", str(path)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "synthetic-secret" not in result.stdout + result.stderr


def test_broken_config_symlink_is_not_treated_as_missing(tmp_path):
    env, path = setup_files(tmp_path, {})
    path.unlink()
    path.symlink_to(tmp_path / "missing.json")
    before = env.read_bytes()
    with pytest.raises(OSError):
        apply_rollout(env, path)
    assert env.read_bytes() == before
