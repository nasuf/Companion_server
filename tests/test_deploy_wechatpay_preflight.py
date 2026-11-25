"""Execute the WeChat Pay deploy preflight against fake .env files."""
from pathlib import Path
import os
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deploy_wechatpay_preflight.sh"
REQUIRED = ("WECHATPAY_MCH_ID", "WECHATPAY_CERT_SERIAL_NO", "WECHATPAY_PUBLIC_KEY_ID",
            "WECHATPAY_API_V3_KEY", "WECHATPAY_NOTIFY_URL", "WECHAT_H5_APP_ID")


def _run(tmp_path, values):
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    # No sudo on CI runners; the script reads the PEM paths as the current user.
    return subprocess.run(["bash", str(SCRIPT), str(env_file)], text=True, capture_output=True,
                          env={**os.environ, "SUDO": ""}, timeout=10)


def _enabled(tmp_path):
    for name in ("apiclient_key.pem", "pub_key.pem"):
        (tmp_path / name).write_text("pem")
    return {"WECHATPAY_ENABLED": "true", **{key: "set" for key in REQUIRED},
            "WECHATPAY_PRIVATE_KEY_PATH": str(tmp_path / "apiclient_key.pem"),
            "WECHATPAY_PUBLIC_KEY_PATH": str(tmp_path / "pub_key.pem")}


@pytest.mark.parametrize("flag", ["false", "", None])
def test_disabled_checks_nothing(tmp_path, flag):
    values = {} if flag is None else {"WECHATPAY_ENABLED": flag}
    assert _run(tmp_path, values).returncode == 0


def test_complete_config_passes(tmp_path):
    result = _run(tmp_path, _enabled(tmp_path))
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("key", REQUIRED)
def test_missing_value_blocks_deploy(tmp_path, key):
    values = _enabled(tmp_path)
    values[key] = ""
    result = _run(tmp_path, values)
    assert result.returncode == 1
    assert f"{key} is required" in result.stdout


@pytest.mark.parametrize("key", ["WECHATPAY_PRIVATE_KEY_PATH", "WECHATPAY_PUBLIC_KEY_PATH"])
@pytest.mark.parametrize("path", ["", "/nonexistent/key.pem"])
def test_unreadable_pem_blocks_deploy(tmp_path, key, path):
    values = _enabled(tmp_path)
    values[key] = path
    result = _run(tmp_path, values)
    assert result.returncode == 1
    assert f"{key} must point to a readable PEM file" in result.stdout


def test_workflow_runs_preflight_before_production_stop():
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    step = next(s for s in workflow["jobs"]["deploy"]["steps"]
                if s["name"] == "Deploy server stack on VPS")
    script = step["with"]["script"]
    assert script.index("bash scripts/deploy_wechatpay_preflight.sh") < script.index(
        'echo "==> Stopping server before migrations"')
