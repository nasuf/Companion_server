"""Legacy APNs logging protection through the configured application filter."""
import logging
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from app.config import settings
from app.observability.log_filter import ContextInjectionFilter
from app.services.notifications.apns import ApnsClient


@pytest.mark.asyncio
async def test_legacy_apns_request_log_redacts_device_token_without_muting_other_requests(monkeypatch, caplog):
    token = "abcd" * 17
    actual_client = httpx.AsyncClient
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(200, headers={"apns-id": "synthetic-legacy-id"},
            extensions={"http_version": b"HTTP/2"})

    def mocked_client(**options):
        return actual_client(**options, transport=httpx.MockTransport(response))

    monkeypatch.setattr(httpx, "AsyncClient", mocked_client)
    monkeypatch.setattr(ApnsClient, "configured", property(lambda self: True))
    monkeypatch.setattr(ApnsClient, "_provider_token", lambda self: "synthetic.provider.signature")
    monkeypatch.setattr(settings, "apns_topic", "com.companion.synthetic")
    application_filter = ContextInjectionFilter()
    caplog.handler.addFilter(application_filter)
    try:
        with caplog.at_level(logging.INFO, logger="httpx"):
            result = await ApnsClient().send_alert(token=token, title="synthetic", body="synthetic",
                payload={"type": "agent_message"}, environment="sandbox")
            async with actual_client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
                await client.get("https://synthetic.invalid/ordinary-request")
    finally:
        caplog.handler.removeFilter(application_filter)
    assert result.ok and len(requests) == 1
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "ordinary-request" in messages and token not in messages


def test_built_image_check_uses_real_logging_setup_without_loading_workspace_env(tmp_path):
    # Even an invalid local .env must not affect the isolated check.
    (tmp_path / ".env").write_text("APP_ENV=production\nAPNS_ENABLED=not-a-boolean\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_apns_log_redaction.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stderr
    assert "APNs logging check passed" in result.stdout
