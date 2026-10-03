"""Verify the built image's APNs logging without credentials or network access.

Run with ``python -m scripts.check_apns_log_redaction``. The child starts in an
empty directory with synthetic settings, so a workspace .env is never loaded.
Only HTTPX MockTransport is used; no application lifespan or database is opened.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _check() -> None:
    if os.environ.get("COMPANION_LOG_CHECK_ISOLATED") != "1":
        raise RuntimeError("Use the isolated launcher")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import asyncio
    from contextlib import redirect_stderr
    import io
    import logging
    import socket
    from unittest.mock import patch

    network_attempts = []

    def deny_network(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError("Network is disabled in the APNs logging check")

    # This is a short-lived child, so the guard remains active through exit.
    socket.create_connection = socket.getaddrinfo = deny_network
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = deny_network

    import httpx
    from app.config import settings
    from app.middleware import configure_logging
    from app.observability import axiom_setup
    from app.observability.context import bind_context
    from app.services.notifications.apns import ApnsClient

    token = "ab" * 32
    recorded = []
    requests = []
    output = io.StringIO()
    actual_client = httpx.AsyncClient

    class RecordingAxiomHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def emit(self, record):
            recorded.append(dict(record.__dict__))

    def respond(request):
        requests.append(request)
        status = 410 if "sandbox" in request.url.host else 200
        return httpx.Response(status, json={"reason": "Unregistered"} if status == 410 else {},
                              headers={"apns-id": "synthetic-id"},
                              extensions={"http_version": b"HTTP/2"})

    def mocked_client(**options):
        return actual_client(**options, transport=httpx.MockTransport(respond))

    async def send():
        with patch.object(httpx, "AsyncClient", side_effect=mocked_client), \
                patch.object(ApnsClient, "configured", property(lambda self: True)), \
                patch.object(ApnsClient, "_provider_token", return_value="synthetic.signature"), \
                patch.object(settings, "apns_topic", "com.companion.synthetic"):
            for environment in ("production", "sandbox"):
                result = await ApnsClient().send_alert(token=token, title="synthetic", body="synthetic",
                    payload={"type": "agent_message"}, environment=environment)
                if environment == "production" and not result.ok:
                    raise RuntimeError("Successful APNs result changed")
                if environment == "sandbox" and not (result.status_code == 410 and result.unregister):
                    raise RuntimeError("Rejected APNs result changed")
        async with actual_client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
            await client.get("https://synthetic.invalid/ordinary-request")

    # Use the real logging setup and real QueueHandler/QueueListener. Replace
    # only the external Axiom client/sink; it must never receive private args.
    with patch.dict(os.environ, {"AXIOM_TOKEN": "synthetic", "AXIOM_DATASET": "synthetic"}), \
            patch("axiom_py.Client"), \
            patch("axiom_py.logging.AxiomHandler", RecordingAxiomHandler), \
            redirect_stderr(output):
        configure_logging()
        try:
            if axiom_setup._listener is None:
                raise RuntimeError("Axiom queue was not configured")
            # Production defaults to WARNING. Exercise the diagnostic INFO
            # path explicitly, without changing the application's log level.
            logging.getLogger("httpx").setLevel(logging.INFO)
            with bind_context(conversation_id="synthetic-conversation"):
                asyncio.run(send())
        finally:
            axiom_setup._shutdown_listener()

    console = output.getvalue()
    queued = repr(recorded)
    if token in console or token in queued:
        raise RuntimeError("APNs device token reached a logging handler")
    if any("/3/device/[redacted]" not in text or "ordinary-request" not in text
           for text in (console, queued)):
        raise RuntimeError("Redacted APNs or ordinary HTTP request logs were lost")
    apns_records = [r for r in recorded if "/3/device/" in str(r.get("msg"))]
    if len(apns_records) != 2 or any(r.get("conversation_id") != "synthetic-conversation" for r in apns_records):
        raise RuntimeError("Queued request count or logging context changed")
    if len(requests) != 2 or any(not r.url.path.endswith(token) for r in requests):
        raise RuntimeError("Logging redaction changed the outgoing request")
    if network_attempts:
        raise RuntimeError("The isolated check attempted network access")
    print("APNs logging check passed: console, Axiom queue, HTTPX and context; no network")


def main() -> int:
    if sys.argv[1:] == ["--isolated-child"]:
        _check()
        return 0
    if sys.argv[1:]:
        raise SystemExit("This check accepts no arguments")
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
    environment.update({
        "COMPANION_LOG_CHECK_ISOLATED": "1", "APP_ENV": "test",
        "DATABASE_URL": "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic",
        "REDIS_URL": "redis://127.0.0.1:1/0",
        "JWT_SECRET": "synthetic-logging-check-secret-at-least-32-characters",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    with tempfile.TemporaryDirectory(prefix="companion-log-check-") as directory:
        return subprocess.call([sys.executable, "-B", "-I", str(Path(__file__).resolve()),
                                "--isolated-child"], env=environment, cwd=directory, timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
