"""Optional live browser extension of test_activity_admin's real HTTP/data test.

Set ACTIVITY_ADMIN_WEB_PROJECT to a sibling Web checkout. CI separately gates
the Web browser fixture; this mode additionally runs it against real API/PG.
"""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import httpx

from app.services.auth import create_jwt


async def check_browser(s, mobile_payload, web_project):
    root = Path(web_project).resolve()
    assert (root / "tests/activityManagement.browser.mjs").is_file()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    sample = s.tmp / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "user_id": s.user,
                "username": "admin-activity-" + s.user,
                "activity": mobile_payload,
                "other_user_id": s.bob,
                "other_activity_id": s.foreign,
            },
            ensure_ascii=False,
        )
    )
    env = {
        **os.environ,
        "ACTIVITY_ADMIN_REAL_API": base,
        "ACTIVITY_ADMIN_SAMPLE": str(sample),
        "ACTIVITY_ADMIN_E2E_TOKEN": create_jwt(s.user, "admin"),
    }
    log = (s.tmp / "http.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.activity_admin_http_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(100):
                try:
                    if (await client.get(base + "/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None:
                    raise AssertionError(
                        "Disposable HTTP host stopped: "
                        + (s.tmp / "http.log").read_text()
                    )
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("Disposable HTTP host not ready")
        result = await asyncio.create_subprocess_exec(
            "node",
            "tests/activityManagement.browser.mjs",
            cwd=root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(result.communicate(), 120)
        except TimeoutError:
            result.kill()
            await result.wait()
            raise
        assert result.returncode == 0, output.decode()
    finally:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)
        log.close()
