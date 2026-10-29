"""GitHub Actions controller. Public logs contain only status/counts, never text.

Dedicated EVAL keys stay in this runner. The server shares its atomic budget
ledger using short-lived, repository-bound GitHub OIDC; no production secret is
copied into Actions. Prompt/case payloads are kept in memory, not artifacts.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import httpx

BASE = "https://banshengcomp.com/api/admin-api/quality-ci"


class Control:
    def __init__(self):
        self.job = None

    async def request(self, path, data):
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            url = (
                os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
                + "&audience=companion-quality-evaluation"
            )
            oidc = await client.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
                },
            )
            oidc.raise_for_status()
            response = await client.post(
                BASE + path,
                headers={"Authorization": "Bearer " + oidc.json()["value"]},
                json=data,
            )
            from app.services.quality import budget, jobs

            if response.status_code == 429:
                raise budget.BudgetExhausted("server_budget")
            if response.status_code == 409:
                raise jobs.LostLease("server_lease_busy_or_lost")
            if response.status_code != 200:
                raise RuntimeError("quality_control_http_" + str(response.status_code))
            return response.json()

    async def action(self, operation, data=None):
        return await self.request(
            "/jobs/" + self.job["id"],
            dict(
                lease_token=self.job["lease_token"],
                operation=operation,
                data=data or {},
            ),
        )

    async def prepare(self, report):
        await self.action("prepare", report)

    async def renew(self):
        await self.action("renew")

    async def result(self, row):
        await self.action("result", row)

    async def baseline(self, fingerprint):
        value = await self.action("baseline", {"corpus_hash": fingerprint})
        return value["id"], value["results"]

    async def finish(self, status, report):
        await self.action("finish", dict(status=status, report=report))
        self.status = status
        self.report = report

    async def reserve(
        self, job_id, model, ceiling, *, lease_token, request=None, **kwargs
    ):
        if not request:
            raise ValueError("quality_request_required_for_reservation")
        return (await self.action("reserve", dict(model=model, **request)))["id"]

    async def settle(self, charge_id, cost, usage):
        await self.action("settle", dict(charge_id=charge_id, usage=usage))

    async def uncertain(self, charge_id):
        await self.action("uncertain", dict(charge_id=charge_id))


def digest(root):
    # Business chat dependencies and replay code only. Admin-only UI/monitoring
    # deployments must not trigger another billed full replay.
    import subprocess

    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "app/services/chat",
            "app/services/interaction",
            "app/services/llm",
            "app/services/prompting",
            "app/services/memory",
            "app/services/runtime",
            "app/services/graph",
            "app/services/temporal",
            "app/services/persona_grounding.py",
            "app/services/runtime_config.py",
            "app/api/realtime",
            "app/config.py",
            "app/services/quality",
            "evals",
            "tests/g02_harness_support.py",
            "tests/graph_harness_support.py",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    value = hashlib.sha256()
    for name in sorted(files):
        value.update(name.encode() + b"\0")
        value.update((root / name).read_bytes())
    return value.hexdigest()


async def main():
    logging.disable(logging.CRITICAL)
    os.environ.update(
        PYTHON_DOTENV_DISABLED="1",
        DATABASE_URL="postgresql://test:test@127.0.0.1:1/quality_no_database",
        REDIS_URL="redis://127.0.0.1:1/15",
    )
    control = Control()
    manifest = await control.request(
        "/begin",
        dict(
            source_digest=digest(Path.cwd()),
            keys_configured=bool(
                os.environ.get("EVAL_ARK_API_KEY")
                and os.environ.get("EVAL_DASHSCOPE_API_KEY")
            ),
        ),
    )
    if manifest["status"] != "running":
        print(
            json.dumps({"status": manifest["status"], "reason": manifest.get("reason")})
        )
        return (
            0 if manifest["status"] in {"skipped", "not_configured", "completed"} else 1
        )
    from app.config import settings
    from app.services import runtime_config
    from app.services.quality import transport, regression

    control.job = manifest["job"]
    runtime_config._PRICING_CACHE = manifest["prices"]
    settings.quality_critical_repeats = manifest["policy"]["critical_repeats"]
    # Only this CLI process uses a remote budget adapter; API workers are intact.
    transport.budget = control

    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            await control.renew()

    task = asyncio.create_task(
        regression.run(
            control.job,
            cases=manifest["cases"],
            sink=control,
            key_overrides={
                "ark": os.environ["EVAL_ARK_API_KEY"],
                "dashscope": os.environ["EVAL_DASHSCOPE_API_KEY"],
            },
        )
    )
    heart = asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait([task, heart], return_when=asyncio.FIRST_COMPLETED)
        for ended in done:
            ended.result()
        print(
            json.dumps(
                dict(
                    status=control.status,
                    gate_status=control.report["gate_status"],
                    evaluated=control.report["evaluated"],
                    expected=control.report["expected"],
                )
            )
        )
        return 0 if control.status == "completed" else 1
    finally:
        for pending in (task, heart):
            pending.cancel()
        await asyncio.gather(task, heart, return_exceptions=True)


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except BaseException as exc:
        # Never print HTTP errors, private context or tokens in a public Actions log.
        print(json.dumps({"status": "incomplete", "error_type": type(exc).__name__}))
        code = 1
    sys.exit(code)
