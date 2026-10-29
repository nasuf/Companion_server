"""Offline checks of the real Actions controller, without keys or network."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.quality import nightly
from app.services.quality import budget, jobs, regression, transport


@pytest.fixture
def controller(monkeypatch):
    for name in (
        "DATABASE_URL",
        "REDIS_URL",
        "PYTHON_DOTENV_DISABLED",
        "EVAL_ARK_API_KEY",
        "EVAL_DASHSCOPE_API_KEY",
    ):
        monkeypatch.setenv(name, "synthetic-only")
    monkeypatch.setattr(nightly, "digest", lambda _: "a" * 64)
    # main deliberately replaces this only in the dedicated CLI process.
    monkeypatch.setattr(transport, "budget", budget)
    control = nightly.Control()
    monkeypatch.setattr(nightly, "Control", lambda: control)
    previous = logging.root.manager.disable
    yield control
    logging.disable(previous)


@pytest.mark.parametrize(
    "status,code",
    [("skipped", 0), ("not_configured", 0), ("completed", 0), ("failed", 1)],
)
async def test_nightly_nonrunning_never_calls_provider(
    controller, monkeypatch, capsys, status, code
):
    controller.request = AsyncMock(return_value={"status": status, "reason": "test"})
    run = AsyncMock()
    monkeypatch.setattr(regression, "run", run)
    assert await nightly.main() == code
    run.assert_not_awaited()
    assert '"status": "' + status + '"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,code", [("completed", 0), ("failed", 1), ("budget_exhausted", 1)]
)
async def test_nightly_reports_gate_and_keeps_dedicated_keys_in_runner(
    controller, monkeypatch, capsys, status, code
):
    from app.config import settings
    from app.services import runtime_config

    monkeypatch.setattr(settings, "quality_critical_repeats", 3)
    monkeypatch.setattr(runtime_config, "_PRICING_CACHE", {})
    controller.request = AsyncMock(
        return_value={
            "status": "running",
            "job": {"id": "job", "lease_token": "lease"},
            "cases": [],
            "prices": {},
            "policy": {"critical_repeats": 3},
        }
    )

    async def replay(job, **kwargs):
        assert kwargs["key_overrides"] == {
            "ark": "synthetic-only",
            "dashscope": "synthetic-only",
        }
        assert transport.budget is controller
        controller.status = status
        controller.report = {
            "gate_status": "passed" if status == "completed" else "incomplete",
            "evaluated": 1,
            "expected": 1,
        }

    monkeypatch.setattr(regression, "run", replay)
    assert await nightly.main() == code
    output = capsys.readouterr().out
    assert "synthetic-only" not in output and "lease" not in output


async def test_nightly_control_errors_are_sanitized(monkeypatch):
    import httpx

    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.invalid?test=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "synthetic-token")
    factory = httpx.AsyncClient
    for status, error in [
        (429, budget.BudgetExhausted),
        (409, jobs.LostLease),
        (500, RuntimeError),
    ]:

        def respond(request):
            return (
                httpx.Response(200, json={"value": "synthetic-jwt"})
                if request.method == "GET"
                else httpx.Response(status, text="private upstream text")
            )

        monkeypatch.setattr(
            nightly.httpx,
            "AsyncClient",
            lambda **kw: factory(transport=httpx.MockTransport(respond)),
        )
        with pytest.raises(error) as caught:
            await nightly.Control().request("/begin", {})
        assert "private" not in str(caught.value)


async def test_nightly_lost_lease_cancels_billed_work(controller, monkeypatch):
    from app.config import settings
    from app.services import runtime_config

    monkeypatch.setattr(settings, "quality_critical_repeats", 3)
    monkeypatch.setattr(runtime_config, "_PRICING_CACHE", {})
    controller.request = AsyncMock(
        return_value={
            "status": "running",
            "job": {"id": "job", "lease_token": "lease"},
            "cases": [],
            "prices": {},
            "policy": {"critical_repeats": 3},
        }
    )
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def replay(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def sleep(_):
        await started.wait()

    controller.renew = AsyncMock(side_effect=jobs.LostLease("synthetic-lost"))
    monkeypatch.setattr(regression, "run", replay)
    monkeypatch.setattr(nightly.asyncio, "sleep", sleep)
    with pytest.raises(jobs.LostLease):
        await nightly.main()
    assert cancelled.is_set()


def test_workflow_keeps_eval_keys_out_of_deployment():
    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/quality-nightly.yml").read_text()
    )
    job = workflow["jobs"]["replay"]
    assert job["permissions"]["id-token"] == "write"
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert (
        "cancel-in-progress: false"
        in (root / ".github/workflows/quality-nightly.yml").read_text()
    )
    assert "EVAL_ARK_API_KEY" not in (root / ".github/workflows/deploy.yml").read_text()
    assert not any("upload-artifact" in s.get("uses", "") for s in job["steps"])
