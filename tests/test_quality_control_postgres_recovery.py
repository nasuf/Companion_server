"""A committed action with a lost HTTP response must not duplicate accounting."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tests.test_quality_ci_api import control
from tests.test_quality_postgres import quality
from tests.test_service_health_postgres import world
from evals.quality import nightly


@pytest.mark.parametrize("operation", ["settle", "result", "reserve"])
async def test_response_loss_after_commit(control, monkeypatch, operation):
    client, _, db = control
    started = (
        await client.post(
            "/admin-api/quality-ci/begin",
            json={"source_digest": "c" * 64, "keys_configured": True},
        )
    ).json()
    job = started["job"]
    request = {
        "model": "ark/model",
        "messages": [{"role": "user", "content": "合成"}],
        "max_tokens": 10,
    }
    api_path = "/admin-api/quality-ci/jobs/" + job["id"]
    reserved = (
        (
            await client.post(
                api_path,
                json={
                    "lease_token": job["lease_token"],
                    "operation": "reserve",
                    "data": request,
                },
            )
        ).json()["id"]
        if operation == "settle"
        else None
    )
    data = {
        "settle": {
            "charge_id": reserved,
            "usage": {"input": 10, "output": 5, "cached_input": 0},
        },
        "result": {
            "case_id": "synthetic",
            "repetition": 0,
            "critical": True,
            "suite": "s",
            "pass": True,
            "turns": 1,
        },
        "reserve": request,
    }[operation]
    count = 0

    async def respond(req):
        nonlocal count
        if req.method == "GET":
            return httpx.Response(200, json={"value": "synthetic-jwt"})
        count += 1
        response = await client.post(api_path, json=json.loads(req.content))
        assert response.status_code == 200
        if count == 1:
            raise httpx.ReadError("response lost after commit", request=req)
        return httpx.Response(200, json=response.json())

    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.invalid?test=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "synthetic-secret")
    factory = httpx.AsyncClient
    monkeypatch.setattr(
        nightly.httpx,
        "AsyncClient",
        lambda **kw: factory(transport=httpx.MockTransport(respond), **kw),
    )
    monkeypatch.setattr(nightly.asyncio, "sleep", AsyncMock())
    runner = nightly.Control()
    runner.job = job
    if operation == "reserve":
        with pytest.raises(nightly.ControlRequestError):
            await runner.action(operation, data)
        assert count == 1
    else:
        assert await runner.action(operation, data) == {"ok": True}
        assert count == 2
    if operation == "result":
        rows = await db.query_raw(
            "SELECT result FROM quality_results WHERE job_id=$1", job["id"]
        )
        assert len(rows) == 1 and rows[0]["result"] == data
    else:
        charges = await db.query_raw(
            "SELECT id,status,cost_cny FROM quality_charges WHERE job_id=$1", job["id"]
        )
        assert len(charges) == 1
        if operation == "settle":
            rows = await db.query_raw(
                "SELECT cost_cny FROM llm_usage WHERE id=$1", "quality-" + reserved
            )
            assert len(rows) == 1
            assert float(rows[0]["cost_cny"]) == pytest.approx(0.000018)
            assert float(charges[0]["cost_cny"]) == pytest.approx(0.000018)
        else:
            assert charges[0]["status"] == "reserved"
