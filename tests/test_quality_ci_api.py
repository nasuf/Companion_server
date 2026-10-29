"""Repository OIDC can only control its own lease and evaluation budget."""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from tests.test_quality_postgres import quality
from tests.test_service_health_postgres import world
from app.api import quality_ci as api
from app.services.quality import jobs, oidc, snapshots, regression, budget
from app.services.quality import search
from app.config import settings


@pytest.fixture
async def control(quality, monkeypatch):
    database, *_ = quality
    monkeypatch.setattr(api, "db", database)
    monkeypatch.setattr(regression, "db", database)
    monkeypatch.setattr(settings, "quality_enabled", True)
    snapshot = {
        "prompts": [{"key": "synthetic", "content": "合成", "is_enabled": True}],
        "route": ["ark", "model"],
        "prices": {"ark/model": {"input": 0.8, "output": 2, "cached_input": 0.16}},
    }
    monkeypatch.setattr(snapshots, "capture", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(regression, "corpus", AsyncMock(return_value=[]))
    identity = {
        "sha": "a" * 40,
        "run_id": str(int(uuid4().hex[:12], 16)),
        "run_attempt": "1",
    }
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[oidc.identity] = lambda: identity
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, identity, database


async def test_ci_complete_cycle_no_business_access_and_idempotent_begin(control):
    client, identity, db = control
    body = {"source_digest": "a" * 64, "keys_configured": True}
    started = (await client.post("/admin-api/quality-ci/begin", json=body)).json()
    assert (
        started["status"] == "running"
        and started["prices"]["ark/model"]["input"] == 0.8
    )
    job = started["job"]
    url = "/admin-api/quality-ci/jobs/" + job["id"]

    async def action(op, data=None, token=None):
        return await client.post(
            url,
            json={
                "operation": op,
                "data": data or {},
                "lease_token": token or job["lease_token"],
            },
        )

    assert (await action("renew", token="x" * 32)).status_code == 409
    assert (
        await action(
            "prepare",
            {"corpus_hash": "h", "repeats": 3, "gate_status": "baseline_recorded"},
        )
    ).status_code == 200
    reserved = await action(
        "reserve",
        {
            "model": "ark/model",
            "messages": [{"role": "user", "content": "合成"}],
            "max_tokens": 10,
        },
    )
    charge = reserved.json()["id"]
    assert (
        await action(
            "reserve", {"model": "unapproved/x", "messages": [], "max_tokens": 10}
        )
    ).status_code == 400
    assert (
        await action("settle", {"charge_id": "unowned", "usage": {}})
    ).status_code == 400
    assert (await action("uncertain", {"charge_id": charge})).status_code == 200
    usage = {"input": 10, "output": 5, "cached_input": 0}
    assert (
        await action(
            "settle", {"charge_id": charge, "usage": {**usage, "cached_input": 11}}
        )
    ).status_code == 400
    assert (
        await action("settle", {"charge_id": charge, "usage": usage})
    ).status_code == 200
    result = {
        "case_id": "synthetic",
        "repetition": 0,
        "critical": True,
        "suite": "s",
        "pass": True,
        "turns": 1,
    }
    assert (await action("result", result)).status_code == 200
    assert (
        await action("result", {**result, "secret": "not allowed"})
    ).status_code == 400
    assert (await action("baseline", {"corpus_hash": "h"})).json() == {
        "id": None,
        "results": [],
    }
    assert (
        await action(
            "finish",
            {"status": "completed", "report": {"gate_status": "baseline_recorded"}},
        )
    ).status_code == 200
    assert (await action("renew")).status_code == 409
    repeated = (await client.post("/admin-api/quality-ci/begin", json=body)).json()
    assert repeated["status"] == "completed"
    identity["run_id"] = str(int(identity["run_id"]) + 1)
    skipped = (await client.post("/admin-api/quality-ci/begin", json=body)).json()
    assert skipped["reason"] == "chat_source_unchanged"
    assert (
        await client.post(
            url, json={"operation": "renew", "lease_token": job["lease_token"]}
        )
    ).status_code == 404
    # OIDC identity is not a JWT admin identity and exposes no business endpoint.
    assert (await client.get("/admin-api/quality/samples/anything")).status_code == 404


async def test_ci_search_reservation_binds_operation_and_includes_tool_fee(
    control, monkeypatch
):
    client, identity, db = control
    snapshot = dict(
        prompts=[dict(key="synthetic", content="合成", is_enabled=True)],
        route=search.MODEL.split("/"),
        prices={search.MODEL: dict(input=0.8, output=2, cached_input=0.16)},
    )
    monkeypatch.setattr(snapshots, "capture", AsyncMock(return_value=snapshot))
    started = (
        await client.post(
            "/admin-api/quality-ci/begin",
            json=dict(source_digest="b" * 64, keys_configured=True),
        )
    ).json()
    job = started["job"]

    async def act(op, data):
        return await client.post(
            "/admin-api/quality-ci/jobs/" + job["id"],
            json=dict(operation=op, data=data, lease_token=job["lease_token"]),
        )

    request = dict(
        model=search.MODEL,
        messages=[dict(role="user", content="合成")],
        max_tokens=900,
        operation="web_search",
    )
    response = await act("reserve", request)
    assert response.status_code == 200
    cid = response.json()["id"]
    row = (await db.query_raw("SELECT * FROM quality_charges WHERE id=$1", cid))[0]
    assert row["request_policy"] == dict(operation="web_search", max_tokens=900)
    assert float(row["reserved_cny"]) == pytest.approx(0.3293728)
    used = dict(
        input=25000,
        output=120,
        cached_input=1000,
        operation="web_search",
        search_policy=search.POLICY,
        web_search_calls=1,
    )
    assert (
        await act(
            "settle",
            dict(
                charge_id=cid,
                usage={
                    k: v
                    for k, v in used.items()
                    if k not in {"operation", "search_policy", "web_search_calls"}
                },
            ),
        )
    ).status_code == 400
    assert (
        await act("settle", dict(charge_id=cid, usage={**used, "input": 32769}))
    ).status_code == 400
    assert (await act("settle", dict(charge_id=cid, usage=used))).status_code == 200
    row = (await db.query_raw("SELECT cost_cny FROM quality_charges WHERE id=$1", cid))[
        0
    ]
    assert float(row["cost_cny"]) == pytest.approx(0.0236)
    cid = (await act("reserve", {**request, "operation": "chat"})).json()["id"]
    assert (await act("settle", dict(charge_id=cid, usage=used))).status_code == 400
    await jobs.finish(job, "completed", {})


@pytest.mark.parametrize(
    "mode", ["disabled", "missing_keys", "busy", "manifest_error", "exhausted"]
)
async def test_ci_unavailable_states_and_failure_notifications(
    control, monkeypatch, mode
):
    client, identity, db = control
    if mode == "disabled":
        monkeypatch.setattr(settings, "quality_enabled", False)
    if mode == "busy":
        monkeypatch.setattr(jobs, "claim", AsyncMock(return_value=None))
    if mode == "manifest_error":
        monkeypatch.setattr(
            regression, "corpus", AsyncMock(side_effect=ValueError("synthetic"))
        )
    body = {"source_digest": "b" * 64, "keys_configured": mode != "missing_keys"}
    if mode == "manifest_error":
        with pytest.raises(ValueError):
            await client.post("/admin-api/quality-ci/begin", json=body)
        jobs.notify.assert_awaited()
        return
    response = await client.post("/admin-api/quality-ci/begin", json=body)
    if mode == "busy":
        assert response.status_code == 409
        return
    if mode in {"disabled", "missing_keys"}:
        assert response.json()["status"] == "not_configured"
        return
    job = response.json()["job"]
    monkeypatch.setattr(
        budget, "reserve", AsyncMock(side_effect=budget.BudgetExhausted("run"))
    )
    assert (
        await client.post(
            "/admin-api/quality-ci/jobs/" + job["id"],
            json={
                "lease_token": job["lease_token"],
                "operation": "reserve",
                "data": {
                    "model": "ark/model",
                    "messages": [{"role": "user", "content": "合成"}],
                    "max_tokens": 10,
                },
            },
        )
    ).status_code == 429


async def test_oidc_header_failures_are_401(monkeypatch):
    from fastapi import HTTPException

    for header in ("", "Bearer " + ("x" * 16000), "Basic x"):
        with pytest.raises(HTTPException) as error:
            await oidc.identity(header)
        assert error.value.status_code == 401
    monkeypatch.setattr(
        oidc, "verify", lambda _: (_ for _ in ()).throw(ValueError("invalid"))
    )
    with pytest.raises(HTTPException) as error:
        await oidc.identity("Bearer invalid")
    assert error.value.status_code == 401
