"""CHAT01.10.2 repetition gate: its own executor, month budget and private cases."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from tests.test_quality_postgres import quality, running
from tests.test_service_health_postgres import world
from app.api import quality_ci as api
from app.config import settings
from app.services.quality import budget, jobs, oidc, repetition_gate, snapshots

ROUTE = ["ark", "doubao-seed-character-260628"]
PRICE = {"input": 0.8, "output": 2, "cached_input": 0.16}


@pytest.fixture
async def gate(quality, monkeypatch):
    database, *_ = quality
    monkeypatch.setattr(api, "db", database)
    monkeypatch.setattr(repetition_gate, "db", database)
    monkeypatch.setattr(settings, "quality_enabled", True)
    snapshot = {
        "prompts": [{"key": "synthetic", "content": "合成", "is_enabled": True}],
        "route": ROUTE,
        "prices": {"/".join(ROUTE): PRICE, **{m: PRICE for m in repetition_gate.JUDGE_MODELS}},
    }
    monkeypatch.setattr(snapshots, "capture", AsyncMock(side_effect=lambda: {
        **snapshot, "prices": dict(snapshot["prices"])}))
    identity = {
        "sha": uuid4().hex + "a" * 8,
        "run_id": str(int(uuid4().hex[:12], 16)),
        "run_attempt": "1",
        "quality_workflow": "repetition_gate",
    }
    case_set = "test-" + uuid4().hex[:12]
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[oidc.identity] = lambda: identity
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, identity, database, case_set
    finally:
        await database.execute_raw(
            "DELETE FROM quality_repetition_cases WHERE case_set=$1", case_set
        )


def item(i):
    return {"case": {"id": f"h{i}", "message": "合成"}, "context": {"topic": {"t": i}}}


async def load(database, case_set, n=2):
    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        return await repetition_gate.import_cases(tx, case_set, [item(i) for i in range(n)])


def next_run(identity):
    identity["run_id"] = str(int(uuid4().hex[:12], 16))


async def test_gate_cycle_runs_on_its_own_executor_and_meters_blocks(gate):
    client, identity, database, case_set = gate
    begin = "/admin-api/quality-ci/repetition-gate/begin"
    body = {"mode": "gate", "case_set": case_set, "candidate_sha": "c" * 40, "keys_configured": True}
    assert (await client.post(begin, json={**body, "mode": "calibrate"})).status_code == 400
    missing = (await client.post(begin, json=body)).json()
    assert missing == {"status": "not_configured", "reason": "case_set_not_loaded"}
    assert await load(database, case_set) == 2
    next_run(identity)
    started = (await client.post(begin, json=body)).json()
    assert started["status"] == "running"
    assert [c["id"] for c in started["cases"]] == ["h0", "h1"]
    assert started["contexts"]["h1"] == {"topic": {"t": 1}}
    assert set(started["prices"]) == {"/".join(ROUTE), *repetition_gate.JUDGE_MODELS,
                                      repetition_gate.EMBEDDING_MODEL}
    assert started["max_block_cny"] == "2"
    assert all(isinstance(k, str) for k in started["protected_prompts"])
    job = started["job"]
    owner = dict(identity)

    # A second gate waits for the first; the nightly executor stays free.
    next_run(identity)
    assert (await client.post(begin, json=body)).status_code == 409
    other = await running()
    assert other is not None
    await jobs.finish(other, "completed", {})

    url = "/admin-api/quality-ci/repetition-gate/jobs/" + job["id"]

    async def act(op, data=None, token=None):
        return await client.post(url, json={"operation": op, "data": data or {},
                                            "lease_token": token or job["lease_token"]})

    identity.update(owner)  # actions are bound to the run that began the job
    assert (await act("renew", token="x" * 32)).status_code == 409
    assert (await act("renew")).status_code == 200
    assert (await act("reserve_block", {"model": "dashscope/other", "amount": "1"})).status_code == 400
    assert (await act("reserve_block", {"model": "/".join(ROUTE), "amount": "3"})).status_code == 400
    block = (await act("reserve_block", {"model": "/".join(ROUTE), "amount": "2"})).json()["id"]
    spare = (await act("reserve_block", {"model": repetition_gate.EMBEDDING_MODEL, "amount": "0.5"})).json()["id"]
    usage = {"input": 1000, "output": 100, "cached_input": 200, "calls": 3, "unknown_cny": 0.01}
    assert (await act("settle_block", {"charge_id": "unowned", "usage": usage})).status_code == 400
    assert (await act("settle_block", {"charge_id": block, "usage": {**usage, "calls": -1}})).status_code == 400
    assert (await act("settle_block", {"charge_id": block, "usage": usage})).status_code == 200
    assert (await act("uncertain", {"charge_id": spare})).status_code == 200
    row = (await database.query_raw(
        "SELECT cost_cny,call_count FROM llm_usage WHERE id=$1", "quality-" + block))[0]
    assert row["call_count"] == 3 and row["cost_cny"] == pytest.approx(0.010872)
    assert (await act("finish", {"status": "passed", "report": {}})).status_code == 400
    assert (await act("finish", {"status": "completed", "report": {"cost_cny": 0.01}})).status_code == 200
    assert (await act("renew")).status_code == 409
    assert (await act("renew", {"pad": "x" * 300_000})).status_code == 413
    missing_job = await client.post(url + "-missing", json={"operation": "renew", "lease_token": job["lease_token"]})
    assert missing_job.status_code == 404

    # Same main, candidate and cases: already decided, not billed again.
    next_run(identity)
    assert (await client.post(begin, json=body)).json() == {"status": "skipped", "reason": "already_evaluated"}
    identity.update(owner)
    assert (await client.post(begin, json=body)).json()["status"] == "completed"
    next_run(identity)
    calibrate = {"mode": "calibrate", "case_set": case_set, "keys_configured": True}
    assert (await client.post(begin, json={**calibrate, "keys_configured": False})).json()["reason"] == (
        "evaluation_secrets_not_configured")
    next_run(identity)
    started = (await client.post(begin, json=calibrate)).json()
    assert started["status"] == "running"
    await jobs.finish({**started["job"]}, "failed", {})



async def test_gate_and_other_kinds_each_keep_their_own_month(gate, monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 3, 4, 12, tzinfo=UTC)

    monkeypatch.setattr(budget, "datetime", FixedClock)
    monkeypatch.setattr(settings, "quality_repetition_gate_monthly_cny", 1)
    monkeypatch.setattr(settings, "quality_monthly_cny", 1)
    ident = await jobs.enqueue("repetition_gate", "test-" + uuid4().hex, uuid4().hex, {})
    assert await jobs.claim(job_id=ident) is None  # never on the primary executor
    job = await jobs.claim(job_id=ident, executor="repetition_gate")
    model = "dashscope/qwen3.5-flash"
    await budget.reserve(job["id"], model, 0.6, lease_token=job["lease_token"])
    with pytest.raises(budget.BudgetExhausted, match="monthly"):
        await budget.reserve(job["id"], model, 0.6, lease_token=job["lease_token"])
    other = await running()
    await budget.reserve(other["id"], model, 0.6, lease_token=other["lease_token"])
    with pytest.raises(budget.BudgetExhausted, match="monthly"):
        await budget.reserve(other["id"], model, 0.6, lease_token=other["lease_token"])
    await jobs.renew(job["id"], job["lease_token"])
    await jobs.finish(other, "completed", {})
    await jobs.finish(job, "completed", {})


async def test_case_import_is_immutable_and_bounded(gate, monkeypatch):
    _, _, database, case_set = gate
    with pytest.raises(ValueError, match="invalid_case_count"):
        await load(database, case_set, 0)
    await load(database, case_set, 3)
    with pytest.raises(ValueError, match="case_set_already_loaded"):
        await load(database, case_set, 1)
    monkeypatch.setattr(repetition_gate, "MAX_CASES", 2)
    with pytest.raises(ValueError, match="exceed_bound"):
        await repetition_gate.cases(case_set)


def test_block_amount_and_cost_bounds():
    job = {"snapshot": {"prices": {"m": PRICE}}}
    assert str(repetition_gate.block_amount("0.5")) == "0.50000000"
    for amount in ("0", "2.01", "-1"):
        with pytest.raises(ValueError):
            repetition_gate.block_amount(amount)
    usage = {"input": 10, "output": 0, "cached_input": 0, "calls": 1}
    assert repetition_gate.block_cost(job, "m", usage) == pytest.approx(0.000008)
    for bad in ({"cached_input": 11}, {"calls": 1.5}, {"unknown_cny": 3}, {"unknown_cny": "1"}):
        with pytest.raises(ValueError):
            repetition_gate.block_cost(job, "m", {**usage, **bad})


async def test_gate_disabled_and_budget_exhausted(gate, monkeypatch):
    client, identity, database, case_set = gate
    await load(database, case_set)
    begin = "/admin-api/quality-ci/repetition-gate/begin"
    body = {"mode": "calibrate", "case_set": case_set, "keys_configured": True}
    monkeypatch.setattr(settings, "quality_enabled", False)
    assert (await client.post(begin, json=body)).json()["reason"] == "quality_not_enabled"
    monkeypatch.setattr(settings, "quality_enabled", True)
    next_run(identity)
    job = (await client.post(begin, json=body)).json()["job"]
    monkeypatch.setattr(budget, "reserve", AsyncMock(side_effect=budget.BudgetExhausted("monthly")))
    response = await client.post(
        "/admin-api/quality-ci/repetition-gate/jobs/" + job["id"],
        json={"operation": "reserve_block", "lease_token": job["lease_token"],
              "data": {"model": "/".join(ROUTE), "amount": "2"}})
    assert response.status_code == 429
    await jobs.finish(job, "budget_exhausted", {})
