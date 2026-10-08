"""Public status/schedule cache-miss regressions, with no provider or DB calls."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api.public import agents as api


@pytest.fixture
def subject(monkeypatch):
    agent = SimpleNamespace(id="agent-synthetic", name="Synthetic", status="active")
    schedule = [{"start": "00:00", "end": "24:00", "status": "home"}]
    read = AsyncMock(return_value=agent)
    cache = AsyncMock(return_value=None)
    overview = AsyncMock(return_value="synthetic life overview")
    generate = AsyncMock(return_value=schedule)
    monkeypatch.setattr(api, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=read)))
    monkeypatch.setattr(api, "get_cached_schedule", cache)
    monkeypatch.setattr(api, "get_life_overview", overview)
    monkeypatch.setattr(api, "get_mbti", lambda _: {"EI": 72})
    # A missing API import must still produce a real cache-miss 500 in the old
    # code, rather than be hidden by installing a previously undefined global.
    if hasattr(api, "generate_daily_schedule"):
        monkeypatch.setattr(api, "generate_daily_schedule", generate)
    monkeypatch.setattr(api, "get_current_status", lambda _: {"status": "home", "type": "rest"})
    monkeypatch.setattr(api, "status_label", lambda _: "在家")
    monkeypatch.setattr(api, "type_label", lambda _: "休息")
    app = FastAPI()
    app.include_router(api.router)

    async def owner():
        return agent

    app.dependency_overrides[api.require_agent_owner] = owner
    return SimpleNamespace(app=app, agent=agent, schedule=schedule, read=read,
                           cache=cache, overview=overview, generate=generate)


async def request(subject, suffix):
    transport = httpx.ASGITransport(app=subject.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://isolated") as client:
        return await client.get(f"/agents/{subject.agent.id}/{suffix}")


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["schedule", "status"])
@pytest.mark.parametrize("cached", [False, True])
async def test_public_schedule_and_status_cache_hit_and_miss(subject, suffix, cached):
    subject.cache.return_value = subject.schedule if cached else None
    response = await request(subject, suffix)
    assert response.status_code == 200
    expected = {"agent_id": subject.agent.id}
    if suffix == "schedule":
        expected["schedule"] = subject.schedule
    else:
        expected.update(status="home", type="rest", status_label="在家", type_label="休息")
    assert response.json() == expected
    if cached:
        subject.generate.assert_not_awaited()
        subject.overview.assert_not_awaited()
    else:
        subject.generate.assert_awaited_once_with(subject.agent.id, "Synthetic", {"EI": 72},
                                                 life_overview="synthetic life overview")
        subject.overview.assert_awaited_once_with(subject.agent.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [None, "archived", "provisioning"])
async def test_missing_or_inactive_agent_cannot_generate_schedule(subject, state):
    subject.read.return_value = None if state is None else SimpleNamespace(status=state)
    response = await request(subject, "status")
    assert response.status_code == 404
    subject.cache.assert_not_awaited()
    subject.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["schedule", "status"])
async def test_ownership_denial_prevents_reads_and_generation(subject, suffix):
    async def deny():
        raise HTTPException(status_code=403, detail="Not your agent")

    subject.app.dependency_overrides[api.require_agent_owner] = deny
    response = await request(subject, suffix)
    assert response.status_code == 403
    subject.read.assert_not_awaited()
    subject.cache.assert_not_awaited()
    subject.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generation_failure_is_not_reported_as_a_successful_status(subject):
    subject.generate.side_effect = RuntimeError("synthetic generation failure")
    response = await request(subject, "status")
    assert response.status_code == 500
    subject.generate.assert_awaited_once()
