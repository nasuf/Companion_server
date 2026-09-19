"""Admin template create: manual/random name & career, plus dual-random batch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.jwt_auth import require_admin_jwt

_PERSONALITY = {
    "lively": 50,
    "rational": 50,
    "emotional": 50,
    "planned": 50,
    "spontaneous": 50,
    "creative": 50,
    "humor": 50,
}


def _admin_override():
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {
        "sub": "admin-1",
        "role": "admin",
    }
    return app


def _agent(agent_id: str, name: str):
    return SimpleNamespace(id=agent_id, name=name, status="provisioning"), SimpleNamespace(
        id=f"ws-{agent_id}",
    )


def test_legacy_create_is_manual_name_random_career(api_client):
    app = _admin_override()
    created = []

    async def _create(**kwargs):
        created.append(kwargs)
        return _agent("tpl-1", kwargs["name"])

    try:
        with (
            patch(
                "app.api.admin.agent_templates.get_or_create_template_user",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="owner-1"),
            ),
            patch(
                "app.api.public.agents.create_agent_with_provisioning",
                new_callable=AsyncMock,
                side_effect=_create,
            ),
            patch(
                "app.services.career.pick_random_active_careers",
                new_callable=AsyncMock,
                return_value=[{"id": "c1", "title": "咖啡师"}],
            ),
        ):
            response = api_client.post(
                "/admin-api/agent-templates",
                json={"name": "柳如烟", "gender": "female", "personality": _PERSONALITY},
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "tpl-1"
    assert body["name"] == "柳如烟"
    assert created[0]["career_template_override"]["title"] == "咖啡师"
    assert created[0]["gender"] == "female"


def test_random_name_manual_career(api_client):
    app = _admin_override()
    created = []

    async def _create(**kwargs):
        created.append(kwargs)
        return _agent("tpl-2", kwargs["name"])

    try:
        with (
            patch(
                "app.api.admin.agent_templates.get_or_create_template_user",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="owner-1"),
            ),
            patch(
                "app.api.admin.agent_templates.list_template_agents",
                new_callable=AsyncMock,
                return_value=[SimpleNamespace(name="已有模板")],
            ),
            patch(
                "app.services.name_templates.pick_random_names",
                new_callable=AsyncMock,
                return_value=[{"name": "陈砚", "nickname": "阿砚", "gender": "male"}],
            ) as pick_names,
            patch(
                "app.services.career.get_active_career_by_id",
                new_callable=AsyncMock,
                return_value={"id": "career-9", "title": "室内设计师"},
            ),
            patch(
                "app.api.public.agents.create_agent_with_provisioning",
                new_callable=AsyncMock,
                side_effect=_create,
            ),
        ):
            response = api_client.post(
                "/admin-api/agent-templates",
                json={
                    "name_mode": "random",
                    "career_mode": "manual",
                    "career_id": "career-9",
                    "gender": "male",
                    "personality": _PERSONALITY,
                },
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "陈砚"
    assert created[0]["career_template_override"]["title"] == "室内设计师"
    pick_names.assert_awaited_once()
    assert pick_names.await_args.kwargs["exclude"] == {"已有模板"}


def test_batch_requires_both_random(api_client):
    app = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/agent-templates",
            json={
                "name": "柳如烟",
                "name_mode": "manual",
                "career_mode": "random",
                "batch_count": 3,
                "gender": "female",
                "personality": _PERSONALITY,
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
    assert response.status_code == 400
    assert "批量" in response.json()["detail"]


def test_dual_random_batch_creates_unique_names(api_client):
    app = _admin_override()
    created = []

    async def _create(**kwargs):
        created.append(kwargs)
        return _agent(f"tpl-{len(created)+1}", kwargs["name"])

    try:
        with (
            patch(
                "app.api.admin.agent_templates.get_or_create_template_user",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="owner-1"),
            ),
            patch(
                "app.api.admin.agent_templates.list_template_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.name_templates.pick_random_names",
                new_callable=AsyncMock,
                return_value=[
                    {"name": "陈砚"},
                    {"name": "吴时远"},
                    {"name": "赵叙"},
                ],
            ),
            patch(
                "app.services.career.pick_random_active_careers",
                new_callable=AsyncMock,
                return_value=[
                    {"title": "咖啡师"},
                    {"title": "花艺师"},
                    {"title": "烘焙师"},
                ],
            ),
            patch(
                "app.api.public.agents.create_agent_with_provisioning",
                new_callable=AsyncMock,
                side_effect=_create,
            ),
        ):
            response = api_client.post(
                "/admin-api/agent-templates",
                json={
                    "name_mode": "random",
                    "career_mode": "random",
                    "batch_count": 3,
                    "gender": "male",
                    "personality": _PERSONALITY,
                },
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert [t["name"] for t in body["templates"]] == ["陈砚", "吴时远", "赵叙"]
    assert [c["career_template_override"]["title"] for c in created] == [
        "咖啡师",
        "花艺师",
        "烘焙师",
    ]


def test_dual_random_batch_keeps_partial_success(api_client):
    from fastapi import HTTPException

    app = _admin_override()
    created = []

    async def _create(**kwargs):
        if kwargs["name"] == "陈砚":
            raise HTTPException(status_code=409, detail="conflict")
        created.append(kwargs)
        return _agent(f"tpl-{kwargs['name']}", kwargs["name"])

    try:
        with (
            patch(
                "app.api.admin.agent_templates.get_or_create_template_user",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="owner-1"),
            ),
            patch(
                "app.api.admin.agent_templates.list_template_agents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.name_templates.pick_random_names",
                new_callable=AsyncMock,
                return_value=[{"name": "陈砚"}, {"name": "吴时远"}],
            ),
            patch(
                "app.services.career.pick_random_active_careers",
                new_callable=AsyncMock,
                return_value=[{"title": "咖啡师"}, {"title": "花艺师"}],
            ),
            patch(
                "app.api.public.agents.create_agent_with_provisioning",
                new_callable=AsyncMock,
                side_effect=_create,
            ),
        ):
            response = api_client.post(
                "/admin-api/agent-templates",
                json={
                    "name_mode": "random",
                    "career_mode": "random",
                    "batch_count": 2,
                    "gender": "male",
                    "personality": _PERSONALITY,
                },
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["templates"][0]["name"] == "吴时远"
    assert body["errors"][0]["name"] == "陈砚"
    app = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/agent-templates",
            json={
                "name_mode": "random",
                "career_mode": "random",
                "personality": _PERSONALITY,
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
    assert response.status_code == 400
    assert "性别" in response.json()["detail"]


def test_manual_name_blank_is_rejected(api_client):
    app = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/agent-templates",
            json={"name": "  ", "personality": _PERSONALITY, "gender": "female"},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
    assert response.status_code == 400
    assert "名称" in response.json()["detail"]
