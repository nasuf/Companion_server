from __future__ import annotations

def test_admin_prompt_replay_invokes_llm_only(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    calls: list[tuple[object, str]] = []
    model = object()

    async def fake_invoke_text(actual_model: object, prompt: str) -> str:
        calls.append((actual_model, prompt))
        return "重跑输出"

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.get_utility_model", lambda: model)
    monkeypatch.setattr("app.api.admin.prompts.invoke_text", fake_invoke_text)
    try:
        response = api_client.post(
            "/admin-api/prompts/replay",
            json={
                "prompt_key": "memory.relevance",
                "rendered_prompt": "\n【任务】只做一次无副作用重跑\n",
                "model_kind": "utility",
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    assert response.json()["output"] == "重跑输出"
    assert response.json()["rendered_prompt"] == "\n【任务】只做一次无副作用重跑\n"
    assert calls == [(model, "\n【任务】只做一次无副作用重跑\n")]


def test_admin_prompt_replay_rejects_unknown_prompt(api_client):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        response = api_client.post(
            "/admin-api/prompts/replay",
            json={
                "prompt_key": "missing.prompt",
                "rendered_prompt": "hello",
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 404


def test_admin_prompt_replay_can_replay_original_message_stack(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    calls: list[list[dict[str, str]]] = []

    class FakeModel:
        async def ainvoke(self, messages):
            calls.append([{"role": getattr(m, "type", ""), "content": m.content} for m in messages])
            return type("Result", (), {"content": "stack 输出"})()

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.get_chat_model", lambda: FakeModel())
    try:
        response = api_client.post(
            "/admin-api/prompts/replay",
            json={
                "prompt_key": "chat.system_base",
                "rendered_prompt": "SYSTEM",
                "model_kind": "chat",
                "messages": [
                    {"role": "system", "content": "SYSTEM"},
                    {"role": "user", "content": "用户原话"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    assert response.json()["output"] == "stack 输出"
    assert calls and [m["content"] for m in calls[0]] == ["SYSTEM", "用户原话"]


def test_admin_prompt_canary_update_exposes_eval_result(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    async def fake_update(*_args, **_kwargs):
        return {
            "prompt_key": "chat.system_base",
            "is_enabled": True,
            "mode": "agents",
            "content": "canary",
            "agent_ids": ["agent-1"],
            "rollout_percent": 0,
            "eval_result": {"ok": True},
            "updated_at": "2026-05-19T00:00:00+00:00",
        }

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.update_prompt_canary_config", fake_update)
    try:
        response = api_client.put(
            "/admin-api/prompts/chat.system_base/canary",
            json={
                "is_enabled": True,
                "mode": "agents",
                "content": "canary",
                "agent_ids": ["agent-1"],
                "rollout_percent": 0,
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    data = response.json()
    assert data["is_enabled"] is True
    assert data["eval_result"]["ok"] is True


def test_admin_prompt_enabled_endpoint(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    calls: list[tuple[str, bool]] = []

    async def fake_set_prompt_enabled(key: str, enabled: bool) -> dict:
        calls.append((key, enabled))
        return {
            "key": key,
            "title": "回忆相关度判断",
            "stage": "日常交流",
            "category": "记忆",
            "description": "d",
            "default_text": "t",
            "content": "t",
            "is_enabled": enabled,
            "canary_config": None,
            "updated_at": None,
            "source": "redis",
        }

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.set_prompt_enabled", fake_set_prompt_enabled)
    try:
        response = api_client.put(
            "/admin-api/prompts/memory.relevance/enabled",
            json={"is_enabled": False},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    assert response.json()["is_enabled"] is False
    assert calls == [("memory.relevance", False)]


def test_admin_prompt_update_conflict_returns_409(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app
    from app.services.prompting.store import PromptUpdateConflictError

    async def fake_update_prompt_text(key: str, content: str, *, expected_updated_at=None):
        raise PromptUpdateConflictError("concurrent edit")

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.update_prompt_text", fake_update_prompt_text)
    try:
        response = api_client.put(
            "/admin-api/prompts/memory.relevance",
            json={"content": "new", "expected_updated_at": "2020-01-01T00:00:00+00:00"},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 409
