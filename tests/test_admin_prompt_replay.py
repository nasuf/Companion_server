from __future__ import annotations

def test_admin_prompt_replay_invokes_llm_only(api_client, monkeypatch):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    calls: list[tuple[object, str, dict]] = []
    model = object()

    async def fake_invoke_text(actual_model: object, prompt: str, **kwargs) -> str:
        calls.append((actual_model, prompt, kwargs))
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
    assert len(calls) == 1
    assert calls[0][0] is model
    assert calls[0][1] == "\n【任务】只做一次无副作用重跑\n"
    # Debug tool must hit the selected model itself — no silent Ollama fallback.
    profile = calls[0][2].get("profile")
    assert profile is not None and profile.allow_ollama_fallback is False


def test_replay_profile_is_no_fallback_no_retry():
    """守卫: replay 是调试工具, 必须打到所选模型本身 (不允许静默切 Ollama);
    失败直接报错, 管理员手动再点一次即可, 不做后台重试拖住 HTTP 请求。"""
    from app.api.admin.prompts import _REPLAY_PROFILE

    assert _REPLAY_PROFILE.allow_ollama_fallback is False
    assert _REPLAY_PROFILE.max_retries == 0
    assert _REPLAY_PROFILE.timeout_s >= 30


def test_admin_prompt_replay_invalid_role_returns_400(api_client, monkeypatch):
    """畸形 messages 必须报 400 参数错误, 不能被通用 502 重跑失败兜底吞掉。"""
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.get_chat_model", lambda: object())
    try:
        response = api_client.post(
            "/admin-api/prompts/replay",
            json={
                "prompt_key": "chat.system_base",
                "rendered_prompt": "SYSTEM",
                "model_kind": "chat",
                "messages": [
                    {"role": "tool", "content": "bad role"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 400
    assert "role is invalid" in response.json()["detail"]


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

    calls: list[tuple[list, dict]] = []
    model = object()

    async def fake_invoke_text(actual_model: object, prompt, **kwargs) -> str:
        assert actual_model is model
        calls.append((prompt, kwargs))
        return "stack 输出"

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.get_chat_model", lambda: model)
    monkeypatch.setattr("app.api.admin.prompts.invoke_text", fake_invoke_text)
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
    assert len(calls) == 1
    stack, kwargs = calls[0]
    # Message stack is converted to LangChain messages, order/content preserved.
    assert [m.content for m in stack] == ["SYSTEM", "用户原话"]
    assert [type(m).__name__ for m in stack] == ["SystemMessage", "HumanMessage"]
    # Messages path also goes through the resilient no-fallback replay profile.
    profile = kwargs.get("profile")
    assert profile is not None and profile.allow_ollama_fallback is False


def test_admin_prompt_replay_output_is_raw_llm_text(api_client, monkeypatch):
    """重跑输出必须是该次 LLM 的原始输出: 不剥时间戳前缀、不按 || 拆条、
    不移除 [EMO:] 标记 — 管理员靠原文判断提示词效果 (生产后处理在 UI 侧说明)。"""
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    raw = "[07-06 00:08] 第一条||第二条 [EMO:高兴/80]"

    async def fake_invoke_text(actual_model: object, prompt, **kwargs) -> str:
        return raw

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    monkeypatch.setattr("app.api.admin.prompts.get_chat_model", lambda: object())
    monkeypatch.setattr("app.api.admin.prompts.invoke_text", fake_invoke_text)
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
    assert response.json()["output"] == raw


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
