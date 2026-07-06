"""段落顺序汇总点 (prompting/section_order.py) 测试.

顺序此前只存在于 build_system_prompt 的代码调用顺序里; 现在收敛为
可管理配置. 这里锁定: 校验从严 / 读取归一化 / 装配尊重覆写 /
slot 登记与代码 _stage 调用一致 / cache 友好默认顺序不被误改.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _fake_redis(store: dict):
    r = AsyncMock()

    async def _get(key):
        return store.get(key)

    async def _set(key, value):
        store[key] = value

    async def _delete(key):
        store.pop(key, None)
        return 1

    r.get = AsyncMock(side_effect=_get)
    r.set = AsyncMock(side_effect=_set)
    r.delete = AsyncMock(side_effect=_delete)
    return r


@pytest.fixture
def so(monkeypatch):
    """section_order 模块 + fake redis/db 存储, 每测试前后清进程缓存."""
    from app.services.prompting import section_order as mod

    mod.invalidate_local_cache()
    redis_store: dict = {}
    rows: dict = {}

    async def _find_unique(where):
        return rows.get(where["promptKey"])

    async def _upsert(where, data):
        row = SimpleNamespace(
            orderJson=data["create"]["orderJson"],
            updatedAt="2026-07-06T00:00:00+00:00",
        )
        rows[where["promptKey"]] = row
        return row

    async def _delete(where):
        return rows.pop(where["promptKey"], None)

    fake_db = SimpleNamespace(
        promptsectionorder=SimpleNamespace(
            find_unique=AsyncMock(side_effect=_find_unique),
            upsert=AsyncMock(side_effect=_upsert),
            delete=AsyncMock(side_effect=_delete),
        ),
    )
    monkeypatch.setattr(mod, "db", fake_db)
    monkeypatch.setattr(mod, "get_redis", AsyncMock(return_value=_fake_redis(redis_store)))
    yield mod, redis_store, rows
    mod.invalidate_local_cache()


# ═══════════════════════════════════════════════════════════════════
# 存储层: 默认 / 校验 / 归一化 / 容错
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_default_order_when_unset(so):
    mod, _, _ = so
    assert await mod.get_chat_section_order() == mod.DEFAULT_CHAT_SECTION_ORDER


@pytest.mark.asyncio
async def test_set_rejects_invalid_orders(so):
    mod, _, _ = so
    with pytest.raises(ValueError, match="unknown"):
        await mod.set_chat_section_order([*mod.DEFAULT_CHAT_SECTION_ORDER, "nonexistent"])
    with pytest.raises(ValueError, match="duplicate"):
        await mod.set_chat_section_order(
            [*mod.DEFAULT_CHAT_SECTION_ORDER, mod.DEFAULT_CHAT_SECTION_ORDER[0]],
        )
    with pytest.raises(ValueError, match="missing"):
        await mod.set_chat_section_order(mod.DEFAULT_CHAT_SECTION_ORDER[:-1])


@pytest.mark.asyncio
async def test_set_then_get_roundtrip(so):
    mod, redis_store, rows = so
    custom = list(reversed(mod.DEFAULT_CHAT_SECTION_ORDER))
    info = await mod.set_chat_section_order(custom)
    assert info["order"] == custom
    assert info["source"] == "custom"
    assert rows[mod.CHAT_SECTION_ORDER_KEY] is not None
    # 写入后立即可读 (Redis + 本地缓存均已更新)
    assert await mod.get_chat_section_order() == custom
    # info 端点回读 custom
    info2 = await mod.get_chat_section_order_info()
    assert info2["order"] == custom and info2["source"] == "custom"


@pytest.mark.asyncio
async def test_reset_returns_to_default(so):
    mod, redis_store, rows = so
    await mod.set_chat_section_order(list(reversed(mod.DEFAULT_CHAT_SECTION_ORDER)))
    info = await mod.reset_chat_section_order()
    assert info["order"] == mod.DEFAULT_CHAT_SECTION_ORDER
    assert info["source"] == "default"
    assert mod.CHAT_SECTION_ORDER_KEY not in rows
    assert await mod.get_chat_section_order() == mod.DEFAULT_CHAT_SECTION_ORDER


@pytest.mark.asyncio
async def test_stale_stored_order_appends_new_slots(so):
    """代码后续新增 slot 时, 旧覆写在读取端自动补全 — 新 section 不静默消失."""
    mod, redis_store, _ = so
    import json
    stale = [s for s in mod.DEFAULT_CHAT_SECTION_ORDER if s != "ai_mood"]
    stale.insert(0, "ghost_slot")  # 已删除的 slot 应被丢弃
    redis_store[f"prompt_section_order:{mod.CHAT_SECTION_ORDER_KEY}"] = json.dumps(stale)
    order = await mod.get_chat_section_order()
    assert "ghost_slot" not in order
    assert order[-1] == "ai_mood"  # 缺失 slot 补到末尾
    assert set(order) == set(mod.DEFAULT_CHAT_SECTION_ORDER)


@pytest.mark.asyncio
async def test_get_falls_back_to_default_on_error(so, monkeypatch):
    """顺序配置任何一层坏掉都退默认 — 绝不让主回复挂掉."""
    mod, _, _ = so
    monkeypatch.setattr(mod, "get_redis", AsyncMock(side_effect=RuntimeError("redis down")))
    assert await mod.get_chat_section_order() == mod.DEFAULT_CHAT_SECTION_ORDER


# ═══════════════════════════════════════════════════════════════════
# 装配层: build_system_prompt 尊重覆写 + slot 登记守卫
# ═══════════════════════════════════════════════════════════════════


async def _prompt_text(key: str, **_kwargs) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    overrides = {
        "chat.system_base": "像朋友一样回复。",
        "chat.response_instruction": "回复自然口语化。",
    }
    if key in overrides:
        return overrides[key]
    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


@pytest.mark.asyncio
async def test_build_system_prompt_respects_custom_order(monkeypatch):
    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.section_order import DEFAULT_CHAT_SECTION_ORDER

    custom = [s for s in DEFAULT_CHAT_SECTION_ORDER if s != "response_instruction"]
    custom.insert(0, "response_instruction")  # 回复要求挪到最前

    async def _custom_order():
        return custom

    monkeypatch.setattr(
        "app.services.chat.prompt_builder.get_chat_section_order", _custom_order,
    )
    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
    ):
        diagnostics: dict = {}
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Nova", values={"gender": "female"}),
            memories=None,
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
            diagnostics=diagnostics,
        )

    assert "## 回复要求" in prompt and "## 核心规则" in prompt
    assert prompt.index("## 回复要求") < prompt.index("## 核心规则")
    assert diagnostics["section_order_source"] == "custom"


@pytest.mark.asyncio
async def test_build_system_prompt_default_order_unchanged():
    """默认顺序下核心段的相对顺序与重构前一致 (回复要求在最后)."""
    from app.services.chat.prompt_builder import build_system_prompt

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
    ):
        diagnostics: dict = {}
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Nova", values={"gender": "female"}),
            memories=None,
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
            diagnostics=diagnostics,
        )

    assert diagnostics["section_order_source"] == "default"
    idx_core = prompt.index("## 核心规则")
    idx_identity = prompt.index("## 你的身份")
    idx_response = prompt.index("## 回复要求")
    assert idx_core < idx_identity < idx_response
    assert prompt.rstrip().endswith(prompt[idx_response:].rstrip())


def test_stage_slots_match_registry():
    """build_system_prompt 里 _stage 的 slot 必须与 CHAT_SECTION_SLOTS 登记
    完全一致 — 新增 section 忘登记会被装配循环静默丢弃, 此守卫拦截."""
    import inspect

    from app.services.chat import prompt_builder
    from app.services.prompting.section_order import DEFAULT_CHAT_SECTION_ORDER

    src = inspect.getsource(prompt_builder.build_system_prompt)
    staged_slots = set(re.findall(r'_stage\(\s*\n?\s*"([a-z0-9_]+)"', src))
    assert staged_slots == set(DEFAULT_CHAT_SECTION_ORDER), (
        "staged slots 与 CHAT_SECTION_SLOTS 不一致: "
        f"only-in-code={staged_slots - set(DEFAULT_CHAT_SECTION_ORDER)}, "
        f"only-in-registry={set(DEFAULT_CHAT_SECTION_ORDER) - staged_slots}"
    )


def test_default_order_keeps_cache_friendly_prefix():
    """默认顺序的前 4 段必须是稳定段 (dashscope prefix cache 经济性),
    回复要求 (n 随机不可 cache) 必须在最后. 改动此顺序请先看
    build_system_prompt docstring 的 cache 说明."""
    from app.services.prompting.section_order import DEFAULT_CHAT_SECTION_ORDER

    assert DEFAULT_CHAT_SECTION_ORDER[:4] == [
        "core_rules", "anti_hallucination", "personality", "consistency",
    ]
    assert DEFAULT_CHAT_SECTION_ORDER[-1] == "response_instruction"


def test_slot_prompt_keys_are_registered():
    """slot 声明的 prompt_keys 必须存在于 registry (防拼写漂移)."""
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    from app.services.prompting.section_order import CHAT_SECTION_SLOTS

    for slot in CHAT_SECTION_SLOTS:
        for key in slot.prompt_keys:
            assert key in PROMPT_DEFINITION_MAP, f"slot {slot.slot} references unknown prompt key {key}"


# ═══════════════════════════════════════════════════════════════════
# Admin API
# ═══════════════════════════════════════════════════════════════════


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app


def test_api_get_section_order(api_client, monkeypatch):
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt
    from app.services.prompting.section_order import DEFAULT_CHAT_SECTION_ORDER

    _admin_override()
    try:
        response = api_client.get("/admin-api/prompts/section-order")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    data = response.json()
    assert data["prompt_key"] == "chat.system_base"
    assert data["default_order"] == DEFAULT_CHAT_SECTION_ORDER
    assert {s["slot"] for s in data["slots"]} == set(DEFAULT_CHAT_SECTION_ORDER)


def test_api_put_section_order_roundtrip(api_client, monkeypatch):
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt

    captured: list[list[str]] = []

    async def fake_set(order):
        captured.append(order)
        return {
            "prompt_key": "chat.system_base",
            "order": order,
            "default_order": order,
            "source": "custom",
            "updated_at": None,
            "slots": [],
        }

    monkeypatch.setattr("app.api.admin.prompts.set_chat_section_order", fake_set)
    _admin_override()
    try:
        response = api_client.put(
            "/admin-api/prompts/section-order",
            json={"order": ["core_rules", "response_instruction"]},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    assert captured == [["core_rules", "response_instruction"]]
    assert response.json()["source"] == "custom"


def test_api_put_section_order_invalid_returns_400(api_client, monkeypatch):
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt

    async def fake_set(order):
        raise ValueError("missing slots: memory")

    monkeypatch.setattr("app.api.admin.prompts.set_chat_section_order", fake_set)
    _admin_override()
    try:
        response = api_client.put(
            "/admin-api/prompts/section-order",
            json={"order": ["core_rules"]},
        )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 400
    assert "missing slots" in response.json()["detail"]


def test_api_delete_section_order(api_client, monkeypatch):
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt

    async def fake_reset():
        return {
            "prompt_key": "chat.system_base",
            "order": ["core_rules"],
            "default_order": ["core_rules"],
            "source": "default",
            "updated_at": None,
            "slots": [],
        }

    monkeypatch.setattr("app.api.admin.prompts.reset_chat_section_order", fake_reset)
    _admin_override()
    try:
        response = api_client.delete("/admin-api/prompts/section-order")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)

    assert response.status_code == 200
    assert response.json()["source"] == "default"
