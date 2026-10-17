"""后台「聊天管理 → 主动交流」参数接口: 真实 SQL (行锁 / jsonb / 修改记录 / 缓存刷新)."""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma import Prisma

from app.api.admin import proactive_tuning as api
from app.services import runtime_config
from app.services.auth import create_jwt
from app.services.proactive import tuning


@pytest.fixture
async def env(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Proactive tuning API requires isolated PostgreSQL")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    db = Prisma(datasource={"url": url}, http={"trust_env": False})
    await db.connect()
    from app.config import settings

    monkeypatch.setattr(settings, "jwt_secret", "synthetic-proactive-tuning-test-secret")
    monkeypatch.setattr(api, "db", db)
    monkeypatch.setattr(runtime_config, "db", db)
    # 只测本接口: 其他配置字段的缓存保持测试前状态
    for name in ("_GLOBAL_CACHE", "_AGENT_CACHE", "_PRICING_CACHE", "_TTS_PRICING_CACHE",
                 "_PROMPTS_CACHE", "_CACHE_LOADED", "_CACHE_REVISION"):
        monkeypatch.setattr(runtime_config, name, getattr(runtime_config, name))
    admin = await db.user.create(data={"username": "tuning-admin-" + uuid4().hex, "role": "admin"})
    original = await db.query_raw("SELECT proactive_policy FROM system_config WHERE id = 1")
    app = FastAPI()
    app.include_router(api.router)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield db, client, admin
    finally:
        await client.aclose()
        await db.execute_raw("DELETE FROM proactive_tuning_changes WHERE admin_id = $1", admin.id)
        if original:
            policy = original[0].get("proactive_policy")
            await db.execute_raw(
                "UPDATE system_config SET proactive_policy = $1::jsonb WHERE id = 1",
                None if policy is None else json.dumps(policy),
            )
        await db.user.delete(where={"id": admin.id})
        await db.disconnect()


def _auth(user_id: str, role: str = "admin") -> dict:
    return {"Authorization": f"Bearer {create_jwt(user_id, role)}"}


def _param(payload: dict, key: str) -> dict:
    return next(p for p in payload["params"] if p["key"] == key)


async def test_get_lists_every_param_with_defaults(env):
    _db, client, admin = env
    resp = await client.get("/admin-api/proactive-tuning", headers=_auth(admin.id))
    assert resp.status_code == 200
    body = resp.json()
    assert {p["key"] for p in body["params"]} == set(tuning.PARAM_MAP)
    window = _param(body, "a_window_1_rate")
    assert window["default"] == 0.05 and window["label"] and window["description"]
    assert {g["key"] for g in body["groups"]} == {g.key for g in tuning.GROUPS}


async def test_put_saves_logs_and_takes_effect(env):
    db, client, admin = env
    resp = await client.put(
        "/admin-api/proactive-tuning",
        headers=_auth(admin.id),
        json={"values": {"followup_respect_active_hours": False, "a_window_1_rate": 0.5}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    window = _param(body, "a_window_1_rate")
    assert (window["value"], window["overridden"]) == (0.5, True)
    # 只存与默认不同的项
    stored = (await db.query_raw("SELECT proactive_policy FROM system_config WHERE id = 1"))[0]
    assert stored["proactive_policy"] == {"followup_respect_active_hours": False, "a_window_1_rate": 0.5}
    # 修改记录: 谁 / 改了什么 / 从多少到多少
    latest = body["changes"][0]
    assert latest["admin_name"] == admin.username
    assert latest["created_at"].endswith("+00:00")
    assert {(c["key"], c["before"], c["after"]) for c in latest["changes"]} == {
        ("followup_respect_active_hours", True, False), ("a_window_1_rate", 0.05, 0.5),
    }
    # 本 worker 立即生效 (缓存已重载)
    assert tuning.value("a_window_1_rate") == 0.5
    assert tuning.value("followup_respect_active_hours") is False

    # null = 恢复默认; 全部恢复后列写回 SQL NULL
    resp = await client.put(
        "/admin-api/proactive-tuning",
        headers=_auth(admin.id),
        json={"values": {"followup_respect_active_hours": None, "a_window_1_rate": None}},
    )
    assert resp.status_code == 200
    stored = (await db.query_raw("SELECT proactive_policy FROM system_config WHERE id = 1"))[0]
    assert stored["proactive_policy"] is None
    assert tuning.value("a_window_1_rate") == 0.05


async def test_invalid_patch_is_rejected_without_writing(env):
    db, client, admin = env
    before = await db.query_raw("SELECT count(*)::int AS n FROM proactive_tuning_changes")
    resp = await client.put(
        "/admin-api/proactive-tuning",
        headers=_auth(admin.id),
        json={"values": {"trigger_scheduled_scene": 0.9}},
    )
    assert resp.status_code == 400
    assert "合计" in resp.json()["detail"]
    after = await db.query_raw("SELECT count(*)::int AS n FROM proactive_tuning_changes")
    assert after == before


async def test_unchanged_values_write_nothing(env):
    db, client, admin = env
    resp = await client.put(
        "/admin-api/proactive-tuning", headers=_auth(admin.id), json={"values": {"daily_max": 3}},
    )
    assert resp.status_code == 200
    rows = await db.query_raw(
        "SELECT count(*)::int AS n FROM proactive_tuning_changes WHERE admin_id = $1", admin.id,
    )
    assert rows[0]["n"] == 0


async def test_non_admin_is_forbidden(env):
    _db, client, admin = env
    resp = await client.get("/admin-api/proactive-tuning", headers=_auth(admin.id, role="user"))
    assert resp.status_code == 403
