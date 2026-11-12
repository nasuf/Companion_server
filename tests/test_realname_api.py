"""实名门槛的接口层: 后台策略 / 统计 / 重置, 用户端状态与核验, 各聊天入口的拦截。"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.services.identity import realname

ADULT_ID = "11010519491231002X"


class PolicyDB:
    """system_config.realname_policy + realname_policy_changes + users, in memory."""

    def __init__(self, stored=None, users=("u1", "u2")):
        self.stored = stored
        self.changes: list[dict] = []
        self.users = set(users)

    async def query_raw(self, sql, *args):
        if "SELECT realname_policy FROM system_config" in sql:
            return [{"realname_policy": self.stored}]
        if "FROM realname_policy_changes" in sql:
            return [{"id": c["id"], "created_at": "2026-11-12T08:00:00", "changes": json.dumps(c["changes"]),
                     "admin_id": c["admin_id"], "admin_name": None} for c in self.changes]
        if "FROM users u LEFT JOIN user_identity_verifications" in sql:
            return [{"id": i, "username": "n-" + i, "display_name": None, "realname_status": None}
                    for i in args[0] if i in self.users]
        if "SELECT id FROM users WHERE id = ANY" in sql:
            return [{"id": i} for i in args[0] if i in self.users]
        raise AssertionError(sql)

    async def execute_raw(self, sql, *args):
        if sql.startswith("UPDATE system_config"):
            self.stored = json.loads(args[0])
        elif "INSERT INTO realname_policy_changes" in sql:
            self.changes.append({"id": args[0], "admin_id": args[1], "changes": json.loads(args[2])})
        return 1

    @asynccontextmanager
    async def tx(self):
        yield self


@pytest.fixture
def policy_api(monkeypatch):
    from app.api.admin import realname_policy as api
    from app.config import settings
    from app.services import runtime_config

    db = PolicyDB()
    monkeypatch.setattr(api, "db", db)
    monkeypatch.setattr(settings, "realname_mode", "off")
    monkeypatch.setattr(settings, "realname_mock_enabled", False)
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "id")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "key")

    async def reload():
        runtime_config._GLOBAL_CACHE = {"realnamePolicy": db.stored}

    monkeypatch.setattr(api, "load_caches", reload)
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    return SimpleNamespace(api=api, db=db, settings=settings)


@pytest.mark.asyncio
async def test_policy_save_reads_back_and_records_change(policy_api):
    api, db = policy_api.api, policy_api.db
    first = await api.get_policy()
    assert first["stored"] is None and first["effective"]["source"] == "env"
    out = await api.put_policy(api.PolicyPut(mode="id_card", scope="users", user_ids=[" u1 ", "u1"], expected={}),
                               admin={"sub": "adm"})
    assert out["stored"] == {"mode": "id_card", "scope": "users", "user_ids": ["u1"]}
    assert out["effective"]["source"] == "admin" and [u["id"] for u in out["users"]] == ["u1"]
    assert db.changes[0]["changes"] == {"before": None, "after": out["stored"]}
    assert out["changes"][0]["changes"]["after"]["mode"] == "id_card"
    assert realname.policy().applies_to("u1") and not realname.policy().applies_to("u2")
    # 原样再存: 不写库、不记修改记录
    again = await api.put_policy(api.PolicyPut(mode="id_card", scope="users", user_ids=["u1"], expected=out["stored"]),
                                 admin={"sub": "adm"})
    assert len(db.changes) == 1 and again["stored"] == out["stored"]


@pytest.mark.asyncio
async def test_policy_rejections(policy_api):
    api, db, settings = policy_api.api, policy_api.db, policy_api.settings
    with pytest.raises(HTTPException) as e:
        await api.put_policy(api.PolicyPut(mode="phone", scope="users", user_ids=[], expected={}), admin={})
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        await api.put_policy(api.PolicyPut(mode="phone", scope="users", user_ids=["ghost"], expected={}), admin={})
    assert e.value.status_code == 400 and "ghost" in e.value.detail
    settings.tencent_realname_secret_id = ""
    with pytest.raises(HTTPException) as e:
        await api.put_policy(api.PolicyPut(mode="id_card", scope="all", expected={}), admin={})
    assert e.value.status_code == 400
    settings.tencent_realname_secret_id = "id"
    db.stored = {"mode": "phone", "scope": "all", "user_ids": []}
    with pytest.raises(HTTPException) as e:  # 读取后被别人改过
        await api.put_policy(api.PolicyPut(mode="off", scope="all", expected={}), admin={})
    assert e.value.status_code == 409
    assert db.changes == []


@pytest.mark.asyncio
async def test_policy_requires_initialized_config(policy_api, monkeypatch):
    api = policy_api.api

    async def empty(sql, *args):
        return []

    monkeypatch.setattr(policy_api.db, "query_raw", empty)
    with pytest.raises(HTTPException) as e:
        await api.get_policy()
    assert e.value.status_code == 503
    with pytest.raises(HTTPException) as e:
        await api.put_policy(api.PolicyPut(mode="off", scope="all", expected={}), admin={})
    assert e.value.status_code == 503


@pytest.mark.asyncio
async def test_summary_distributions(monkeypatch):
    from app.config import settings
    from app.services import runtime_config

    rows = {
        "active_users": [{"active_users": 5, "phone_bound": 3}],
        "status": [{"status": "verified", "n": 3}, {"status": "minor", "n": 1}],
        "failure": [{"reason": None, "n": 2}],
        "profile": [{"gender": "male", "province": "广东", "age": 22},
                    {"gender": "female", "province": None, "age": 31},
                    {"gender": None, "province": "广东", "age": None},
                    {"gender": "male", "province": "北京", "age": 55}],
    }

    async def query_raw(sql, *args):
        if "phone_bound" in sql:
            return rows["active_users"]
        if "GROUP BY v.status" in sql:
            return rows["status"]
        if "failure_reason AS reason" in sql:
            assert args == ("90",)  # 窗口被钳到 90 天
            return rows["failure"]
        return rows["profile"]

    monkeypatch.setattr(realname, "db", SimpleNamespace(query_raw=query_raw))
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    monkeypatch.setattr(settings, "realname_mode", "phone")
    out = await realname.summary(days=365)
    assert out["mode"] == "phone" and out["policy"]["source"] == "env"
    assert out["genders"] == {"male": 2, "female": 1}
    assert out["age_buckets"] == {"18-24": 1, "30-34": 1, "50+": 1}
    assert out["provinces"][0] == {"province": "广东", "count": 2}
    assert {"province": "未知", "count": 1} in out["provinces"]
    assert out["recent_failures"] == [{"reason": "unknown", "count": 2}]
    assert out["statuses"] == {"verified": 3, "minor": 1} and out["window_days"] == 90
    assert [realname._age_bucket(a) for a in (18, 25, 30, 35, 40, 49, 50)] == [
        "18-24", "25-29", "30-34", "35-39", "40-49", "40-49", "50+"]


def test_helpers_edge_cases(monkeypatch):
    from app.config import settings

    assert realname.Policy("phone", "users", frozenset({"b", "a"})).public()["user_ids"] == ["a", "b"]
    first17 = "11010518991231123"
    old = first17 + realname._CHECK[sum(int(d) * w for d, w in zip(first17, realname._WEIGHTS)) % 11]
    assert realname.normalize_id_number(old) is None  # 1900 年以前
    with pytest.raises(ValueError):
        realname.decrypt("v0:abc")
    assert isinstance(realname._today(), date)
    monkeypatch.setattr(settings, "realname_hash_key", "")
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(realname.RealnameError, match="not_configured"):
        realname.id_digest(ADULT_ID)


@pytest.mark.asyncio
async def test_provider_transport_failure_and_unexpected_db_error(monkeypatch):
    import httpx
    from app.config import settings

    real_client = httpx.AsyncClient

    def boom(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(realname.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(boom)))
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "AKID")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "secret")
    with pytest.raises(realname.RealnameError, match="provider_unavailable"):
        await realname._tencent_check("张三", ADULT_ID)

    async def broken(*a, **kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(realname, "_record", broken)
    with pytest.raises(RuntimeError, match="connection reset"):  # 不是唯一冲突就原样抛出
        await realname._record_claim("u1", status="verified", name="张三", digest="d", adult=True,
                                     id_number=ADULT_ID)


@pytest.mark.asyncio
async def test_admin_user_endpoints(monkeypatch):
    from app.api.admin import users

    monkeypatch.setattr(realname, "summary", AsyncMock(return_value={"mode": "off"}))
    monkeypatch.setattr(realname, "admin_reset", AsyncMock(side_effect=[1, 0]))
    assert await users.realname_summary(days=7, _={}) == {"mode": "off"}
    realname.summary.assert_awaited_once_with(days=7)
    assert await users.realname_reset("u1", claims={}) == {"reset": True}
    assert await users.realname_reset("u2", claims={}) == {"reset": False}


@pytest.mark.asyncio
async def test_user_status_and_verify_endpoints(monkeypatch):
    from app.api.public import auth
    from app.models.auth import RealnameVerifyRequest

    state = realname.RealnameState("id_card", True, None)
    monkeypatch.setattr(realname, "load_state", AsyncMock(return_value=state))
    monkeypatch.setattr(auth, "audit_auth_request_event", MagicMock())
    user = {"sub": "u1", "role": "user"}
    assert await auth.realname_status(payload=user) == {"mode": "id_card", "status": None, "required": ["id_card"]}

    body = RealnameVerifyRequest(name="张三", id_number=ADULT_ID, consent=False)
    with pytest.raises(HTTPException) as e:
        await auth.realname_verify(body, request=MagicMock(), payload=user)
    assert e.value.status_code == 400

    body = RealnameVerifyRequest(name="张三", id_number=ADULT_ID, consent=True)
    monkeypatch.setattr(realname, "verify", AsyncMock(side_effect=realname.RealnameError("mismatch")))
    with pytest.raises(HTTPException) as e:
        await auth.realname_verify(body, request=MagicMock(), payload=user)
    assert (e.value.status_code, e.value.detail) == (400, "姓名与身份证号不一致")
    monkeypatch.setattr(realname, "verify", AsyncMock(side_effect=realname.RealnameError("brand_new")))
    with pytest.raises(HTTPException) as e:
        await auth.realname_verify(body, request=MagicMock(), payload=user)
    assert e.value.detail == "实名认证失败"

    monkeypatch.setattr(realname, "verify", AsyncMock(return_value="verified"))
    monkeypatch.setattr(realname, "load_state", AsyncMock(return_value=realname.RealnameState("id_card", True, "verified")))
    assert (await auth.realname_verify(body, request=MagicMock(), payload=user))["required"] == []
    auth.audit_auth_request_event.assert_called_with(
        "realname_verify_verified", auth.audit_auth_request_event.call_args.args[1], user_id="u1", outcome="verified")
