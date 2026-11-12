"""自助注销: 7 天冷静期, 登录撤销, 到期删除; 冷静期内不发主动消息和推送。"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.services import account_deletion as ad


class Users:
    def __init__(self):
        self.rows = {
            "u1": dict(id="u1", status="active", role="user", username="u1", due=None),
            "adm": dict(id="adm", status="active", role="admin", username="adm", due=None),
        }
        self.phones = {"u1": "13800000000"}

    async def find_unique(self, *, where):
        r = self.rows.get(where["id"])
        return SimpleNamespace(id=r["id"], status=r["status"], role=r["role"], username=r["username"],
                               deletionDueAt=r["due"]) if r else None

    async def find_first(self, *, where):
        p = self.phones.get(where["userId"])
        return SimpleNamespace(providerAccountId=p) if p else None

    async def execute_raw(self, sql, *args):
        assert sql.startswith("UPDATE users SET deletion_requested_at")
        self.rows[args[0]]["due"] = datetime.fromisoformat(args[2])
        return 1

    async def query_raw(self, sql, *args):
        if sql.startswith("UPDATE users SET deletion_requested_at=NULL"):
            r = self.rows[args[0]]
            if r["due"] is None:
                return []
            r["due"] = None
            return [{"id": args[0]}]
        if sql.startswith("SELECT deletion_due_at IS NOT NULL AS pending"):
            r = self.rows.get(args[0])
            return [{"pending": r["due"] is not None}] if r else []
        if sql.startswith("SELECT id FROM users WHERE deletion_due_at"):
            now = datetime.fromisoformat(args[0])
            return [{"id": r["id"]} for r in self.rows.values()
                    if r["due"] and r["due"] <= now and r["role"] != "admin"]
        raise AssertionError(sql)


@pytest.fixture
def users(monkeypatch):
    u = Users()
    db = SimpleNamespace(user=SimpleNamespace(find_unique=u.find_unique),
                         authidentity=SimpleNamespace(find_first=u.find_first),
                         execute_raw=u.execute_raw, query_raw=u.query_raw)
    monkeypatch.setattr(ad, "db", db)
    monkeypatch.setattr(ad, "_disconnect", AsyncMock())
    monkeypatch.setattr("app.services.sms.verify_code", AsyncMock(side_effect=lambda p, c: c == "123456"))
    return u


@pytest.mark.asyncio
async def test_request_needs_code_then_login_cancels(users):
    with pytest.raises(ad.DeletionError, match="bad_code"):
        await ad.request_deletion("u1", code="000000")
    assert users.rows["u1"]["due"] is None
    due = await ad.request_deletion("u1", code="123456")
    assert timedelta(days=6, hours=23) < due - ad._now() <= timedelta(days=7)
    ad._disconnect.assert_awaited_once_with("u1")
    assert await ad.pending("u1")
    user = await users.find_unique(where={"id": "u1"})
    assert await ad.cancel_on_login(user) is True
    assert not await ad.pending("u1")
    assert await ad.cancel_on_login(await users.find_unique(where={"id": "u1"})) is False


@pytest.mark.asyncio
async def test_protected_and_unavailable_accounts(users):
    with pytest.raises(ad.DeletionError, match="protected"):
        await ad.request_deletion("adm", code=None)
    users.rows["u1"]["status"] = "deleting"
    with pytest.raises(ad.DeletionError, match="unavailable"):
        await ad.request_deletion("u1", code="123456")
    with pytest.raises(ad.DeletionError, match="unavailable"):
        await ad.request_deletion("ghost", code=None)


@pytest.mark.asyncio
async def test_account_without_phone_needs_no_code(users):
    users.phones.pop("u1")
    await ad.request_deletion("u1", code=None)
    assert await ad.pending("u1")


@pytest.mark.asyncio
async def test_purge_only_due_and_retries_failures(users, monkeypatch):
    from app.services.runtime import data_reset

    users.rows["u1"]["due"] = ad._now() - timedelta(minutes=1)
    users.rows["u2"] = dict(id="u2", status="active", role="user", username="u2", due=ad._now() + timedelta(days=3))
    users.rows["u3"] = dict(id="u3", status="active", role="user", username="u3", due=ad._now() - timedelta(hours=1))
    calls = []

    async def hard_delete(uid):
        calls.append(uid)
        if uid == "u3":
            raise RuntimeError("erasure residual")

    monkeypatch.setattr(data_reset, "hard_delete_user_data", hard_delete)
    stats = await ad.purge_due()
    assert stats == {"due": 2, "deleted": 1, "failed": 1}
    assert sorted(calls) == ["u1", "u3"]


@pytest.mark.asyncio
async def test_disconnect_failure_is_tolerated(monkeypatch):
    from app.services.runtime import ws_manager
    from app.services.notifications import presence

    monkeypatch.setattr(ws_manager.manager, "revoke_user", AsyncMock(side_effect=RuntimeError("redis")))
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock())
    monkeypatch.setattr(presence, "remove_online", AsyncMock())
    await ad._disconnect("u1")
    presence.remove_online.assert_awaited_once_with("u1")


@pytest.mark.asyncio
async def test_endpoint_maps_errors(monkeypatch):
    from unittest.mock import MagicMock
    from app.api.public import auth
    from app.models.auth import AccountDeletionRequest

    monkeypatch.setattr(auth, "audit_auth_request_event", MagicMock())
    monkeypatch.setattr(ad, "request_deletion", AsyncMock(side_effect=ad.DeletionError("bad_code")))
    with pytest.raises(HTTPException) as e:
        await auth.request_account_deletion(AccountDeletionRequest(code="1"), request=MagicMock(),
                                            payload={"sub": "u1"})
    assert (e.value.status_code, e.value.detail) == (400, "验证码错误或已过期")
    monkeypatch.setattr(ad, "request_deletion", AsyncMock(side_effect=ad.DeletionError("weird")))
    with pytest.raises(HTTPException) as e:
        await auth.request_account_deletion(AccountDeletionRequest(), request=MagicMock(), payload={"sub": "u1"})
    assert e.value.detail == "注销申请失败"
    monkeypatch.setattr(ad, "request_deletion", AsyncMock(return_value=datetime(2026, 11, 20, 1, 2, 3)))
    out = await auth.request_account_deletion(AccountDeletionRequest(code="123456"), request=MagicMock(),
                                              payload={"sub": "u1"})
    assert out == {"deletion_due_at": "2026-11-20T01:02:03Z", "cooldown_days": 7}


@pytest.mark.asyncio
async def test_proactive_and_push_suppressed_during_cooldown(monkeypatch):
    from app.services.proactive import gates
    from app.services.identity import realname
    from app.services.notifications import dispatcher

    monkeypatch.setattr(gates.db, "query_raw", AsyncMock(return_value=[{"status": "active"}]))
    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value=None))
    monkeypatch.setattr(ad, "pending", AsyncMock(return_value=True))
    gate = await gates.check_workspace(SimpleNamespace(workspace_id="w", user_id="u1"))
    assert (gate.action, gate.reason) == ("stop", "deletion_pending")

    marked = []
    monkeypatch.setattr(dispatcher, "_mark_status", AsyncMock(side_effect=lambda *a: marked.append(a)))
    monkeypatch.setattr(dispatcher, "is_user_foreground", AsyncMock(return_value=False))
    await dispatcher._dispatch_one({"id": "e1", "userId": "u1", "workspaceId": "w", "conversationId": "c",
                                    "type": "chat_reply"})
    assert marked == [("e1", "suppressed", "deletion_pending")]


@pytest.mark.asyncio
async def test_pending_is_false_for_unrelated_rows_and_unknown_users(users):
    assert await ad.pending("ghost") is False
    users.rows["u1"]["due"] = None
    assert await ad.pending("u1") is False


@pytest.mark.asyncio
async def test_push_and_gate_pass_when_not_pending(monkeypatch):
    from app.services.proactive import gates
    from app.services.identity import realname
    from app.services.notifications import dispatcher

    monkeypatch.setattr(gates.db, "query_raw", AsyncMock(return_value=[{"status": "active"}]))
    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value=None))
    monkeypatch.setattr(ad, "pending", AsyncMock(return_value=False))
    assert await gates.check_workspace(SimpleNamespace(workspace_id="w", user_id="u1")) is None
    marked = []
    monkeypatch.setattr(dispatcher, "_mark_status", AsyncMock(side_effect=lambda *a: marked.append(a)))
    monkeypatch.setattr(dispatcher, "is_user_foreground", AsyncMock(return_value=True))
    await dispatcher._dispatch_one({"id": "e2", "userId": "u1", "workspaceId": "w", "conversationId": "c",
                                    "type": "chat_reply"})
    assert marked == [("e2", "suppressed", "app_foreground")]


@pytest.mark.asyncio
async def test_scheduler_job_logs_and_survives_failure(monkeypatch):
    import jobs.scheduler as sch

    async def run(name, ttl, body):
        assert (name, ttl) == ("account_deletion_purge", 1800)
        await body()

    monkeypatch.setattr(sch, "_run_distributed_job", run)
    monkeypatch.setattr(sch, "purge_due_account_deletions", AsyncMock(return_value={"due": 1, "deleted": 1, "failed": 0}))
    await sch._run_account_deletion_purge()
    failed = []
    monkeypatch.setattr(sch, "_job_failed", lambda label, exc: failed.append(label))
    monkeypatch.setattr(sch, "purge_due_account_deletions", AsyncMock(side_effect=RuntimeError("db")))
    await sch._run_account_deletion_purge()
    assert failed == ["Account deletion purge"]
