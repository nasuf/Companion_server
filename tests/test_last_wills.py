from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.public import last_wills
from app.models.last_will import LastWillContact, LastWillCreate, LastWillUpdate
from app.services import last_will as last_will_service
from app.services import last_will_crypto
from app.services.user_activity import (
    UserActivityWriteError,
    local_activity_date,
    record_user_activity,
)


def test_active_last_will_requires_contact():
    with pytest.raises(HTTPException) as exc_info:
        last_wills._normalize_status("active", [])

    assert exc_info.value.status_code == 400


def test_active_last_will_requires_content():
    contact = LastWillContact(name="妈妈", email="mom@example.com")

    with pytest.raises(HTTPException) as exc_info:
        last_wills._normalize_status("active", [contact], "")

    assert exc_info.value.status_code == 400


def test_draft_last_will_allows_empty_content():
    draft = LastWillCreate(content="", status="draft")

    assert draft.content == ""


def test_create_last_will_normalizes_blank_optional_scope_ids():
    draft = LastWillCreate(
        agent_id=" ",
        workspace_id="",
        content="留给重要的人",
        status="draft",
    )

    assert draft.agent_id is None
    assert draft.workspace_id is None


def test_contact_requires_email_or_phone():
    with pytest.raises(ValueError):
        LastWillContact(name="妈妈")


def test_contact_rejects_invalid_email():
    with pytest.raises(ValueError):
        LastWillContact(name="妈妈", email="not-an-email")


def test_last_will_crypto_round_trips_new_sensitive_payloads(monkeypatch):
    monkeypatch.setattr(last_will_crypto.settings, "last_will_encryption_key", "k" * 32)

    protected = last_will_crypto.protect_text("一段很重要的话")
    assert protected.startswith("enc:v1:")
    assert last_will_crypto.reveal_text(protected) == "一段很重要的话"
    assert last_will_crypto.reveal_text("历史明文") == "历史明文"

    contact = {"name": "妈妈", "email": "mom@example.com", "phone": "13800000000"}
    protected_contact = last_will_crypto.protect_contact(contact)
    assert protected_contact["email"].startswith("enc:v1:")
    assert last_will_crypto.reveal_contact(protected_contact) == contact


@pytest.mark.asyncio
async def test_create_last_will_rejects_second_will(monkeypatch):
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(last_wills, "db", db)

    with pytest.raises(HTTPException) as exc_info:
        await last_wills.create_last_will(
            LastWillCreate(
                content="留给重要的人",
                contacts=[],
                status="draft",
            ),
            user={"sub": "user-id", "role": "user"},
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_create_last_will_can_be_user_scoped_without_agent(monkeypatch):
    created = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    row = {
        "id": "will-id",
        "userId": "user-id",
        "agentId": None,
        "workspaceId": None,
        "content": "留给重要的人",
        "inactivityDays": 30,
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "status": "draft",
        "lastSeenAt": None,
        "startedAt": None,
        "triggeredAt": None,
        "deliveredAt": None,
        "createdAt": created,
        "updatedAt": created,
    }
    db = SimpleNamespace(
        aiagent=SimpleNamespace(find_unique=AsyncMock()),
        query_raw=AsyncMock(side_effect=[[{"id": "will-id"}], [row]]),
    )
    monkeypatch.setattr(last_wills, "db", db)

    item = await last_wills.create_last_will(
        LastWillCreate(
            content="留给重要的人",
            contacts=[LastWillContact(name="妈妈", email="mom@example.com")],
            status="draft",
        ),
        user={"sub": "user-id", "role": "user"},
    )

    insert_sql = db.query_raw.await_args_list[0].args[0]
    insert_args = db.query_raw.await_args_list[0].args[1:]
    assert "ON CONFLICT (user_id) DO NOTHING" in insert_sql
    assert insert_args[2] is None
    db.aiagent.find_unique.assert_not_awaited()
    assert item.agent_id is None


@pytest.mark.asyncio
async def test_list_last_wills_is_user_level_after_optional_agent_validation(monkeypatch):
    updated = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    db = SimpleNamespace(
        aiagent=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(id="agent-id", userId="user-id", status="active")
            )
        ),
        chatworkspace=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(id="workspace-id", userId="user-id", agentId="agent-id")
            )
        ),
        query_raw=AsyncMock(
            return_value=[
                {
                    "id": "will-id",
                    "userId": "user-id",
                    "agentId": None,
                    "workspaceId": "other-workspace",
                    "content": "留给重要的人",
                    "inactivityDays": 30,
                    "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
                    "status": "draft",
                    "lastSeenAt": None,
                    "startedAt": None,
                    "triggeredAt": None,
                    "deliveredAt": None,
                    "createdAt": updated,
                    "updatedAt": updated,
                }
            ]
        ),
    )
    monkeypatch.setattr(last_wills, "db", db)

    items = await last_wills.list_last_wills(
        agent_id="agent-id",
        workspace_id="workspace-id",
        user={"sub": "user-id", "role": "user"},
    )

    query_sql = db.query_raw.await_args.args[0]
    assert "lw.agent_id =" not in query_sql
    assert "lw.workspace_id =" not in query_sql
    assert items[0].id == "will-id"


@pytest.mark.asyncio
async def test_list_last_wills_without_agent_does_not_touch_agent_scope(monkeypatch):
    updated = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    db = SimpleNamespace(
        aiagent=SimpleNamespace(find_unique=AsyncMock()),
        query_raw=AsyncMock(
            return_value=[
                {
                    "id": "will-id",
                    "userId": "user-id",
                    "agentId": None,
                    "workspaceId": None,
                    "content": "留给重要的人",
                    "inactivityDays": 30,
                    "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
                    "status": "draft",
                    "lastSeenAt": None,
                    "startedAt": None,
                    "triggeredAt": None,
                    "deliveredAt": None,
                    "createdAt": updated,
                    "updatedAt": updated,
                }
            ]
        ),
    )
    monkeypatch.setattr(last_wills, "db", db)

    items = await last_wills.list_last_wills(
        user={"sub": "user-id", "role": "user"},
    )

    db.aiagent.find_unique.assert_not_awaited()
    assert items[0].agent_id is None


@pytest.mark.asyncio
async def test_update_last_will_to_draft_clears_started_at(monkeypatch):
    started = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    row = {
        "id": "will-id",
        "userId": "user-id",
        "agentId": "agent-id",
        "workspaceId": None,
        "content": "留给重要的人",
        "inactivityDays": 30,
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "status": "active",
        "lastSeenAt": None,
        "startedAt": started,
        "triggeredAt": None,
        "deliveredAt": None,
        "createdAt": started,
        "updatedAt": started,
    }
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[row]),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(last_wills, "db", db)

    await last_wills.update_last_will(
        "will-id",
        LastWillUpdate(status="draft"),
        user={"sub": "user-id", "role": "user"},
    )

    update_sql = db.execute_raw.await_args.args[0]
    assert "status = $1" in update_sql
    assert "started_at = NULL" in update_sql


@pytest.mark.asyncio
async def test_update_last_will_to_cancelled_clears_content_timer_only(monkeypatch):
    started = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    row = {
        "id": "will-id",
        "userId": "user-id",
        "agentId": "agent-id",
        "workspaceId": None,
        "content": "留给重要的人",
        "inactivityDays": 60,
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "status": "active",
        "lastSeenAt": None,
        "startedAt": started,
        "triggeredAt": None,
        "deliveredAt": None,
        "createdAt": started,
        "updatedAt": started,
    }
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[row]),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(last_wills, "db", db)

    await last_wills.update_last_will(
        "will-id",
        LastWillUpdate(content="", status="cancelled"),
        user={"sub": "user-id", "role": "user"},
    )

    update_sql = db.execute_raw.await_args.args[0]
    update_args = db.execute_raw.await_args.args[1:]
    assert "content = $1" in update_sql
    assert "status = $2" in update_sql
    assert "started_at = NULL" in update_sql
    assert "contacts =" not in update_sql
    assert "inactivity_days =" not in update_sql
    assert update_args[:2] == ("", "cancelled")


@pytest.mark.asyncio
async def test_delete_last_will_clears_content_but_keeps_settings(monkeypatch):
    started = datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    row = {
        "id": "will-id",
        "userId": "user-id",
        "agentId": "agent-id",
        "workspaceId": None,
        "content": "留给重要的人",
        "inactivityDays": 60,
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "status": "active",
        "lastSeenAt": None,
        "startedAt": started,
        "triggeredAt": None,
        "deliveredAt": None,
        "createdAt": started,
        "updatedAt": started,
    }
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[row]),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(last_wills, "db", db)

    await last_wills.delete_last_will("will-id", user={"sub": "user-id", "role": "user"})

    sql = db.execute_raw.await_args.args[0]
    assert "UPDATE last_wills" in sql
    assert "content = ''" in sql
    assert "DELETE FROM last_wills" not in sql
    assert "contacts" not in sql
    assert "inactivity_days" not in sql


@pytest.mark.asyncio
async def test_record_user_activity_upserts_daily_ledger(monkeypatch):
    db = SimpleNamespace(execute_raw=AsyncMock(return_value=1))
    monkeypatch.setattr("app.services.user_activity.db", db)

    now = datetime(2026, 5, 29, 2, 30, tzinfo=UTC)
    await record_user_activity("user-id", source="auth_me", now=now)

    assert db.execute_raw.await_count == 2
    user_sql = db.execute_raw.await_args_list[0].args[0]
    assert "UPDATE users" in user_sql
    ledger_sql = db.execute_raw.await_args_list[1].args[0]
    ledger_args = db.execute_raw.await_args_list[1].args[1:]
    assert "ON CONFLICT (user_id, local_date)" in ledger_sql
    assert ledger_args[0] == "user-id"
    assert ledger_args[1] == local_activity_date(now).isoformat()
    assert ledger_args[2] == "auth_me"


@pytest.mark.asyncio
async def test_record_user_activity_raises_only_when_all_heartbeat_writes_fail(monkeypatch):
    db = SimpleNamespace(execute_raw=AsyncMock(side_effect=[RuntimeError("user"), RuntimeError("ledger")]))
    monkeypatch.setattr("app.services.user_activity.db", db)

    with pytest.raises(UserActivityWriteError):
        await record_user_activity(
            "user-id",
            source="auth_me",
            now=datetime(2026, 5, 29, 2, 30, tzinfo=UTC),
            raise_on_total_failure=True,
        )


@pytest.mark.asyncio
async def test_scan_due_last_wills_uses_consecutive_missed_login_days(monkeypatch):
    row = {
        "id": "will-id",
        "userId": "user-id",
        "contacts": [
            {"name": "妈妈", "email": "mom@example.com", "phone": "13800000000"}
        ],
        "inactivityDays": 7,
        "lastActivityDate": date(2026, 5, 20),
        "userLastSeenAt": None,
        "userUpdatedAt": datetime(2026, 5, 20, tzinfo=UTC),
        "userCreatedAt": datetime(2026, 5, 1, tzinfo=UTC),
    }
    db = SimpleNamespace(
        query_raw=AsyncMock(
            side_effect=[
                [row],
                [{"id": "will-id"}],
                [{"id": "delivery-email"}],
                [{"id": "delivery-phone"}],
            ]
        ),
    )
    monkeypatch.setattr(last_will_service, "db", db)

    stats = await last_will_service.scan_due_last_wills(
        datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    )

    assert stats == {"checked": 1, "triggered": 1, "deliveries": 2}
    assert db.query_raw.await_count == 4
    update_sql = db.query_raw.await_args_list[1].args[0]
    assert "SET status = 'triggered'" in update_sql
    assert "RETURNING id" in update_sql
    delivery_sql = db.query_raw.await_args_list[2].args[0]
    assert "last_will_deliveries" in delivery_sql


@pytest.mark.asyncio
async def test_scan_due_last_wills_skips_before_threshold(monkeypatch):
    row = {
        "id": "will-id",
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "inactivityDays": 10,
        "lastActivityDate": date(2026, 5, 25),
        "userLastSeenAt": None,
        "userUpdatedAt": datetime(2026, 5, 25, tzinfo=UTC),
        "userCreatedAt": datetime(2026, 5, 1, tzinfo=UTC),
    }
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[row]),
    )
    monkeypatch.setattr(last_will_service, "db", db)

    stats = await last_will_service.scan_due_last_wills(
        datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    )

    assert stats == {"checked": 1, "triggered": 0, "deliveries": 0}
    assert db.query_raw.await_count == 1


@pytest.mark.asyncio
async def test_scan_due_last_wills_skips_after_concurrent_trigger(monkeypatch):
    row = {
        "id": "will-id",
        "contacts": [{"name": "妈妈", "email": "mom@example.com"}],
        "inactivityDays": 5,
        "lastActivityDate": date(2026, 5, 20),
        "userLastSeenAt": None,
        "userUpdatedAt": datetime(2026, 5, 20, tzinfo=UTC),
        "userCreatedAt": datetime(2026, 5, 1, tzinfo=UTC),
    }
    db = SimpleNamespace(query_raw=AsyncMock(side_effect=[[row], []]))
    monkeypatch.setattr(last_will_service, "db", db)

    stats = await last_will_service.scan_due_last_wills(
        datetime(2026, 5, 29, 3, 0, tzinfo=UTC)
    )

    assert stats == {"checked": 1, "triggered": 0, "deliveries": 0}
    assert db.query_raw.await_count == 2
