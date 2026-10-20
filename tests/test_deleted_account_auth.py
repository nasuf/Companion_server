from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from app.api import jwt_auth


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [None, "deleting", "archived"])
async def test_old_token_cannot_access_removed_or_inactive_account(monkeypatch, state):
    record = (
        None
        if state is None
        else SimpleNamespace(role="admin", status=state, archivedAt=None)
    )
    monkeypatch.setattr(
        "app.db.db",
        SimpleNamespace(
            user=SimpleNamespace(find_unique=AsyncMock(return_value=record))
        ),
    )
    monkeypatch.setattr(
        jwt_auth, "decode_jwt", lambda _: {"sub": "owned", "role": "admin"}
    )
    with pytest.raises(HTTPException) as error:
        await jwt_auth.require_user(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic")
        )
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_old_admin_claim_uses_current_database_role(monkeypatch):
    monkeypatch.setattr(
        "app.db.db",
        SimpleNamespace(
            user=SimpleNamespace(
                find_unique=AsyncMock(
                    return_value=SimpleNamespace(
                        role="user", status="active", archivedAt=None
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        jwt_auth, "decode_jwt", lambda _: {"sub": "owned", "role": "admin"}
    )
    payload = await jwt_auth.require_user(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic")
    )
    assert payload["role"] == "user"
    with pytest.raises(HTTPException) as error:
        await jwt_auth.require_admin_jwt(payload)
    assert error.value.status_code == 403
