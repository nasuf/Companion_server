from unittest.mock import AsyncMock

import pytest

from app.services.offline import gift_repository, gift_service


def test_address_editor_returns_only_authenticated_owners_unmasked_data(api_client, auth_header, monkeypatch):
    address = AsyncMock(return_value={
        "id": "address-1", "recipient_name": "测试", "phone": "13812345678",
        "city": "上海市", "detail": "测试路123号",
    })
    monkeypatch.setattr(gift_repository, "default_address", address)
    response = api_client.get("/offline/gifts/address?user_id=other", headers=auth_header("owner"))
    assert response.status_code == 200
    assert response.json()["phone"] == "13812345678"
    assert response.headers["cache-control"] == "no-store"
    address.assert_awaited_once_with("owner", masked=False)


def test_address_delete_is_authenticated_owner_scoped_and_idempotent(api_client, auth_header, monkeypatch):
    execute = AsyncMock(side_effect=[1, 0])
    monkeypatch.setattr(gift_repository.db, "execute_raw", execute)
    for _ in range(2):
        response = api_client.delete("/offline/gifts/address?user_id=other", headers=auth_header("owner"))
        assert response.status_code == 204
        assert response.content == b""
    assert execute.await_count == 2
    for call in execute.await_args_list:
        sql, owner = call.args
        assert owner == "owner"
        assert "DELETE FROM gift_addresses" in sql
        assert "user_id = $1" in sql
        assert "is_default = TRUE" in sql
        assert "real_world_gifts" not in sql  # Independent order snapshots survive.


@pytest.mark.parametrize("method", ["get", "delete"])
def test_address_operations_require_login(api_client, monkeypatch, method):
    read = AsyncMock()
    delete = AsyncMock()
    monkeypatch.setattr(gift_service, "get_address", read)
    monkeypatch.setattr(gift_service, "delete_address", delete)
    response = getattr(api_client, method)("/offline/gifts/address")
    assert response.status_code in (401, 403)
    read.assert_not_awaited()
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_home_address_remains_masked(monkeypatch):
    monkeypatch.setattr(gift_service.repo, "resolve_user_context", AsyncMock(return_value=None))
    address = AsyncMock(return_value=None)
    monkeypatch.setattr(gift_repository, "default_address", address)
    monkeypatch.setattr(gift_repository, "list_gifts", AsyncMock(return_value=[]))
    home = await gift_service.get_gifts("owner")
    assert home.address is None
    address.assert_awaited_once_with("owner", masked=True)


@pytest.mark.asyncio
async def test_delete_failure_propagates_without_claiming_success(monkeypatch):
    monkeypatch.setattr(gift_repository.db, "execute_raw", AsyncMock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        await gift_service.delete_address("owner")
