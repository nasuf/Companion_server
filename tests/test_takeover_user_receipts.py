"""Only actual user visibility receipts update human messages; missing receipts stay unread."""
from unittest.mock import AsyncMock
from types import SimpleNamespace
import httpx
import pytest
from fastapi import FastAPI
from app.api.public import conversations
from app.api.jwt_auth import require_user
from app.services.admin_takeover import service
from app.services.runtime.ws_manager import manager


@pytest.fixture
def receipts(monkeypatch):
    query = AsyncMock(return_value=[{'id': 'message', 'client_id': 'draft', 'takeover_id': 'case', 'owner_admin_id': 'admin'}])
    monkeypatch.setattr(service, 'db', SimpleNamespace(query_raw=query))
    monkeypatch.setattr(manager, 'send_to_admin', AsyncMock(return_value=True))
    app = FastAPI()
    app.include_router(conversations.router)
    app.dependency_overrides[require_user] = lambda: {'sub': 'actual-user', 'role': 'user'}
    return app, query


async def test_read_uses_token_identity_and_only_notifies_the_owning_admin(receipts):
    app, query = receipts
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/conversations/conv/read', json={'message_ids': ['message', 'message'], 'user_id': 'victim'})
        assert response.status_code == 200
    assert query.await_args.args[1:] == ('conv', 'actual-user', ['message'])
    manager.send_to_admin.assert_awaited_once_with('admin', 'takeover_user_read', {
        'takeover_id': 'case', 'conversation_id': 'conv', 'message_ids': ['message'], 'client_ids': ['draft'],
    })


async def test_missing_or_already_read_messages_are_a_noop(receipts):
    app, query = receipts
    query.return_value = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.post('/conversations/conv/read', json={'message_ids': ['foreign']})).status_code == 200
    manager.send_to_admin.assert_not_awaited()


@pytest.mark.parametrize('ids', [[], ['id'] * 101, ['x' * 81], [True], ['']])
async def test_invalid_or_unbounded_receipts_are_rejected_before_lookup(receipts, ids):
    app, query = receipts
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.post('/conversations/conv/read', json={'message_ids': ids})).status_code == 422
    query.assert_not_awaited()


def test_history_receipt_does_not_confuse_send_success_with_read():
    assert service.takeover_message_read('assistant', {'human_takeover': True}, 99) is False
    assert service.takeover_message_read('assistant', {'human_takeover': True, 'takeover_user_read_at': 'seen'}, 99) is True
    assert service.takeover_message_read('assistant', {'real_world_type': 'activity', 'trigger_type': 'offline_activity_recommendation_card'}, 99) is False
    assert service.takeover_message_read('assistant', {}, 99) is None


async def test_user_read_requires_authentication():
    app = FastAPI()
    app.include_router(conversations.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.post('/conversations/conv/read', json={'message_ids': ['message']})).status_code == 401
