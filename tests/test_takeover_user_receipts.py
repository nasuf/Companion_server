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


async def test_database_read_receipt_survives_admin_notification_failure(receipts):
    app, query = receipts
    manager.send_to_admin.side_effect = ConnectionError('admin websocket unavailable')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/conversations/conv/read', json={'message_ids': ['message']})
    assert response.status_code == 200 and query.await_count == 1


async def test_history_marks_only_recipient_receipt_messages(monkeypatch):
    def row(message_id, role, metadata):
        return SimpleNamespace(id=message_id, conversationId='conv', role=role,
                               content='content', metadata=metadata, createdAt='2026-10-14T00:00:00Z')
    messages = [
        row('human', 'assistant', {'human_takeover': True}),
        row('ordinary', 'assistant', {}),
        row('activity', 'assistant', {'real_world_type': 'activity',
             'trigger_type': 'offline_activity_recommendation_card'}),
        row('user', 'user', {'human_takeover': True}),
    ]
    monkeypatch.setattr(conversations, 'db', SimpleNamespace(
        message=SimpleNamespace(find_many=AsyncMock(return_value=messages))))
    monkeypatch.setattr(service, 'takeover_read_cursor', AsyncMock(return_value=0))
    items = await conversations.list_messages(
        conversation_id='conv', limit=100, offset=0, include_metadata=True,
        include_achievements=False, include_usage=False, conv=SimpleNamespace(userId='user', agentId='agent'),
        user={'sub': 'user', 'role': 'user'},
    )
    by_id = {item.id: item for item in items}
    assert by_id['human'].read is False and by_id['human'].metadata['takeover_receipt_required'] is True
    assert by_id['activity'].read is False and by_id['activity'].metadata['takeover_receipt_required'] is True
    assert 'takeover_receipt_required' not in by_id['ordinary'].metadata
    assert 'takeover_receipt_required' not in by_id['user'].metadata


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
