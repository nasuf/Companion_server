"""Admin gift reset: authenticated scope, complete deletion and rollback."""
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Prisma, Json

from app.services.offline import gift_repository, gift_service

COUNTS = dict(deleted_gifts=7, deleted_tracking_events=14, deleted_messages=21, reset_trigger_states=2)


@pytest.mark.parametrize('role,status', [(None, 401), ('user', 403), ('admin', 200)])
def test_clear_requires_admin_and_uses_jwt_owner(api_client, auth_header, monkeypatch, role, status):
    clear = AsyncMock(return_value=COUNTS)
    monkeypatch.setattr(gift_service, 'clear_all_gifts', clear)
    headers = auth_header('owner', role=role) if role else {}
    response = api_client.delete('/offline/admin/gifts?user_id=foreign&workspace_id=one', headers=headers)
    assert response.status_code == status
    if role == 'admin':
        assert response.json() == COUNTS
        clear.assert_awaited_once_with('owner')
    else:
        clear.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_reports_actual_counts_and_propagates_failure(monkeypatch):
    clear = AsyncMock(side_effect=[COUNTS, RuntimeError('unavailable')])
    monkeypatch.setattr(gift_repository, 'clear_user_gifts', clear)
    assert await gift_service.clear_all_gifts('owner') == COUNTS
    with pytest.raises(RuntimeError, match='unavailable'):
        await gift_service.clear_all_gifts('owner')


@pytest.fixture
async def gift_db(monkeypatch):
    url = os.environ.get('PROACTIVE_E2E_DATABASE_URL', '')
    if not url:
        pytest.skip('Requires isolated PostgreSQL via PROACTIVE_E2E_DATABASE_URL')
    assert urlsplit(url).hostname in {'localhost', '127.0.0.1'}
    assert urlsplit(url).path == '/companion_proactive_e2e'
    db = Prisma(datasource={'url': url}, http={'trust_env': False})
    await db.connect()
    monkeypatch.setattr(gift_repository, 'db', db)
    users, agents, spaces, convs = [], [], [], []
    try:
        for i in range(3):
            if i != 1:
                user = await db.user.create(data={'username': 'gift-reset-' + uuid4().hex})
                users.append(user.id)
            agent = await db.aiagent.create(data={'userId': user.id, 'name': 'Synthetic'})
            agents.append(agent.id)
            space = await db.chatworkspace.create(data={'userId': user.id, 'agentId': agent.id, 'status': 'archived' if i == 1 else 'active'})
            spaces.append(space.id)
            conv = await db.conversation.create(data={'userId': user.id, 'agentId': agent.id, 'workspaceId': space.id})
            convs.append(conv.id)
            await db.realworldtriggerstate.create(data={'userId': user.id, 'agentId': agent.id, 'workspaceId': space.id,
                'lastGiftPaidAt': datetime.now(timezone.utc), 'lastActivityRecommendationAt': datetime.now(timezone.utc), 'metadata': Json({'keep': True})})
        await db.giftaddress.create(data={'userId': users[0], 'recipientName': 'Synthetic', 'phone': '13800000000', 'city': 'Test', 'detail': 'Synthetic'})
        yield db, users, agents, spaces, convs
    finally:
        await db.realworldgift.delete_many(where={'userId': {'in': users}})
        await db.realworldtriggerstate.delete_many(where={'userId': {'in': users}})
        await db.giftaddress.delete_many(where={'userId': {'in': users}})
        await db.message.delete_many(where={'conversationId': {'in': convs}})
        await db.conversation.delete_many(where={'id': {'in': convs}})
        await db.chatworkspace.delete_many(where={'id': {'in': spaces}})
        await db.aiagent.delete_many(where={'id': {'in': agents}})
        await db.user.delete_many(where={'id': {'in': users}})
        await db.disconnect()


async def seed_gift(fixture, status='shipping', index=0):
    db, users, agents, spaces, convs = fixture
    owner = users[1] if index == 2 else users[0]
    gift = await db.realworldgift.create(data={'userId': owner, 'agentId': agents[index],
        'workspaceId': spaces[index], 'conversationId': convs[index], 'status': status, 'giftName': 'Synthetic'})
    for _ in range(2):
        await db.gifttrackingevent.create(data={'giftId': gift.id, 'status': status, 'title': 'Synthetic', 'occurredAt': datetime.now(timezone.utc)})
    # Gift card, user's thanks and assistant reply; legacy card-only metadata too.
    for metadata in ({'component_card': {'type': 'offline_gift', 'payload': {'gift_id': gift.id}}},
                     {'real_world_type': 'gift', 'source_id': gift.id},
                     {'real_world_type': 'gift', 'source_id': gift.id}):
        await db.message.create(data={'conversationId': convs[index], 'role': 'assistant', 'content': 'Synthetic', 'metadata': Json(metadata)})
    return gift


@pytest.mark.asyncio
async def test_all_statuses_workspaces_related_messages_and_foreign_isolation(gift_db):
    db, users, agents, spaces, convs = gift_db
    gifts = []
    for i, status in enumerate(('pending_address', 'selecting', 'ordered', 'shipping', 'delivered', 'failed', 'skipped')):
        gifts.append(await seed_gift(gift_db, status, i % 2))
    # Legacy gifts without workspace must be included.
    await db.realworldgift.update(where={'id': gifts[0].id}, data={'workspaceId': None})
    foreign = await seed_gift(gift_db, 'delivered', 2)
    for conv, metadata in ((convs[0], {}), (convs[0], {'real_world_type': 'activity', 'source_id': gifts[0].id}),
        (convs[0], {'real_world_type': 'gift', 'source_id': foreign.id}),
        (convs[2], {'real_world_type': 'gift', 'source_id': gifts[0].id})):
        await db.message.create(data={'conversationId': conv, 'role': 'user', 'content': 'Keep', 'metadata': Json(metadata)})
    before = await db.realworldtriggerstate.find_many(where={'userId': users[0]})
    assert await gift_repository.clear_user_gifts(users[0]) == COUNTS
    assert await db.realworldgift.count(where={'userId': users[0]}) == 0
    assert await db.realworldgift.count(where={'userId': users[1]}) == 1
    assert await db.gifttrackingevent.count(where={'giftId': foreign.id}) == 2
    assert await db.message.count(where={'conversationId': {'in': convs}}) == 7  # 3 foreign + 4 unrelated.
    assert await db.giftaddress.count(where={'userId': users[0]}) == 1
    after = await db.realworldtriggerstate.find_many(where={'userId': users[0]})
    for state in after:
        old = next(item for item in before if item.id == state.id)
        assert state.lastGiftPaidAt is None
        assert state.lastActivityRecommendationAt == old.lastActivityRecommendationAt
        assert state.metadata == old.metadata
    assert (await db.realworldtriggerstate.find_first(where={'userId': users[1]})).lastGiftPaidAt is not None
    assert await gift_repository.clear_user_gifts(users[0]) == {key: 0 for key in COUNTS}


@pytest.mark.asyncio
async def test_delete_failure_rolls_back_gifts_tracking_messages_and_cooldown(gift_db):
    db, users, agents, spaces, convs = gift_db
    gift = await seed_gift(gift_db)
    name = 'gift_reset_failure_' + uuid4().hex
    # Isolated DB only: fail parent deletion after dependent DELETEs have run.
    await db.execute_raw(f"""CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF OLD.id = '{gift.id}' THEN RAISE EXCEPTION 'synthetic reset failure'; END IF; RETURN OLD; END $$""")
    await db.execute_raw(f'CREATE TRIGGER {name} BEFORE DELETE ON real_world_gifts FOR EACH ROW EXECUTE FUNCTION {name}()')
    try:
        with pytest.raises(Exception, match='synthetic reset failure'):
            await gift_repository.clear_user_gifts(users[0])
        assert await db.realworldgift.count(where={'id': gift.id}) == 1
        assert await db.gifttrackingevent.count(where={'giftId': gift.id}) == 2
        assert await db.message.count(where={'conversationId': convs[0]}) == 3
        assert (await db.realworldtriggerstate.find_first(where={'userId': users[0]})).lastGiftPaidAt is not None
    finally:
        await db.execute_raw(f'DROP TRIGGER {name} ON real_world_gifts')
        await db.execute_raw(f'DROP FUNCTION {name}()')


@pytest.mark.asyncio
async def test_authenticated_api_to_database_clears_and_returns_empty_home(gift_db, auth_header):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.public import offline
    db, users, agents, spaces, convs = gift_db
    await seed_gift(gift_db, 'shipping')
    await seed_gift(gift_db, 'delivered', 1)
    app = FastAPI()
    app.include_router(offline.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.delete('/offline/admin/gifts', headers=auth_header(users[0], role='admin'))
        assert response.status_code == 200
        assert response.json() == dict(deleted_gifts=2, deleted_tracking_events=4, deleted_messages=6, reset_trigger_states=2)
        # Same data sources used by home/tracking views, not mocked counts.
        assert await gift_repository.list_gifts(users[0]) == []
        again = await client.delete('/offline/admin/gifts', headers=auth_header(users[0], role='admin'))
        assert again.json() == {key: 0 for key in COUNTS}
