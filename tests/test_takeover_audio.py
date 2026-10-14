"""Real scoped audio upload, manual send and retry on the owned local database."""
import base64
from unittest.mock import AsyncMock

import pytest

from app.api.public import speech
from app.services.admin_takeover import service
from app.services.chat_media import repo as media
from tests.test_admin_takeover_postgres import activate, auth, human  # noqa: F401
from tests.test_runtime_execution_foundation import flow  # noqa: F401
from tests.test_chat_ingress_effects_postgres import attachment
from tests.test_speech_to_text import _test_m4a


async def test_admin_recording_is_bound_and_playable_after_reload_without_agent_work(human, monkeypatch, tmp_path):
    h = human
    await activate(h)
    h.client._transport.app.include_router(speech.router)
    monkeypatch.setattr(media, 'db', h.flow.db)
    monkeypatch.setattr(speech.chat_media_storage, '_MEDIA_DIR', tmp_path)
    monkeypatch.setattr(speech, '_enforce_rate_limit', AsyncMock())
    asr = AsyncMock(side_effect=AssertionError('voice-only upload must not need ASR'))
    monkeypatch.setattr(speech, 'transcribe_audio', asr)
    blob = _test_m4a(3)
    data = dict(conversation_id=h.flow.ids['conversation'], name='voice.m4a', mime='audio/mp4',
        size=len(blob), duration_seconds=3, display_mode='voice', base64=base64.b64encode(blob).decode())
    # Other admins cannot upload into the claimed conversation.
    assert (await h.client.post('/chat/transcribe', json=data, headers=auth(h, 1))).status_code == 404
    uploaded = await h.client.post('/chat/transcribe', json=data, headers=auth(h))
    assert uploaded.status_code == 200, uploaded.text
    audio = uploaded.json()['attachment']
    assert audio['kind'] == 'audio' and audio['duration_seconds'] == 3
    row = (await h.flow.db.query_raw('SELECT * FROM chat_message_attachments WHERE id=$1', audio['id']))[0]
    assert row['message_id'] is None
    assert (tmp_path / row['storage_key']).read_bytes() == blob
    base = f'/admin-api/takeovers/{h.case}/messages'
    body = {'text': uploaded.json()['text'], 'client_id': 'recording-1',
        'attachments': [{'id': audio['id'], 'url': 'https://forged.invalid', 'duration_seconds': 999}]}
    sent = await h.client.post(base, json=body, headers=auth(h))
    assert sent.status_code == 200, sent.text
    message = await h.flow.db.message.find_unique(where={'id': sent.json()['message_id']})
    assert message.role == 'assistant' and message.metadata['human_takeover']
    stored = message.metadata['attachments'][0]
    assert stored['id'] == audio['id'] and stored['url'] == audio['url']
    assert stored['duration_seconds'] == 3
    assert service.takeover_message_read('assistant', message.metadata, 0) is False
    row = (await h.flow.db.query_raw('SELECT message_id FROM chat_message_attachments WHERE id=$1', audio['id']))[0]
    assert row['message_id'] == message.id
    event = h.manager.send_event.await_args_list[0]
    assert event.args[1] == 'reply' and event.args[2]['attachments'] == [stored]
    retry = await h.client.post(base, json=body, headers=auth(h))
    assert retry.json()['message_id'] == message.id and retry.json()['duplicate']
    assert h.manager.send_event.await_count == 2
    # A new client ID cannot reuse an already bound resource.
    assert (await h.client.post(base, json={**body, 'client_id': 'other'}, headers=auth(h))).status_code == 422
    assert await h.flow.db.message.count(where={'conversationId': data['conversation_id']}) == 1
    assert await h.flow.db.agentrun.count(where={'conversationId': data['conversation_id']}) == 0
    assert await h.flow.db.query_raw('SELECT used FROM user_message_quota WHERE user_id=$1', h.flow.ids['owner']) == []
    asr.assert_not_awaited()


@pytest.mark.parametrize('variant', ['image', 'foreign_user', 'foreign_conversation', 'missing'])
async def test_admin_audio_resource_forgery_is_rejected_atomically(human, variant):
    h = human
    await activate(h)
    ident = await attachment(h.flow)
    if variant != 'image':
        await h.flow.db.execute_raw("UPDATE chat_message_attachments SET kind='audio' WHERE id=$1", ident)
    if variant == 'foreign_user':
        await h.flow.db.execute_raw('UPDATE chat_message_attachments SET user_id=$2 WHERE id=$1', ident, h.admins[1])
    if variant == 'foreign_conversation':
        other = await h.flow.db.conversation.create(data={'userId': h.flow.ids['owner'], 'agentId': h.flow.ids['agent']})
        await h.flow.db.execute_raw('UPDATE chat_message_attachments SET conversation_id=$2 WHERE id=$1', ident, other.id)
    response = await h.client.post(f'/admin-api/takeovers/{h.case}/messages', headers=auth(h),
        json={'text': '', 'client_id': 'forgery', 'attachments': [{'id': 'missing' if variant == 'missing' else ident}]})
    assert response.status_code == 422, response.text
    assert await h.flow.db.message.count(where={'conversationId': h.flow.ids['conversation']}) == 0
    assert (await h.flow.db.query_raw('SELECT message_id FROM chat_message_attachments WHERE id=$1', ident))[0]['message_id'] is None
    h.manager.send_event.assert_not_awaited()
