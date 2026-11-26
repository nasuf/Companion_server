"""聊天「语音通话」全局开关：默认关闭、读不到也关闭、后台切换实时生效。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.realtime_voice import feature_gate


@pytest.fixture
def gate_db(monkeypatch):
    row = SimpleNamespace(voiceCallEnabled=False, offlineActivityEnabled=True, offlineGiftEnabled=False)

    async def save(*, where, data):
        for field, value in data['update'].items():
            setattr(row, field, value)

    model = SimpleNamespace(find_unique=AsyncMock(side_effect=lambda **_: row), upsert=AsyncMock(side_effect=save))
    fake = SimpleNamespace(systemconfig=model)
    monkeypatch.setattr(feature_gate, 'db', fake)
    from app.services.offline import module_settings
    monkeypatch.setattr(module_settings, 'db', fake)
    return row, model


@pytest.mark.asyncio
async def test_missing_row_and_read_failure_both_mean_off(monkeypatch):
    monkeypatch.setattr(feature_gate, 'db', SimpleNamespace(systemconfig=SimpleNamespace(
        find_unique=AsyncMock(return_value=None))))
    assert await feature_gate.is_voice_call_enabled() is False
    monkeypatch.setattr(feature_gate, 'db', SimpleNamespace(systemconfig=SimpleNamespace(
        find_unique=AsyncMock(side_effect=RuntimeError('db down')))))
    assert await feature_gate.is_voice_call_enabled() is False


def test_client_reads_the_switch_live(api_client, auth_header, gate_db):
    row, _ = gate_db
    user = auth_header('caller')
    assert api_client.get('/chat/features').status_code == 401
    r = api_client.get('/chat/features', headers=user)
    assert r.json() == {'voice_call_enabled': False}
    assert r.headers['Cache-Control'] == 'no-store'
    row.voiceCallEnabled = True
    assert api_client.get('/chat/features', headers=user).json() == {'voice_call_enabled': True}


def test_admin_toggles_voice_without_touching_offline_flags(api_client, auth_header, gate_db):
    row, model = gate_db
    admin = auth_header('operator', role='admin')
    assert api_client.get('/admin-api/offline-settings', headers=auth_header('caller')).status_code == 403
    assert api_client.get('/admin-api/offline-settings', headers=admin).json() == {
        'activity_enabled': True, 'gift_enabled': False, 'voice_call_enabled': False}
    r = api_client.put('/admin-api/offline-settings', headers=admin, json={'voice_call_enabled': True})
    assert r.status_code == 200
    assert r.json() == {'activity_enabled': True, 'gift_enabled': False, 'voice_call_enabled': True}
    assert model.upsert.await_args.kwargs['data']['update'] == {'voiceCallEnabled': True}
    # 只改线下开关时，语音开关保持原值。
    r = api_client.put('/admin-api/offline-settings', headers=admin, json={'gift_enabled': True})
    assert r.json()['voice_call_enabled'] is True and row.offlineGiftEnabled is True
