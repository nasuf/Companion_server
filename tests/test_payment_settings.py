from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from app.services.payments import availability


@pytest.fixture
def gate_db(monkeypatch):
    row = SimpleNamespace(realMoneyPaymentsEnabled=False, wechatPayPurchasesEnabled=False,
                          androidAutoRenewEnabled=False, offlineGiftEnabled=True)
    async def save(*, where, data):
        for field, value in data['update'].items():
            setattr(row, field, value)
    model = SimpleNamespace(find_unique=AsyncMock(side_effect=lambda **_: row), upsert=AsyncMock(side_effect=save))
    monkeypatch.setattr(availability, 'db', SimpleNamespace(systemconfig=model))
    return row, model


@pytest.mark.asyncio
async def test_default_closed_and_read_is_not_cached(monkeypatch):
    read = AsyncMock(side_effect=[None, SimpleNamespace(realMoneyPaymentsEnabled=True), SimpleNamespace(realMoneyPaymentsEnabled=False)])
    monkeypatch.setattr(availability, 'db', SimpleNamespace(systemconfig=SimpleNamespace(find_unique=read)))
    assert not (await availability.get_payment_settings())['real_money_payments_enabled']
    assert (await availability.get_payment_settings())['real_money_payments_enabled']
    assert not (await availability.get_payment_settings())['real_money_payments_enabled']
    assert read.await_count == 3


@pytest.mark.asyncio
async def test_storage_failure_is_unknown_not_enabled_or_false(monkeypatch):
    monkeypatch.setattr(availability, 'db', SimpleNamespace(systemconfig=SimpleNamespace(find_unique=AsyncMock(side_effect=RuntimeError('db down')))))
    with pytest.raises(HTTPException) as error:
        await availability.get_payment_settings()
    assert error.value.status_code == 503


def test_only_admin_can_modify_or_read_admin_switch(api_client, auth_header, gate_db):
    _, db = gate_db
    assert api_client.get('/admin-api/payment-settings').status_code == 401
    assert api_client.get('/admin-api/payment-settings', headers=auth_header('user')).status_code == 403
    assert api_client.put('/admin-api/payment-settings', headers=auth_header('user'), json={'real_money_payments_enabled': True}).status_code == 403
    db.upsert.assert_not_awaited()


def test_runtime_toggle_changes_purchase_permission_and_only_one_column(api_client, auth_header, gate_db):
    row, db = gate_db
    admin = auth_header('operator', role='admin')
    user = auth_header('shopper')
    assert api_client.get('/iap/apple/availability').status_code == 401
    assert api_client.post('/iap/apple/purchase-permission').status_code == 401
    r = api_client.get('/iap/apple/availability', headers=user)
    assert r.json() == {'real_money_payments_enabled': False}
    assert r.headers['cache-control'] == 'no-store'
    assert api_client.post('/iap/apple/purchase-permission', headers=user).status_code == 403
    for enabled in (True, False):
        r = api_client.put('/admin-api/payment-settings', headers=admin, json={'real_money_payments_enabled': enabled})
        assert r.status_code == 200 and r.json()['real_money_payments_enabled'] is enabled
        assert db.upsert.await_args.kwargs['data']['update'] == {'realMoneyPaymentsEnabled': enabled}
        assert row.offlineGiftEnabled is True
        assert api_client.get('/admin-api/payment-settings', headers=admin).json()['real_money_payments_enabled'] is enabled
        assert api_client.post('/iap/apple/purchase-permission', headers=user).status_code == (200 if enabled else 403)


@pytest.mark.parametrize('body', [{}, {'real_money_payments_enabled': None}, {'real_money_payments_enabled': 'false'}, {'real_money_payments_enabled': 1}, {'real_money_payments_enabled': True, 'offlineGiftEnabled': False}])
def test_admin_payload_is_strict(api_client, auth_header, gate_db, body):
    _, db = gate_db
    assert api_client.put('/admin-api/payment-settings', headers=auth_header('operator', role='admin'), json=body).status_code == 422
    db.upsert.assert_not_awaited()


def test_failed_save_does_not_claim_success(api_client, auth_header, gate_db):
    _, db = gate_db
    db.upsert.side_effect = HTTPException(503, 'database unavailable')
    r = api_client.put('/admin-api/payment-settings', headers=auth_header('operator', role='admin'), json={'real_money_payments_enabled': True})
    assert r.status_code == 503
    assert api_client.get('/iap/apple/availability', headers=auth_header('shopper')).json()['real_money_payments_enabled'] is False


def test_closed_gate_does_not_block_paid_transaction_settlement(api_client, auth_header, gate_db):
    from app.api.deps import require_redis
    from app.main import app
    result = {'status': 'granted', 'kind': 'consumable', 'wallet': {'ticket_balance': 90, 'point_balance': 0, 'achievement_points_synced': 0}, 'vip': {'is_vip': False, 'vip_until': None, 'vip_trial_available': True, 'gift_ticket_balance': 0, 'ticket_balance': 90, 'point_balance': 0, 'spendable_tickets': 90}}
    app.dependency_overrides[require_redis] = lambda: None
    try:
        with patch('app.api.public.iap.grant.verify_and_grant', new=AsyncMock(return_value=result)) as grant:
            r = api_client.post('/iap/apple/verify', headers=auth_header('shopper'), json={'transaction_id': 'already-paid'})
            assert r.status_code == 200
            grant.assert_awaited_once()
        with patch('app.api.public.iap.notifications.apply_notification', new=AsyncMock()) as apply:
            assert api_client.post('/iap/apple/notifications', json={'signedPayload': 'synthetic-signed-notification'}).status_code == 200
            apply.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(require_redis, None)


def test_apple_and_wechat_switches_are_independent(api_client, auth_header, gate_db):
    """只开微信支付：Apple 内购的 availability / purchase-permission 仍然关着。"""
    row, db = gate_db
    admin = auth_header('operator', role='admin')
    user = auth_header('shopper')
    r = api_client.put('/admin-api/payment-settings', headers=admin, json={'wechat_pay_purchases_enabled': True})
    assert r.status_code == 200
    assert r.json() == {'real_money_payments_enabled': False, 'wechat_pay_purchases_enabled': True,
                        'android_auto_renew_enabled': False}
    assert db.upsert.await_args.kwargs['data']['update'] == {'wechatPayPurchasesEnabled': True}
    assert row.realMoneyPaymentsEnabled is False
    # App 读的接口只有 Apple 的字段，形状和旧版一致。
    assert api_client.get('/iap/apple/availability', headers=user).json() == {'real_money_payments_enabled': False}
    assert api_client.post('/iap/apple/purchase-permission', headers=user).status_code == 403
    # 旧版后台只发 Apple 字段：不能顺带把微信开关改掉。
    api_client.put('/admin-api/payment-settings', headers=admin, json={'real_money_payments_enabled': True})
    assert row.wechatPayPurchasesEnabled is True
    assert api_client.post('/iap/apple/purchase-permission', headers=user).status_code == 200
    api_client.put('/admin-api/payment-settings', headers=admin, json={'wechat_pay_purchases_enabled': False})
    assert row.realMoneyPaymentsEnabled is True


@pytest.mark.parametrize('body', [{'wechat_pay_purchases_enabled': None}, {'wechat_pay_purchases_enabled': 'true'},
                                  {'wechat_pay_purchases_enabled': True, 'real_money_payments_enabled': None}])
def test_wechat_switch_payload_is_strict(api_client, auth_header, gate_db, body):
    _, db = gate_db
    assert api_client.put('/admin-api/payment-settings', headers=auth_header('operator', role='admin'), json=body).status_code == 422
    db.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_wechat_gate_reads_its_own_column(gate_db):
    row, _ = gate_db
    row.realMoneyPaymentsEnabled = True
    with pytest.raises(HTTPException) as error:
        await availability.require_wechat_pay_purchases()
    assert error.value.status_code == 403
    row.wechatPayPurchasesEnabled, row.realMoneyPaymentsEnabled = True, False
    await availability.require_wechat_pay_purchases()
    with pytest.raises(HTTPException):
        await availability.require_real_money_payments()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [{}, {'unknown_switch': True}])
async def test_service_rejects_unknown_or_empty_changes(gate_db, changes):
    _, db = gate_db
    with pytest.raises(ValueError):
        await availability.set_payment_settings(actor_id='a', **changes)
    db.upsert.assert_not_awaited()


def test_android_store_settings_expose_both_android_flags(api_client, auth_header, gate_db):
    """安卓商城一次读两个开关；Apple 的开关不出现在里面。"""
    row, _ = gate_db
    admin = auth_header('operator', role='admin')
    user = auth_header('shopper')
    assert api_client.get('/store/android-settings', headers=user).json() == {
        'wechat_pay_purchases_enabled': False, 'android_auto_renew_enabled': False}
    api_client.put('/admin-api/payment-settings', headers=admin,
                   json={'android_auto_renew_enabled': True})
    assert row.androidAutoRenewEnabled is True
    # 只改安卓档位开关，微信支付与 Apple 开关都不动。
    assert row.wechatPayPurchasesEnabled is False
    assert row.realMoneyPaymentsEnabled is False
    r = api_client.get('/store/android-settings', headers=user)
    assert r.headers['Cache-Control'] == 'no-store'
    assert r.json() == {'wechat_pay_purchases_enabled': False, 'android_auto_renew_enabled': True}
    assert api_client.get('/store/android-settings').status_code == 401
