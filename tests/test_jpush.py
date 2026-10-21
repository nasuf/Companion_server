import json
from unittest.mock import AsyncMock
import httpx
import pytest
from pydantic import SecretStr
from app.config import settings
from app.services.notifications import devices
from app.services.notifications.jpush import JPushClient


@pytest.fixture
def single_app_config(monkeypatch):
    monkeypatch.setattr(settings, 'jpush_enabled', None)
    monkeypatch.setattr(settings, 'jpush_apps', '{}')
    monkeypatch.setattr(settings, 'jiguang_push_app_key', 'single-key')
    monkeypatch.setattr(settings, 'jiguang_push_master_secret', SecretStr('single-secret'))
    monkeypatch.setattr(settings, 'jiguang_push_package_name', 'com.bansheng.prod')


def test_jiguang_env_names_are_loaded_and_secret_is_redacted(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv('JIGUANG_PUSH_APP_KEY', 'synthetic-key')
    monkeypatch.setenv('JIGUANG_PUSH_MASTER_SECRET', 'synthetic-secret')
    config = Settings(_env_file=None)
    assert config.jiguang_push_app_key == 'synthetic-key'
    assert config.jiguang_push_master_secret.get_secret_value() == 'synthetic-secret'
    assert 'synthetic-secret' not in repr(config)


def test_single_app_credentials_are_enabled_for_prod_only(single_app_config):
    client = JPushClient()
    assert client.credentials('com.bansheng.prod') == {
        'app_key': 'single-key', 'master_secret': 'single-secret'}
    assert client.credentials('com.bansheng.dev') is None
    assert client.credentials(None) is None


def test_explicit_disable_overrides_valid_credentials(single_app_config, monkeypatch):
    monkeypatch.setattr(settings, 'jpush_enabled', False)
    assert JPushClient().credentials('com.bansheng.prod') is None


@pytest.mark.parametrize('missing', ['jiguang_push_app_key', 'jiguang_push_master_secret'])
def test_incomplete_credentials_cannot_send(single_app_config, monkeypatch, missing):
    monkeypatch.setattr(settings, missing, SecretStr('') if 'secret' in missing else '')
    assert JPushClient().credentials('com.bansheng.prod') is None


def test_explicit_app_mapping_overrides_single_app(single_app_config, monkeypatch):
    monkeypatch.setattr(settings, 'jpush_apps', json.dumps({'com.bansheng.prod': {
        'app_key': 'mapped-key', 'master_secret': 'mapped-secret'}}))
    assert JPushClient().credentials('com.bansheng.prod')['app_key'] == 'mapped-key'


@pytest.mark.parametrize('mapping', ['{', '[]', '{"com.bansheng.prod": {}}'])
def test_invalid_mapping_fails_closed(single_app_config, monkeypatch, mapping):
    monkeypatch.setattr(settings, 'jpush_apps', mapping)
    assert JPushClient().credentials('com.bansheng.prod') is None

@pytest.mark.parametrize('token', ['registration_XYZ123', 'aabb112233445566'])
async def test_android_token_is_preserved(monkeypatch, token):
    query = AsyncMock(return_value=[{'id': 'device'}])
    monkeypatch.setattr(devices.db, 'query_raw', query)
    await devices.register_device(user_id='user', platform='android', token=token,
        environment='production', bundle_id='com.bansheng.prod')
    assert query.await_args.args[3:5] == ('jpush', token)

async def test_logout_preserves_registration_id(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr(devices.db, 'execute_raw', execute)
    await devices.disable_device_for_user(user_id='user', token='registration_XYZ123')
    assert execute.await_args.args[1:3] == ('user', 'registration_XYZ123')
    assert "provider = 'apns'" in execute.await_args.args[0]

async def test_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, 'jpush_enabled', False)
    result = await JPushClient().send_alert(token='rid', package='com.bansheng.prod', title='hi', body='hello', payload={})
    assert not result.ok and result.reason == 'jpush_not_configured'

@pytest.mark.parametrize('status,data,ok', [(200, {'msg_id': '123'}, True),
    (401, {'error': {'code': 1004, 'message': 'sensitive'}}, False), (200, {}, False)])
async def test_targeted_request_and_redacted_errors(monkeypatch, status, data, ok):
    import app.services.notifications.jpush as module
    monkeypatch.setattr(settings, 'jpush_enabled', True)
    monkeypatch.setattr(settings, 'jpush_apps', json.dumps({'com.bansheng.prod': {'app_key':'key', 'master_secret':'secret'}}))
    real_client = httpx.AsyncClient
    def handler(request):
        body = json.loads(request.content)
        assert body['audience'] == {'registration_id': ['rid']}
        assert body['platform'] == ['android']
        assert body['notification']['android']['extras']['route'] == 'chat'
        return httpx.Response(status, json=data)
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    result = await JPushClient().send_alert(token='rid', package='com.bansheng.prod', title='hi', body='hello', payload={'route':'chat'})
    assert result.ok == ok
    assert 'sensitive' not in (result.reason or '')


@pytest.mark.parametrize('platform,token,package', [
    ('unknown', 'abcdefgh', 'com.bansheng.prod'),
    ('android', 'bad token', 'com.bansheng.prod'),
    ('android', 'registration_XYZ123', None),
    ('ios', '', None),
])
async def test_invalid_device_registration_does_not_write(monkeypatch, platform, token, package):
    query = AsyncMock()
    monkeypatch.setattr(devices.db, 'query_raw', query)
    with pytest.raises(ValueError):
        await devices.register_device(user_id='user', platform=platform, token=token,
                                      environment='production', bundle_id=package)
    query.assert_not_awaited()


async def test_jpush_device_query_is_bounded_and_preserves_provider(monkeypatch):
    query = AsyncMock(return_value=[{'id': 'device', 'token': 'registration_XYZ123',
                                   'environment': 'production', 'bundleId': 'com.bansheng.prod'}])
    monkeypatch.setattr(devices.db, 'query_raw', query)
    result = await devices.list_enabled_jpush_devices(user_id='user')
    assert result[0].provider == 'jpush'
    assert result[0].bundle_id == 'com.bansheng.prod'
    assert 'LIMIT 100' in query.await_args.args[0]
    assert query.await_args.args[1] == 'user'


async def test_transport_failure_is_redacted(single_app_config, monkeypatch):
    import app.services.notifications.jpush as module
    real_client = httpx.AsyncClient
    def handler(request):
        raise httpx.ConnectError('sensitive-provider-diagnostic', request=request)
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    result = await JPushClient().send_alert(token='rid', package='com.bansheng.prod',
                                          title='hi', body='hello', payload={})
    assert result.reason == 'jpush_transport_error'
    assert not result.ok
