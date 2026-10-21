from pathlib import Path

import pytest

from scripts.deploy_jpush_preflight import validate


@pytest.fixture
def config():
    return {'JPUSH_ENABLED': 'true', 'JIGUANG_PUSH_APP_KEY': 'a' * 24,
            'JIGUANG_PUSH_MASTER_SECRET': 'b' * 24,
            'JIGUANG_PUSH_PACKAGE_NAME': 'com.bansheng.prod'}


def test_valid_forwarded_configuration(config):
    validate(config)


@pytest.mark.parametrize('key', ['JIGUANG_PUSH_APP_KEY', 'JIGUANG_PUSH_MASTER_SECRET',
                                'JIGUANG_PUSH_PACKAGE_NAME'])
@pytest.mark.parametrize('value', ['', 'bad\nINJECTED=true', 'bad$(command)'])
def test_invalid_forwarded_configuration_is_redacted(config, key, value):
    config[key] = value
    with pytest.raises(ValueError) as error:
        validate(config)
    assert key in str(error.value)
    assert value not in str(error.value) if value else True
    assert not config['JIGUANG_PUSH_MASTER_SECRET'] or config['JIGUANG_PUSH_MASTER_SECRET'] not in str(error.value)


def test_disabled_integration_needs_no_credentials():
    validate({'JPUSH_ENABLED': 'false'})


def test_bad_boolean_is_rejected(config):
    config['JPUSH_ENABLED'] = 'maybe'
    with pytest.raises(ValueError, match='JPUSH_ENABLED'):
        validate(config)


def test_workflow_forwards_secret_as_environment_before_replacing_env():
    workflow = (Path(__file__).parents[1] / '.github/workflows/deploy.yml').read_text()
    assert 'JIGUANG_PUSH_MASTER_SECRET: ${{ secrets.JIGUANG_PUSH_MASTER_SECRET }}' in workflow
    assert 'JIGUANG_PUSH_APP_KEY: ${{ vars.JIGUANG_PUSH_APP_KEY }}' in workflow
    assert 'envs: JIGUANG_PUSH_APP_KEY,JIGUANG_PUSH_MASTER_SECRET,JIGUANG_PUSH_PACKAGE_NAME,JPUSH_ENABLED' in workflow
    assert workflow.index('python3 scripts/deploy_jpush_preflight.py') < workflow.index('cat > .env <<EOF')
    for key in ['JIGUANG_PUSH_APP_KEY', 'JIGUANG_PUSH_MASTER_SECRET', 'JIGUANG_PUSH_PACKAGE_NAME', 'JPUSH_ENABLED']:
        assert f'{key}=${key}' in workflow
