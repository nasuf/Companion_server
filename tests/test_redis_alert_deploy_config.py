import json
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from scripts.redis_alert_deploy_config import apply_config


def test_recipients_survive_redeploy_without_changing_other_settings(tmp_path):
    env, config = tmp_path / '.env', tmp_path / 'recipients.json'
    recipient = str(UUID(int=1))
    config.write_text(json.dumps([recipient]))
    for _ in range(2):
        env.write_text('JWT_SECRET=synthetic\nREDIS_MEMORY_ALERT_USER_IDS=old\n')
        assert apply_config(env, config) == 1
        assert env.read_text() == f'JWT_SECRET=synthetic\nREDIS_MEMORY_ALERT_USER_IDS={recipient}\n'
        assert env.stat().st_mode & 0o777 == 0o600
    config.unlink()
    assert apply_config(env, config) == 0
    assert 'REDIS_MEMORY_ALERT_USER_IDS=\n' in env.read_text()


@pytest.mark.parametrize('invalid', [None, {}, ['bad\nJWT_SECRET=bad'], [str(UUID(int=1))]*2,
    [str(UUID(int=n)) for n in range(21)], [None], 'x'*5000])
def test_invalid_persistent_config_does_not_replace_environment(tmp_path, invalid):
    env, config = tmp_path / '.env', tmp_path / 'recipients.json'
    env.write_text('JWT_SECRET=synthetic\n')
    config.write_text(json.dumps(invalid))
    with pytest.raises((ValueError, TypeError)):
        apply_config(env, config)
    assert env.read_text() == 'JWT_SECRET=synthetic\n'


def test_deploy_applies_recipients_before_stopping_running_service():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/deploy.yml').read_text())
    script = next(step['with']['script'] for step in workflow['jobs']['deploy']['steps']
        if 'redis_alert_deploy_config.py' in step.get('with', {}).get('script', ''))
    assert '/app/companion-secrets/redis-alert-recipients.json' in script
    assert script.index('redis_alert_deploy_config.py') < script.index('stop server')
