from unittest.mock import patch

import pytest

from scripts.check_rollback_version import (
    DELAY_FLOOR,
    DELAY_SAFE_FLOOR,
    NORMAL_FLOOR,
    ancestor,
    qualify,
    supports_topic_policy as actual_topic_capability,
)


@pytest.fixture(autouse=True)
def topic_capability_for_legacy_floor_tests(monkeypatch):
    monkeypatch.setattr('scripts.check_rollback_version.supports_topic_policy', lambda *_: True)


def test_normal_qualified_image():
    with patch('scripts.check_rollback_version.ancestor', return_value=True):
        assert qualify('.', NORMAL_FLOOR)['mode'] == 'normal'


def test_routine_floor_keeps_filler_shortcut_paused():
    # c302aa61 is the first image that reads the stored 0% filler probability.
    assert NORMAL_FLOOR.startswith('c302aa61')
    assert DELAY_SAFE_FLOOR.startswith('410f2ba1')


def _delay_safe_only(_repo, base, _target):
    return base != NORMAL_FLOOR


@pytest.mark.parametrize('emergency', [False, True])
def test_delay_safe_image_requires_accepting_filler_regression(emergency):
    with patch('scripts.check_rollback_version.ancestor', side_effect=_delay_safe_only), \
            pytest.raises(ValueError, match='accept-filler-shortcut'):
        qualify('.', DELAY_SAFE_FLOOR, emergency=emergency, delay_enabled=False, queued=0, inflight=0)


def test_accepted_delay_safe_image_keeps_delay_and_reports_regression():
    with patch('scripts.check_rollback_version.ancestor', side_effect=_delay_safe_only):
        result = qualify('.', DELAY_SAFE_FLOOR, accept_filler_shortcut=True)
    assert result['mode'] == 'filler_shortcut_restored'
    assert result['minimum_commit'] == DELAY_SAFE_FLOOR


def test_old_image_cannot_be_routine_even_with_delay_off():
    with patch('scripts.check_rollback_version.ancestor', return_value=False), \
            pytest.raises(ValueError, match='--emergency'):
        qualify('.', DELAY_FLOOR, accept_filler_shortcut=True, delay_enabled=False, queued=0, inflight=0)


@pytest.mark.parametrize('delay,queued,inflight', [(None,0,0),(True,0,0),(False,None,0),
                                                (False,0,None),(False,1,0),(False,0,1)])
def test_emergency_old_image_requires_explicit_off_and_drained(delay, queued, inflight):
    with patch('scripts.check_rollback_version.ancestor', return_value=False), \
            pytest.raises(ValueError, match='delay OFF'):
        qualify('.', 'old', emergency=True, accept_filler_shortcut=True,
                delay_enabled=delay, queued=queued, inflight=inflight)


def test_pre_db8_emergency_reports_unsafe_path():
    with patch('scripts.check_rollback_version.ancestor', return_value=False):
        result = qualify('.', 'old', emergency=True, accept_filler_shortcut=True,
                         delay_enabled=False, queued=0, inflight=0)
        assert result['older_than_delay_fix'] is True
        assert result['filler_shortcut_restored'] is True
        assert result['mode'] == 'emergency_delay_disabled'


def test_unknown_git_revision_is_not_treated_as_safe():
    with patch('scripts.check_rollback_version.subprocess.run') as run:
        run.return_value.returncode = 128
        with pytest.raises(ValueError):
            ancestor('.', NORMAL_FLOOR, 'missing')


@pytest.mark.parametrize('rows',[None,1,100,-1])
def test_topic_unaware_rollback_cannot_drop_preferences_even_in_emergency(rows):
    with patch('scripts.check_rollback_version.supports_topic_policy',return_value=False):
        with pytest.raises(ValueError,match='preferences forbid'):
            qualify('.',NORMAL_FLOOR,topic_policy_rows=rows,emergency=True,
                accept_filler_shortcut=True,delay_enabled=False,queued=0,inflight=0)


def test_empty_topic_state_allows_existing_version_checks():
    with patch('scripts.check_rollback_version.supports_topic_policy',return_value=False), \
            patch('scripts.check_rollback_version.ancestor',return_value=True):
        assert qualify('.',NORMAL_FLOOR,topic_policy_rows=0)['mode']=='normal'


def test_topic_capability_requires_all_current_source_entrypoints():
    from scripts.check_rollback_version import supports_topic_policy
    from types import SimpleNamespace
    with patch('scripts.check_rollback_version.subprocess.run',side_effect=[
        SimpleNamespace(returncode=0,stdout='topic_policy_v1'),
        SimpleNamespace(returncode=0,stdout='record_turn_preferences'),
        SimpleNamespace(returncode=0,stdout='no current guard'),
    ]):
        assert not actual_topic_capability('.', 'candidate')
