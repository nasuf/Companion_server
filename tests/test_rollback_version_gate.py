from unittest.mock import patch

import pytest

from scripts.check_rollback_version import DELAY_FLOOR, NORMAL_FLOOR, ancestor, qualify


def test_normal_qualified_image():
    with patch('scripts.check_rollback_version.ancestor', return_value=True):
        assert qualify('.', NORMAL_FLOOR)['mode'] == 'normal'


def test_old_image_cannot_be_routine_even_with_delay_off():
    with patch('scripts.check_rollback_version.ancestor', return_value=False), pytest.raises(ValueError):
        qualify('.', DELAY_FLOOR, delay_enabled=False, queued=0, inflight=0)


@pytest.mark.parametrize('delay,queued,inflight', [(None,0,0),(True,0,0),(False,None,0),
                                                (False,0,None),(False,1,0),(False,0,1)])
def test_emergency_old_image_requires_explicit_off_and_drained(delay, queued, inflight):
    with patch('scripts.check_rollback_version.ancestor', return_value=False), pytest.raises(ValueError):
        qualify('.', 'old', emergency=True, delay_enabled=delay, queued=queued, inflight=inflight)


def test_pre_db8_emergency_reports_unsafe_path():
    with patch('scripts.check_rollback_version.ancestor', return_value=False):
        result = qualify('.', 'old', emergency=True, delay_enabled=False, queued=0, inflight=0)
        assert result['older_than_delay_fix'] is True
        assert result['mode'] == 'emergency_delay_disabled'


def test_unknown_git_revision_is_not_treated_as_safe():
    with patch('scripts.check_rollback_version.subprocess.run') as run:
        run.return_value.returncode = 128
        with pytest.raises(ValueError):
            ancestor('.', NORMAL_FLOOR, 'missing')
