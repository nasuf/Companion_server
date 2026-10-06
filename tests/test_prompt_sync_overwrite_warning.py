"""Startup must use complete Web history, including resets back to defaults."""
from types import SimpleNamespace as Row
from datetime import datetime, timezone, timedelta
import pytest
from app.services.prompting.store import history_requires_web

@pytest.mark.parametrize('history,custom,sticky,protected', [
    (['bootstrap'], False, False, False),
    (['bootstrap','code_sync','enable','disable'], False, False, False),
    (['bootstrap','manual_save','reset_default'], False, False, True),
    (['bootstrap','restore:old-version'], False, False, True),
    (['bootstrap','unknown-operation'], False, False, True),
    ([], False, False, True),
    (['bootstrap'], True, False, True),
    (['bootstrap'], False, True, True),
])
def test_history_ownership(history, custom, sticky, protected):
    now=datetime.now(timezone.utc)
    row=Row(createdAt=now, webManaged=sticky, content='custom' if custom else 'default', defaultContent='default')
    assert history_requires_web(row,[Row(changeType=h, createdAt=now) for h in history]) is protected

def test_bootstrap_after_an_unrecorded_window_is_protected():
    now=datetime.now(timezone.utc)
    row=Row(createdAt=now-timedelta(hours=1), webManaged=False, content='default', defaultContent='default')
    assert history_requires_web(row,[Row(changeType='bootstrap',createdAt=now)]) is True
