from types import SimpleNamespace

import pytest

from evals.context_embedding.run_eval import context_policy_override


def test_no_override_by_default():
    assert context_policy_override(SimpleNamespace()) == {}
    assert context_policy_override(SimpleNamespace(rounds=None, main_history_tokens=None)) == {}


def test_override_uses_admin_validation():
    assert context_policy_override(SimpleNamespace(rounds=8, main_history_tokens=6000)) == {
        "rounds": 8, "main_history_tokens": 6000}
    # Default values are not stored, same as the admin setting.
    assert context_policy_override(SimpleNamespace(rounds=5, main_history_tokens=None)) == {}
    with pytest.raises(ValueError):
        context_policy_override(SimpleNamespace(rounds=0, main_history_tokens=None))


def test_override_reaches_hot_path(monkeypatch):
    from app.services import runtime_config
    from app.services.chat import context_policy

    override = context_policy_override(SimpleNamespace(rounds=12, main_history_tokens=None))
    monkeypatch.setattr(runtime_config, "chat_context_policy_overrides", lambda: dict(override))
    assert context_policy.current().rounds == 12
