"""setup_axiom() 行为测试 — 确保 env 缺/库缺时优雅降级, 不 crash."""

from __future__ import annotations

import logging
from unittest.mock import patch

from app.observability.axiom_setup import setup_axiom


def test_setup_axiom_skips_when_no_token(monkeypatch):
    """缺 AXIOM_TOKEN → 返回 False, root logger handler 数不变."""
    monkeypatch.delenv("AXIOM_TOKEN", raising=False)
    monkeypatch.delenv("AXIOM_DATASET", raising=False)
    root = logging.getLogger()
    n_before = len(root.handlers)
    assert setup_axiom() is False
    assert len(root.handlers) == n_before


def test_setup_axiom_skips_when_no_dataset(monkeypatch):
    """有 token 但缺 dataset 也不装 — 两者必须都给."""
    monkeypatch.setenv("AXIOM_TOKEN", "xaat-test")
    monkeypatch.delenv("AXIOM_DATASET", raising=False)
    root = logging.getLogger()
    n_before = len(root.handlers)
    assert setup_axiom() is False
    assert len(root.handlers) == n_before


def test_setup_axiom_attaches_handler_when_configured(monkeypatch):
    """env 齐 + axiom-py 在 → handler 装到 root, 含 ContextInjectionFilter."""
    monkeypatch.setenv("AXIOM_TOKEN", "xaat-test")
    monkeypatch.setenv("AXIOM_DATASET", "companion-test")
    monkeypatch.delenv("AXIOM_ORG_ID", raising=False)

    root = logging.getLogger()
    handlers_before = list(root.handlers)
    try:
        with patch("axiom_py.Client") as mock_client_cls, \
             patch("axiom_py.logging.AxiomHandler") as mock_handler_cls:
            mock_handler_cls.return_value = logging.NullHandler()  # 装个真 Handler 兼容 addFilter
            assert setup_axiom() is True
            mock_client_cls.assert_called_once()
            mock_handler_cls.assert_called_once()
        # 多了一个 handler
        assert len(root.handlers) == len(handlers_before) + 1
    finally:
        # 测试隔离: 把测试加的 handler 摘掉
        for h in list(root.handlers):
            if h not in handlers_before:
                root.removeHandler(h)


def test_setup_axiom_skips_when_axiom_py_not_installed(monkeypatch):
    """import axiom_py 失败 → 返回 False, log warning, 不 crash."""
    monkeypatch.setenv("AXIOM_TOKEN", "xaat-test")
    monkeypatch.setenv("AXIOM_DATASET", "companion-test")

    import builtins
    real_import = builtins.__import__

    def _no_axiom(name, *args, **kwargs):
        if name.startswith("axiom_py"):
            raise ImportError("simulated: axiom-py not installed")
        return real_import(name, *args, **kwargs)

    root = logging.getLogger()
    n_before = len(root.handlers)
    with patch("builtins.__import__", side_effect=_no_axiom):
        assert setup_axiom() is False
    assert len(root.handlers) == n_before


def test_setup_axiom_skips_when_client_construction_fails(monkeypatch):
    """Client/AxiomHandler 构造异常 → 返回 False, 不 crash app."""
    monkeypatch.setenv("AXIOM_TOKEN", "xaat-bad")
    monkeypatch.setenv("AXIOM_DATASET", "companion-test")
    root = logging.getLogger()
    n_before = len(root.handlers)
    with patch("axiom_py.Client", side_effect=RuntimeError("bad token")):
        assert setup_axiom() is False
    assert len(root.handlers) == n_before
