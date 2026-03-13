"""Tests for the cache service."""

from app.services.cache import _make_key


def test_make_key_deterministic():
    """Same inputs produce same key."""
    key1 = _make_key("emb", "hello world")
    key2 = _make_key("emb", "hello world")
    assert key1 == key2


def test_make_key_different_namespaces():
    """Different namespaces produce different keys."""
    key1 = _make_key("emb", "hello")
    key2 = _make_key("ret", "hello")
    assert key1 != key2


def test_make_key_format():
    """Key follows expected format."""
    key = _make_key("emb", "test")
    assert key.startswith("cache:emb:")
