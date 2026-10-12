"""Isolated network/transaction doubles for knowledge import policy tests.

Atomic rollback, DB locks and permissions are tested separately against PG.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services.agent_template.knowledge_import import parse_knowledge_document_snapshot
from app.services.agent_template import knowledge as kn


def mock_append(monkeypatch, text, *, existing=()):
    items, origin = parse_knowledge_document_snapshot(text.encode())
    rows = [{"content": content} for content in existing]
    tx = MagicMock()
    tx.query_raw = AsyncMock(side_effect=[[], [], [{"id": "ws"}], rows])
    tx.memorychangelog.create = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=tx)
    context.__aexit__ = AsyncMock(return_value=False)
    database = MagicMock()
    database.query_raw = AsyncMock(return_value=rows)
    database.tx.return_value = context
    monkeypatch.setattr(kn, "db", database)
    monkeypatch.setattr(kn, "get_active_workspace", AsyncMock(return_value=SimpleNamespace(id="ws")))
    monkeypatch.setattr(kn, "generate_embedding", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(kn, "store_embedding", AsyncMock())
    monkeypatch.setattr(kn, "persist_input_origin", AsyncMock(return_value=SimpleNamespace(kind="input")))
    monkeypatch.setattr(kn, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(kn, "_bust_knowledge_rows_cache", AsyncMock())
    monkeypatch.setattr(kn.memory_repo, "invalidate_scope", AsyncMock())
    create = AsyncMock(return_value=SimpleNamespace(id="new-memory"))
    monkeypatch.setattr(kn.memory_repo, "create", create)
    return items, origin, create, tx
