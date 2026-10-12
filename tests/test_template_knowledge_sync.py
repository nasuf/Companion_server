"""Template knowledge append + sync service (agent_template/knowledge.py).

DB / Redis / embedding are mocked; these tests pin the orchestration
semantics: exact-content idempotency, canary vs full watermark behavior, and
verbatim row copying.
"""

from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_template import knowledge as knowledge_mod
from app.services.agent_template.knowledge import (
    KNOWLEDGE_IMPORTANCE,
    KNOWLEDGE_MAIN_CATEGORY,
    KNOWLEDGE_SUB_CATEGORY,
    KnowledgeSyncBusy,
    _build_copy_rows,
    append_knowledge_to_template,
    start_knowledge_sync,
)
from app.services.agent_template.knowledge_import import KnowledgeItem
from app.services.memory.provenance import KNOWLEDGE_SEED


def _item(summary: str) -> KnowledgeItem:
    return KnowledgeItem(section="s", label="l", content=summary, summary=summary)


def _template_row(memory_id: str, content: str, created_at: datetime | None = None):
    return SimpleNamespace(
        id=memory_id,
        content=content,
        createdAt=created_at or datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
        type="life",
        mainCategory="生活",
        subCategory="工作",
        level=1,
        importance=0.86,
        mentionCount=0,
        isArchived=False,
        occurTime=None,
        statementTime=None,
        recurrence=None,
        provenance=KNOWLEDGE_SEED,
    )


def _ws(ws_id: str):
    ws = MagicMock()
    ws.id = ws_id
    return ws


@contextlib.asynccontextmanager
async def _noop_lock(*_args, **_kwargs):
    """Redis-free stand-in for distributed_lock in unit tests."""
    yield True


# ── _build_copy_rows (pure) ────────────────────────────────────────────


def test_build_copy_rows_skips_existing_and_copies_fields():
    rows = [_template_row("t1", "知识A"), _template_row("t2", "知识B")]
    new_rows, id_pairs = _build_copy_rows(
        rows, existing_contents={"知识A"}, user_id="u-9", workspace_id="ws-9"
    )
    assert len(new_rows) == 1 and len(id_pairs) == 1
    payload = new_rows[0]
    assert payload["userId"] == "u-9"
    assert payload["workspaceId"] == "ws-9"
    assert payload["content"] == "知识B"
    assert payload["provenance"] == KNOWLEDGE_SEED
    assert payload["level"] == 1
    # id pair links the template row to the freshly generated clone id.
    assert id_pairs[0][0] == "t2" and id_pairs[0][1] == payload["id"]


def test_build_copy_rows_all_existing_is_noop():
    rows = [_template_row("t1", "知识A")]
    new_rows, id_pairs = _build_copy_rows(
        rows, existing_contents={"知识A"}, user_id="u", workspace_id="w"
    )
    assert new_rows == [] and id_pairs == []


# ── append_knowledge_to_template ───────────────────────────────────────


@pytest.mark.asyncio
async def test_append_stores_new_and_skips_existing(monkeypatch):
    from tests.knowledge_input_test_support import mock_append
    from app.services.agent_template.knowledge_import import parse_knowledge_document_snapshot
    text = "知识\n内容：已有知识\n内容：新知识"
    parsed, _ = parse_knowledge_document_snapshot(text.encode())
    items, origin, create, tx = mock_append(monkeypatch, text, existing=[parsed[0].summary])
    result = await append_knowledge_to_template(
        template_agent_id="tpl-1", template_user_id="sys-owner", items=items, input_origin=origin,
    )
    assert result["parsed"] == 2
    assert result["stored"] == 1
    assert result["skipped_duplicates"] == 1
    create.assert_awaited_once()
    kwargs = create.call_args.kwargs
    assert create.call_args.args == ("ai",)
    assert kwargs["content"] == items[1].summary
    assert kwargs["level"] == 1
    assert kwargs["importance"] == KNOWLEDGE_IMPORTANCE
    assert kwargs["mainCategory"] == KNOWLEDGE_MAIN_CATEGORY
    assert kwargs["subCategory"] == KNOWLEDGE_SUB_CATEGORY
    assert kwargs["workspaceId"] == "ws"
    assert kwargs["provenance"] == KNOWLEDGE_SEED
    assert kwargs["database"] is tx
    assert kwargs["evidence_sources"]
    assert kwargs["valueUpdatedAt"] == kwargs["statementTime"]
    assert kwargs["valueUpdatedAt"].tzinfo == UTC


@pytest.mark.asyncio
async def test_append_without_workspace_raises():
    from app.services.agent_template.knowledge_import import parse_knowledge_document_snapshot
    items, origin = parse_knowledge_document_snapshot("知识\n内容：测试".encode())
    with patch.object(knowledge_mod, "get_active_workspace", AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="活跃 workspace"):
            await append_knowledge_to_template(
                template_agent_id="tpl-1", template_user_id="sys-owner", items=items, input_origin=origin,
            )


# ── start_knowledge_sync ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_sync_rejects_when_already_running():
    running = {"status": "running", "started_at_ts": time.time()}
    with patch.object(knowledge_mod, "get_sync_progress", AsyncMock(return_value=running)):
        with pytest.raises(KnowledgeSyncBusy):
            await start_knowledge_sync(template_agent_id="tpl-1", agent_ids=None)


@pytest.mark.asyncio
async def test_start_sync_stale_running_does_not_block():
    stale = {"status": "running", "started_at_ts": time.time() - 7200}
    fake_aimemory = MagicMock()
    fake_aimemory.find_many = AsyncMock(return_value=[_template_row("t1", "知识A")])
    fired: list = []
    placeholder = AsyncMock()
    with (
        patch.object(knowledge_mod, "get_sync_progress", AsyncMock(return_value=stale)),
        patch.object(knowledge_mod, "get_active_workspace", AsyncMock(return_value=_ws("tpl-ws"))),
        patch.object(knowledge_mod.db, "aimemory", fake_aimemory),
        patch.object(
            knowledge_mod,
            "_resolve_target_agents",
            AsyncMock(return_value=[{"id": "a1", "user_id": "u1"}]),
        ),
        patch.object(knowledge_mod, "_write_progress", placeholder),
        patch.object(knowledge_mod, "fire_background", fired.append),
    ):
        result = await start_knowledge_sync(template_agent_id="tpl-1", agent_ids=None)
    assert result == {"started": True, "mode": "all", "total_agents": 1}
    assert len(fired) == 1
    fired[0].close()  # never awaited — silence the warning
    # The anti-race "running" placeholder must land before the job is fired.
    placeholder.assert_awaited_once()
    assert placeholder.call_args.args[1]["status"] == "running"


@pytest.mark.asyncio
async def test_start_sync_requires_knowledge_rows():
    fake_aimemory = MagicMock()
    fake_aimemory.find_many = AsyncMock(return_value=[])
    with (
        patch.object(knowledge_mod, "get_sync_progress", AsyncMock(return_value=None)),
        patch.object(knowledge_mod, "get_active_workspace", AsyncMock(return_value=_ws("tpl-ws"))),
        patch.object(knowledge_mod.db, "aimemory", fake_aimemory),
    ):
        with pytest.raises(ValueError, match="还没有知识记忆"):
            await start_knowledge_sync(template_agent_id="tpl-1", agent_ids=None)


@pytest.mark.asyncio
async def test_start_sync_selected_mode():
    fake_aimemory = MagicMock()
    fake_aimemory.find_many = AsyncMock(return_value=[_template_row("t1", "知识A")])
    resolve = AsyncMock(return_value=[{"id": "a1", "user_id": "u1"}])
    fired: list = []
    with (
        patch.object(knowledge_mod, "get_sync_progress", AsyncMock(return_value=None)),
        patch.object(knowledge_mod, "get_active_workspace", AsyncMock(return_value=_ws("tpl-ws"))),
        patch.object(knowledge_mod.db, "aimemory", fake_aimemory),
        patch.object(knowledge_mod, "_resolve_target_agents", resolve),
        patch.object(knowledge_mod, "_write_progress", AsyncMock()),
        patch.object(knowledge_mod, "fire_background", fired.append),
    ):
        result = await start_knowledge_sync(template_agent_id="tpl-1", agent_ids=["a1"])
    assert result["mode"] == "selected"
    resolve.assert_awaited_once_with("tpl-1", ["a1"])
    fired[0].close()


# _sync_agent scope, copying, idempotency and rollback are exercised against
# PostgreSQL in test_knowledge_sync_evidence_postgres.py.


# ── _run_sync_locked: watermark semantics ──────────────────────────────


@pytest.mark.asyncio
async def test_full_sync_success_advances_watermark():
    latest = datetime(2026, 7, 23, 9, 30, tzinfo=UTC)
    rows = [
        _template_row("t1", "知识A", created_at=datetime(2026, 7, 22, 9, 0, tzinfo=UTC)),
        _template_row("t2", "知识B", created_at=latest),
    ]
    advance = AsyncMock()
    progress_writes: list[dict] = []

    async def _capture(_tid, progress):
        progress_writes.append(dict(progress))

    with (
        patch.object(knowledge_mod, "_sync_agent", AsyncMock(return_value=2)),
        patch.object(knowledge_mod, "_advance_watermark", advance),
        patch.object(knowledge_mod, "_write_progress", _capture),
    ):
        await knowledge_mod._run_sync_locked(
            template_agent_id="tpl-1",
            template_rows=rows,
            targets=[{"id": "a1", "user_id": "u1"}, {"id": "a2", "user_id": "u2"}],
            full_mode=True,
        )

    advance.assert_awaited_once_with("tpl-1", latest)
    final = progress_writes[-1]
    assert final["status"] == "done"
    assert final["synced_agents"] == 2
    assert final["copied_memories"] == 4
    assert final["watermark_advanced"] is True


@pytest.mark.asyncio
async def test_full_sync_with_failure_keeps_watermark():
    advance = AsyncMock()
    sync_agent = AsyncMock(side_effect=[1, RuntimeError("no_active_workspace")])
    with (
        patch.object(knowledge_mod, "_sync_agent", sync_agent),
        patch.object(knowledge_mod, "_advance_watermark", advance),
        patch.object(knowledge_mod, "_write_progress", AsyncMock()),
    ):
        await knowledge_mod._run_sync_locked(
            template_agent_id="tpl-1",
            template_rows=[_template_row("t1", "知识A")],
            targets=[{"id": "a1", "user_id": "u1"}, {"id": "a2", "user_id": "u2"}],
            full_mode=True,
        )
    advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_canary_sync_never_advances_watermark():
    advance = AsyncMock()
    with (
        patch.object(knowledge_mod, "_sync_agent", AsyncMock(return_value=3)),
        patch.object(knowledge_mod, "_advance_watermark", advance),
        patch.object(knowledge_mod, "_write_progress", AsyncMock()),
    ):
        await knowledge_mod._run_sync_locked(
            template_agent_id="tpl-1",
            template_rows=[_template_row("t1", "知识A")],
            targets=[{"id": "a1", "user_id": "u1"}],
            full_mode=False,
        )
    advance.assert_not_awaited()


async def test_watermark_failure_keeps_committed_counts_and_reports_finalization_error():
    writes = []

    async def capture(_tid, progress):
        writes.append(dict(progress))

    with (
        patch.object(knowledge_mod, "_sync_agent", AsyncMock(return_value=2)),
        patch.object(knowledge_mod, "_advance_watermark", AsyncMock(side_effect=RuntimeError("db offline"))),
        patch.object(knowledge_mod, "_write_progress", capture),
    ):
        await knowledge_mod._run_sync_locked(
            template_agent_id="tpl-1", template_rows=[_template_row("t1", "知识A")],
            targets=[{"id": "a1", "user_id": "u1"}], full_mode=True,
        )
    final = writes[-1]
    assert final["status"] == "error" and final["error"] == "watermark_write_failed"
    assert final["watermark_advanced"] is False and final["finished_at"]
    assert final["synced_agents"] == 1 and final["copied_memories"] == 2
    assert final["processed_agents"] == 1 and final["failed_agents"] == 0


async def test_watermark_missing_target_is_not_success():
    with patch.object(knowledge_mod.db, "execute_raw", AsyncMock(return_value=0)):
        with pytest.raises(ValueError, match="watermark_target_unavailable"):
            await knowledge_mod._advance_watermark("missing", datetime.now(UTC))


def test_build_copy_rows_deduplicates_a_dirty_template_batch_without_mutating_input():
    existing={"already"}
    rows=[_template_row("t1","same"),_template_row("t2","same"),_template_row("t3",""),_template_row("t4","already")]
    new,pairs=_build_copy_rows(rows,existing_contents=existing,user_id="u",workspace_id="w")
    assert len(new)==len(pairs)==1 and new[0]['content']=='same'
    assert new[0]['valueUpdatedAt'] is not None and existing=={"already"}


async def test_start_sync_over_limit_rejects_before_launch(monkeypatch):
    monkeypatch.setattr(knowledge_mod,'MAX_KNOWLEDGE_SYNC_MEMORIES',1)
    fake=MagicMock();fake.find_many=AsyncMock(return_value=[_template_row('t1','one'),_template_row('t2','two')])
    fired=MagicMock()
    with (
        patch.object(knowledge_mod,'get_sync_progress',AsyncMock(return_value=None)),
        patch.object(knowledge_mod,'get_active_workspace',AsyncMock(return_value=_ws('tpl-ws'))),
        patch.object(knowledge_mod.db,'aimemory',fake),
        patch.object(knowledge_mod,'fire_background',fired),
    ):
        with pytest.raises(ValueError,match='over_limit'):
            await start_knowledge_sync(template_agent_id='tpl-1',agent_ids=None)
    assert fake.find_many.await_args.kwargs['take']==2
    fired.assert_not_called()
