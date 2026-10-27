"""Watermark backlog fetch: skipped batches must not lose messages.

The caller-provided window is only messages[-6:]. When a batch is skipped
(distributed lock contention / Redis outage) while the user keeps chatting,
more than 6 new messages accumulate; the older ones slide out of the window
and were silently never extracted once a later batch advanced the watermark.
The pipeline now fetches `createdAt > watermark` straight from the DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def scoped_window_unit_dependencies(monkeypatch):
    # SQL receipt-hole detection is exercised with real PG in the ledger tests.
    monkeypatch.setattr('app.services.memory.recording.message_audit.unrecorded_since_admission', AsyncMock(return_value=[]))
    # This module tests window/backlog selection. Actual conversation ownership
    # and observation collection are covered by the processing integration tests.
    from app.services.chat import post_process
    from app.services.memory.recording import diagnostics
    monkeypatch.setattr(post_process, "_resolve_workspace_id_for_conversation",
        AsyncMock(side_effect=lambda conversation_id, user_id, workspace_id: workspace_id or "ws-1"))
    monkeypatch.setattr(diagnostics, "_write", AsyncMock())
    from app.services.memory.recording import chat_sources
    # Selection/coordination only; real source resolution is tested on PostgreSQL.
    monkeypatch.setattr(chat_sources, 'load_chat_sources', AsyncMock(side_effect=lambda **kw: kw['messages']))

UTC = timezone.utc


def _win_msg(role: str, content: str, ts: datetime | None) -> dict:
    msg: dict = {"role": role, "content": content}
    if ts is not None:
        msg["createdAt"] = ts.isoformat()
        msg["id"] = f"{role}:{ts.isoformat()}"
    return msg


def _db_row(mid: str, role: str, content: str, ts: datetime):
    return SimpleNamespace(id=mid, role=role, content=content, createdAt=ts)


def _patch_db(rows_by_role: dict[str, list]):
    mock_db = MagicMock()

    async def _find_many(where=None, order=None, take=None):
        return rows_by_role.get(where.get("role"), [])

    mock_db.message.find_many = AsyncMock(side_effect=_find_many)
    return mock_db


@pytest.mark.asyncio
async def test_backlog_beyond_window_is_extracted():
    """8 条积压 > 窗口 6 条: DB 增量必须把滑出窗口的旧消息也抽到."""
    from app.services.chat.post_process import _bg_memory_pipeline

    wm = datetime.now(UTC) - timedelta(minutes=30)
    base = datetime.now(UTC) - timedelta(minutes=10)
    backlog = [
        _db_row(f"m{i}", "user", f"积压消息{i}", base + timedelta(seconds=i))
        for i in range(8)
    ]
    # Window only carries the last message — older ones slid out.
    window = [_win_msg("user", "积压消息7", base + timedelta(seconds=7))]

    with patch("app.services.chat.post_process.db", _patch_db({"user": backlog})), \
         patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline",
               AsyncMock(return_value=[])) as mock_pipeline:
        await _bg_memory_pipeline("u1", window, conversation_id="c1", workspace_id="ws-1")

    new_conv = mock_pipeline.await_args.kwargs["new_conversation"]
    for i in range(8):
        assert f"积压消息{i}" in new_conv, f"积压消息{i} 应被抽取"
    # Watermark advances to the newest backlog row.
    user_set = next(c for c in mock_set.await_args_list if c.args[1] == "user")
    assert user_set.args[2] == base + timedelta(seconds=7)
    # Evidence ids come from the DB rows.
    assert mock_pipeline.await_args.kwargs["evidence_message_ids"] == [
        f"m{i}" for i in range(8)
    ]


@pytest.mark.asyncio
async def test_db_fetch_failure_holds_progress_without_window_fallback():
    from app.services.chat.post_process import _bg_memory_pipeline

    wm = datetime.now(UTC) - timedelta(minutes=30)
    base = datetime.now(UTC)
    window = [
        _win_msg("user", "老消息", wm - timedelta(minutes=5)),
        _win_msg("user", "新消息", base),
    ]
    mock_db = MagicMock()
    mock_db.message.find_many = AsyncMock(side_effect=RuntimeError("db down"))

    with patch("app.services.chat.post_process.db", mock_db), \
         patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as advance, \
         patch("app.services.chat.post_process.process_memory_pipeline",
               AsyncMock(return_value=[])) as mock_pipeline:
        results = await _bg_memory_pipeline("u1", window, conversation_id="c1", workspace_id="ws-1")

    mock_pipeline.assert_not_awaited()
    advance.assert_not_awaited()
    assert all(r.status == "failed" and r.reason == "backlog_unavailable" for r in results)


@pytest.mark.asyncio
async def test_synthetic_reply_skipped_when_persisted_rows_exist():
    """合成回复 (无 ts) 与已持久化的同轮回复不能重复抽取 — DB 行为准."""
    from app.services.chat.post_process import _bg_memory_pipeline

    wm = datetime.now(UTC) - timedelta(minutes=30)
    base = datetime.now(UTC)
    persisted = [_db_row("a1", "assistant", "已持久化的回复", base)]
    window = [
        _win_msg("user", "hi", wm - timedelta(minutes=1)),  # old → context
        _win_msg("assistant", "已持久化的回复", None),  # synthetic duplicate
    ]

    with patch("app.services.chat.post_process.db", _patch_db({"assistant": persisted})), \
         patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()), \
         patch("app.services.chat.post_process.process_memory_pipeline",
               AsyncMock(return_value=[])) as mock_pipeline:
        await _bg_memory_pipeline("u1", window, conversation_id="c1", workspace_id="ws-1")

    ai_call = next(
        c for c in mock_pipeline.await_args_list if c.kwargs["side"] == "ai"
    )
    # Extracted exactly once (from the persisted row), not twice.
    assert ai_call.kwargs["new_conversation"].count("已持久化的回复") == 1
    assert ai_call.kwargs["evidence_message_ids"] == ["a1"]


@pytest.mark.asyncio
async def test_synthetic_reply_never_fills_an_empty_persisted_backlog():
    """回复尚未持久化 (DB 增量为空) → 合成消息仍要当场抽取."""
    from app.services.chat.post_process import _bg_memory_pipeline

    wm = datetime.now(UTC) - timedelta(minutes=30)
    window = [_win_msg("assistant", "刚生成还没落库的回复", None)]

    with patch("app.services.chat.post_process.db", _patch_db({})), \
         patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=wm)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set, \
         patch("app.services.chat.post_process.process_memory_pipeline",
               AsyncMock(return_value=[])) as mock_pipeline:
        await _bg_memory_pipeline("u1", window, conversation_id="c1", workspace_id="ws-1")

    mock_pipeline.assert_not_awaited()
    mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_watermark_skips_db_fetch():
    """首次运行 (无水位线) 不做 DB 增量 — 否则会把整段历史会话重抽一遍."""
    from app.services.chat.post_process import _bg_memory_pipeline

    mock_db = MagicMock()
    mock_db.message.find_many = AsyncMock(return_value=[])
    base = datetime.now(UTC)
    window = [_win_msg("user", "hi", base)]

    with patch("app.services.chat.post_process.db", mock_db), \
         patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=None)), \
         patch("app.services.chat.post_process.set_watermark", AsyncMock()), \
         patch("app.services.chat.post_process.process_memory_pipeline",
               AsyncMock(return_value=[])):
        await _bg_memory_pipeline("u1", window, conversation_id="c1", workspace_id="ws-1")

    mock_db.message.find_many.assert_not_awaited()
