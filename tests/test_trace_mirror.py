"""trace_mirror 单测: 写入 / 读取 / 失败兜底."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _detail(trace_id: str = "t1", conv_id: str = "c1") -> dict:
    """构造 load_public_trace 的最小返回 shape."""
    return {
        "trace": {
            "trace_id": trace_id,
            "conversation_id": conv_id,
            "message": "hi",
            "duration_ms": 1234,
            "total_tokens": 50,
            "llm_step_count": 3,
            "external_url": f"https://smith/public/{trace_id}/r",
            "step_count": 5,
        },
        "steps": [{"id": "s1"}, {"id": "s2"}],
    }


class TestWriteTraceMirror:
    @pytest.mark.asyncio
    async def test_happy_path_upserts(self, monkeypatch):
        from app.services.chat import trace_mirror

        upsert_calls = []
        async def fake_upsert(*, where, data):
            upsert_calls.append((where, data))
        fake_db = MagicMock()
        fake_db.messagetrace.upsert = AsyncMock(side_effect=fake_upsert)
        monkeypatch.setattr(trace_mirror, "db", fake_db)

        ok = await trace_mirror.write_trace_mirror(
            detail=_detail("trace-abc", "conv-xyz"), message_id="m1",
        )
        assert ok is True
        assert len(upsert_calls) == 1
        where, data = upsert_calls[0]
        assert where == {"traceId": "trace-abc"}
        assert data["create"]["traceId"] == "trace-abc"
        assert data["create"]["conversationId"] == "conv-xyz"
        assert data["create"]["messageId"] == "m1"
        assert "traceId" not in data["update"]
        assert data["update"]["conversationId"] == "conv-xyz"

    @pytest.mark.asyncio
    async def test_missing_trace_id_returns_false(self):
        from app.services.chat import trace_mirror
        d = _detail()
        d["trace"]["trace_id"] = None
        d["trace"]["root_id"] = None
        ok = await trace_mirror.write_trace_mirror(detail=d)
        assert ok is False

    @pytest.mark.asyncio
    async def test_missing_conv_id_returns_false(self):
        from app.services.chat import trace_mirror
        d = _detail()
        d["trace"]["conversation_id"] = None
        ok = await trace_mirror.write_trace_mirror(detail=d)
        assert ok is False

    @pytest.mark.asyncio
    async def test_db_failure_returns_false_swallowed(self, monkeypatch):
        from app.services.chat import trace_mirror

        fake_db = MagicMock()
        fake_db.messagetrace.upsert = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr(trace_mirror, "db", fake_db)

        ok = await trace_mirror.write_trace_mirror(detail=_detail())
        assert ok is False


class TestReadTraceMirror:
    def _row(self, *, trace_id="t1", conv_id="c1"):
        row = MagicMock()
        row.traceId = trace_id
        row.conversationId = conv_id
        row.rootMessage = "hi"
        row.totalDurationMs = 100
        row.totalTokens = 20
        row.llmStepCount = 1
        row.shareUrl = "https://smith/x/r"
        row.summaryJson = {"name": "chat_request", "status": "success"}
        row.stepsJson = [{"id": "s1"}]
        return row

    @pytest.mark.asyncio
    async def test_get_by_trace_id_hit(self, monkeypatch):
        from app.services.chat import trace_mirror
        fake_db = MagicMock()
        fake_db.messagetrace.find_unique = AsyncMock(return_value=self._row())
        monkeypatch.setattr(trace_mirror, "db", fake_db)

        result = await trace_mirror.get_trace_mirror("t1")
        assert result is not None
        assert result["trace"]["trace_id"] == "t1"
        assert result["trace"]["conversation_id"] == "c1"
        assert result["trace"]["external_url"] == "https://smith/x/r"
        assert result["trace"]["status"] == "success"  # 从 summaryJson 保留
        assert result["steps"] == [{"id": "s1"}]

    @pytest.mark.asyncio
    async def test_get_by_trace_id_miss(self, monkeypatch):
        from app.services.chat import trace_mirror
        fake_db = MagicMock()
        fake_db.messagetrace.find_unique = AsyncMock(return_value=None)
        monkeypatch.setattr(trace_mirror, "db", fake_db)
        assert await trace_mirror.get_trace_mirror("nope") is None

    @pytest.mark.asyncio
    async def test_get_by_trace_id_empty_arg(self):
        from app.services.chat import trace_mirror
        assert await trace_mirror.get_trace_mirror("") is None

    @pytest.mark.asyncio
    async def test_get_by_trace_id_db_error(self, monkeypatch):
        from app.services.chat import trace_mirror
        fake_db = MagicMock()
        fake_db.messagetrace.find_unique = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr(trace_mirror, "db", fake_db)
        assert await trace_mirror.get_trace_mirror("t1") is None

    @pytest.mark.asyncio
    async def test_get_by_message_returns_latest(self, monkeypatch):
        from app.services.chat import trace_mirror
        rows = [self._row(trace_id="t-newest")]
        fake_db = MagicMock()
        fake_db.messagetrace.find_many = AsyncMock(return_value=rows)
        monkeypatch.setattr(trace_mirror, "db", fake_db)

        result = await trace_mirror.get_trace_mirror_by_message("m1")
        assert result is not None
        assert result["trace"]["trace_id"] == "t-newest"
        # find_many 应按 createdAt desc + take=1
        kwargs = fake_db.messagetrace.find_many.call_args.kwargs
        assert kwargs["order"] == {"createdAt": "desc"}
        assert kwargs["take"] == 1

    @pytest.mark.asyncio
    async def test_get_by_message_miss(self, monkeypatch):
        from app.services.chat import trace_mirror
        fake_db = MagicMock()
        fake_db.messagetrace.find_many = AsyncMock(return_value=[])
        monkeypatch.setattr(trace_mirror, "db", fake_db)
        assert await trace_mirror.get_trace_mirror_by_message("m1") is None

    @pytest.mark.asyncio
    async def test_row_to_detail_handles_corrupt_json(self):
        from app.services.chat import trace_mirror
        row = MagicMock()
        row.traceId = "t1"
        row.conversationId = "c1"
        row.rootMessage = None
        row.totalDurationMs = None
        row.totalTokens = None
        row.llmStepCount = None
        row.shareUrl = None
        row.summaryJson = "garbled non-dict"  # 损坏
        row.stepsJson = "garbled non-list"
        result = trace_mirror._row_to_detail(row)
        assert result["trace"]["trace_id"] == "t1"
        assert result["steps"] == []
