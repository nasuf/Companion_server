"""Known returned writes are a lower bound, never successful batch receipts."""
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.services.chat import post_process as post
from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.recording import pipeline as pl, diagnostics as d
from app.services.memory.recording.outcomes import MemorySideOutcome


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pl, "resolve_workspace_id", AsyncMock(return_value="workspace"))
    monkeypatch.setattr(pl, "should_memorize", AsyncMock(return_value=True))
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户周末参观了合成展览", "importance": .6, "main_category": "生活"},
        {"content": "用户下班买了一本合成书", "importance": .6, "main_category": "生活"},
    ]}))
    for name in ["log_memory_evidence", "record_entities_for_memory", "record_topics_for_memory",
                 "record_preferences_for_memory", "log_memory_changelog"]:
        monkeypatch.setattr(pl, name, AsyncMock())
    store = AsyncMock(side_effect=["saved-one", RuntimeError("PRIVATE_PROVIDER_ERROR")])
    monkeypatch.setattr(pl, "store_memory", store)
    return store


@pytest.mark.parametrize("prior", [0, 1])
@pytest.mark.parametrize("conflict", [False, True])
async def test_failed_write_retains_only_confirmed_lower_bound(pipeline, prior, conflict):
    failure = MemoryWriteError(409, "PRIVATE_RECORD_DETAIL") if conflict else RuntimeError("PRIVATE_PROVIDER_ERROR")
    pipeline.side_effect = ["saved-one"] * prior + [failure]
    with pytest.raises(pl.MemoryExtractionError) as caught:
        await pl.process_memory_pipeline("user", "user: 合成参观记录和买书记录", workspace_id="workspace")
    assert caught.value.confirmed_write_count == prior
    assert caught.value.kind == ("memory_write_conflict" if conflict else "memory_write_failed")
    assert "PRIVATE" not in str(caught.value)


async def test_reminder_followup_failure_reports_returned_memory(pipeline, monkeypatch):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户设置合成提醒", "importance": .45, "main_category": "生活",
         "sub_category": "提醒", "recurrence": "daily", "occur_time": "2026-01-01T08:00:00+08:00"},
    ]}))
    monkeypatch.setattr(pl, "_create_reminder_timetrigger", AsyncMock(side_effect=RuntimeError("PRIVATE_TRIGGER_ERROR")))
    with pytest.raises(pl.MemoryExtractionError) as caught:
        await pl.process_memory_pipeline("user", "user: 每天提醒合成事项", workspace_id="workspace")
    assert caught.value.confirmed_write_count == 1
    assert caught.value.kind == "memory_followup_failed"


async def test_partial_result_reaches_diagnostics_without_advancing_progress(monkeypatch):
    monkeypatch.setattr(post, "get_watermark", AsyncMock(return_value=None))
    progress = AsyncMock(); monkeypatch.setattr(post, "set_watermark", progress)
    monkeypatch.setattr(post, "process_memory_pipeline", AsyncMock(side_effect=pl.MemoryExtractionError(
        "PRIVATE", kind="memory_write_failed", confirmed_write_count=2)))
    result = await post._pipeline_with_watermark("user", [
        {"id": "source", "role": "user", "content": "合成文本", "createdAt": datetime.now(UTC)},
    ], "conversation", side="user", workspace_id="workspace")
    assert result.memory_count is None and result.confirmed_write_count == 2
    assert result.status == "failed" and result.progress == "unchanged"
    assert d._safe_side(result)["confirmed_write_count"] == 2
    progress.assert_not_awaited()


@pytest.mark.parametrize("count,expected", [(None,None), (False,None), (-1,None), (1001,None), ("PRIVATE",None), (0,0), (3,3)])
def test_projection_validates_lower_bound(count, expected):
    result = MemorySideOutcome("user", "failed", "memory_write_failed", confirmed_write_count=count)
    safe = d._safe_side(result)
    assert safe["memory_count"] is None and safe["confirmed_write_count"] == expected
    assert "PRIVATE" not in str(safe)


@pytest.mark.parametrize("hook", ["log_memory_evidence", "record_entities_for_memory", "record_topics_for_memory", "record_preferences_for_memory"])
async def test_advisory_failure_does_not_hide_the_committed_write(pipeline, monkeypatch, hook):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户喜欢合成咖啡", "importance": .6, "main_category": "偏好", "sub_category": "饮食喜好"},
    ], "preferences": [{"value": "合成咖啡"}]}))
    failing = AsyncMock(side_effect=RuntimeError("advisory unavailable"))
    monkeypatch.setattr(pl, hook, failing)
    assert await pl.process_memory_pipeline("user", "user: 合成喜好记录", workspace_id="workspace") == ["saved-one"]
    failing.assert_awaited_once()
    pipeline.assert_awaited_once()


async def test_optional_audit_failures_preserve_demotion_and_returned_write(pipeline, monkeypatch):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户设置合成提醒", "importance": .6, "main_category": "生活", "sub_category": "提醒",
         "occur_time": "2000-01-01T00:00:00+00:00", "recurrence": "once"},
    ]}))
    audit = AsyncMock(side_effect=RuntimeError("optional audit unavailable"))
    monkeypatch.setattr(pl, "log_memory_changelog", audit)
    trigger = AsyncMock(); monkeypatch.setattr(pl, "_create_reminder_timetrigger", trigger)
    text = f"user: {pl._IMPORTANCE_EXPRESSIONS[0]}，请记下合成事项"
    assert await pl.process_memory_pipeline("user", text, workspace_id="workspace") == ["saved-one"]
    assert pipeline.await_args.kwargs["sub_category"] == "其他"
    assert [call.args[2] for call in audit.await_args_list] == ["user_emphasized", "reminder_past_time_demoted"]
    trigger.assert_not_awaited()


async def test_invalid_extracted_clock_remains_unknown(pipeline, monkeypatch):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户参观合成展览", "importance": .6, "main_category": "生活", "occur_time": "invalid-date"},
    ]}))
    assert await pl.process_memory_pipeline("user", "user: 合成参观记录", workspace_id="workspace") == ["saved-one"]
    assert pipeline.await_args.kwargs["occur_time"] is None


async def test_low_value_fact_is_not_reported_as_a_write(pipeline, monkeypatch):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户说了合成无价值事实", "importance": .05, "main_category": "生活"},
    ]}))
    assert await pl.process_memory_pipeline("user", "user: 合成普通叙述", workspace_id="workspace") == []
    pipeline.assert_not_awaited()


async def test_name_classification_repair_preserves_the_success_result(pipeline, monkeypatch):
    monkeypatch.setattr(pl, "extract_memories", AsyncMock(return_value={"memories": [
        {"content": "用户叫花卷", "importance": .6, "main_category": "生活", "sub_category": "其他"},
    ]}))
    assert await pl.process_memory_pipeline("user", "user: 我的名字是花卷", workspace_id="workspace") == ["saved-one"]
    assert (pipeline.await_args.kwargs["main_category"], pipeline.await_args.kwargs["sub_category"]) == ("身份", "姓名")
