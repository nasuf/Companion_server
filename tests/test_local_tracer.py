"""本地 trace 采集 (local_tracer.py) 单测.

覆盖:
- LocalTracer 生命周期 (enter/attach_to_parent/close, ContextVar handler)
- 真实 langchain fake model 回调链路: unary + streaming → trace_runs 行
- 行 → normalized step 的 shape 与 public_trace._normalize_step 对齐 (golden)
- inputs/outputs 存储形态兼容 trace_enrich 的提取器
- dotted_order / child_ids 合成, 卡死 run 标记, settled 判定
- resolve_trace_for_message 本地分支 (mirror 新鲜 / 过期重建 / trace_expired)
- 保留期清理
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings


def _unjson(value):
    """Unwrap prisma.Json wrapper captured in mock call args."""
    return value.data if hasattr(value, "data") else value


def _fake_db():
    fake = MagicMock()
    fake.tracerun.create = AsyncMock()
    fake.tracerun.upsert = AsyncMock()
    fake.tracerun.update = AsyncMock()
    fake.tracerun.count = AsyncMock(return_value=0)
    fake.tracerun.find_many = AsyncMock(return_value=[])
    fake.tracerun.delete_many = AsyncMock(return_value=0)
    return fake


def _row(
    *,
    run_id: str,
    trace_id: str = "trace-1",
    parent_id: str | None = None,
    name: str = "ChatOpenAI",
    run_type: str = "llm",
    status: str = "success",
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    first_token_at: datetime | None = None,
    model_name: str | None = "qwen3.5-flash",
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 20,
    total_tokens: int | None = 120,
    prompt_token_details=None,
    inputs=None,
    outputs=None,
    error: str | None = None,
    events=None,
    extra=None,
):
    base_time = datetime(2026, 7, 19, 3, 0, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=run_id,
        traceId=trace_id,
        parentId=parent_id,
        name=name,
        runType=run_type,
        status=status,
        error=error,
        startedAt=started_at or base_time,
        endedAt=ended_at,
        firstTokenAt=first_token_at,
        modelName=model_name,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        totalTokens=total_tokens,
        promptTokenDetails=prompt_token_details,
        inputsJson=inputs,
        outputsJson=outputs,
        eventsJson=events or [],
        extraJson=extra,
        createdAt=base_time,
    )


class TestLocalTracerLifecycle:
    def test_off_backend_is_noop(self, monkeypatch):
        from app.services.chat.local_tracer import LocalTracer

        monkeypatch.setattr(settings, "trace_backend", "off")
        tracer = LocalTracer("hi", "conv1").enter()
        assert tracer.is_active is False
        assert tracer.trace_id is None
        assert tracer.safe_trace_id is None
        tracer.close()  # no-op, 不抛异常

    async def test_enter_writes_root_and_installs_handler(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        fake = _fake_db()
        import app.db
        with patch.object(app.db, "db", fake):
            tracer = local_tracer.LocalTracer("你好", "conv-x").enter()
            assert tracer.trace_id
            assert tracer.safe_trace_id == tracer.trace_id
            assert local_tracer._local_trace_handler.get() is not None
            tracer.close()
            assert local_tracer._local_trace_handler.get() is None
            await asyncio.sleep(0.05)

        # root run 一次 create (status running) + close 一次 update
        assert fake.tracerun.create.await_count == 1
        root_data = fake.tracerun.create.await_args.kwargs["data"]
        assert root_data["id"] == tracer.trace_id
        assert root_data["name"] == "chat_request"
        assert root_data["runType"] == "chain"
        assert fake.tracerun.update.await_count == 1

    async def test_close_is_idempotent(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        fake = _fake_db()
        import app.db
        with patch.object(app.db, "db", fake):
            tracer = local_tracer.LocalTracer("hi", "conv1").enter()
            tracer.close()
            tracer.close()
            tracer.close()
            await asyncio.sleep(0.05)
        assert fake.tracerun.update.await_count == 1

    def test_attach_to_parent_inherits_trace_id(self, monkeypatch):
        from app.services.chat.local_tracer import LocalTracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        tracer = LocalTracer("片段", "conv1").attach_to_parent("parent-id")
        assert tracer.trace_id == "parent-id"
        tracer.close()  # attached 模式不写 root end

    def test_attach_to_parent_none_propagates_none(self, monkeypatch):
        from app.services.chat.local_tracer import LocalTracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        tracer = LocalTracer("x", "conv1").attach_to_parent(None)
        assert tracer.trace_id is None
        tracer.close()


class TestManualRunRecording:
    """Raw-HTTP providers (Ark Responses API) bypass the langchain handler."""

    async def test_manual_run_row_shape_matches_handler_rows(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        fake = _fake_db()
        import app.db
        started = datetime(2026, 7, 25, 8, 0, 0, tzinfo=timezone.utc)
        ended = datetime(2026, 7, 25, 8, 0, 3, tzinfo=timezone.utc)
        with patch.object(app.db, "db", fake):
            tracer = local_tracer.LocalTracer("明天天气", "conv-1").enter()
            local_tracer.record_manual_llm_run(
                name="ArkResponsesAPI",
                model_name="doubao-seed-character-260628",
                provider="ark",
                messages=[
                    {"role": "system", "content": "人设 prompt"},
                    {"role": "user", "content": "明天天气"},
                ],
                output_text="明天晴，31℃",
                started_at=started,
                ended_at=ended,
                input_tokens=4830,
                output_tokens=29,
                cached_input_tokens=0,
                metadata={"web_search_calls": 1},
            )
            await asyncio.sleep(0.05)
            tracer.close()
            await asyncio.sleep(0.05)

        manual = next(
            call.kwargs["data"] for call in fake.tracerun.create.await_args_list
            if call.kwargs["data"].get("name") == "ArkResponsesAPI"
        )
        assert manual["traceId"] == tracer.trace_id
        assert manual["parentId"] == tracer.trace_id  # attached to synthetic root
        assert manual["runType"] == "llm"
        assert manual["status"] == "success"
        assert manual["promptTokens"] == 4830
        assert manual["completionTokens"] == 29
        assert manual["totalTokens"] == 4859
        assert _unjson(manual["promptTokenDetails"]) == {"cache_read": 0}

        # trace_enrich fingerprints on inputs.messages[0][0].kwargs.content —
        # the serialized shape must match what the langchain handler writes.
        inputs = _unjson(manual["inputsJson"])
        assert inputs["messages"][0][0]["kwargs"]["content"] == "人设 prompt"
        assert inputs["messages"][0][0]["kwargs"]["type"] == "system"
        assert inputs["messages"][0][1]["kwargs"]["type"] == "human"
        outputs = _unjson(manual["outputsJson"])
        assert outputs["generations"][0][0]["text"] == "明天晴，31℃"
        extra = _unjson(manual["extraJson"])
        assert extra["metadata"]["ls_provider"] == "ark"
        assert extra["metadata"]["web_search_calls"] == 1

    async def test_manual_run_is_noop_without_open_trace(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        fake = _fake_db()
        import app.db
        now = datetime(2026, 7, 25, 8, 0, 0, tzinfo=timezone.utc)
        with patch.object(app.db, "db", fake):
            # No tracer.enter() → handler ContextVar is unset (background job).
            local_tracer.record_manual_llm_run(
                name="ArkResponsesAPI", model_name="m", messages=[],
                output_text="x", started_at=now, ended_at=now,
            )
            await asyncio.sleep(0.05)
        assert fake.tracerun.create.await_count == 0

    async def test_manual_run_enriches_as_main_reply(self, monkeypatch):
        """End-to-end: a manual row read back must label as the main prompt."""
        from app.services.chat.local_tracer import _row_to_step
        from app.services.chat.trace_enrich import enrich_step
        from app.services.prompting.registry import PROMPT_DEFINITION_MAP

        system_text = PROMPT_DEFINITION_MAP["chat.system_base"].default_text
        row = _row(
            run_id="manual-1",
            name="ArkResponsesAPI",
            model_name="doubao-seed-character-260628",
            inputs={"messages": [[
                {"kwargs": {"type": "system", "content": system_text}},
                {"kwargs": {"type": "human", "content": "明天天气"}},
            ]]},
            outputs={"generations": [[{"text": "明天晴，31℃"}]]},
        )
        step = enrich_step(_row_to_step(row))
        assert step["prompt_key"] == "chat.system_base"
        assert step["decision_label"]


class TestCreateTracerFactory:
    def test_local_backend_returns_local_tracer(self, monkeypatch):
        from app.services.chat.local_tracer import LocalTracer
        from app.services.chat.tracing import create_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        assert isinstance(create_tracer("hi", "c1"), LocalTracer)

    def test_langsmith_backend_returns_langsmith_tracer(self, monkeypatch):
        from app.services.chat.tracing import LangSmithTracer, create_tracer

        monkeypatch.setattr(settings, "trace_backend", "langsmith")
        assert isinstance(create_tracer("hi", "c1"), LangSmithTracer)

    def test_off_backend_returns_inactive_local(self, monkeypatch):
        from app.services.chat.local_tracer import LocalTracer
        from app.services.chat.tracing import create_tracer

        monkeypatch.setattr(settings, "trace_backend", "off")
        tracer = create_tracer("hi", "c1")
        assert isinstance(tracer, LocalTracer)
        assert tracer.is_active is False


class TestCallbackCollection:
    """真实 langchain fake model 走 configure hook → trace_runs 行."""

    async def _run_chat(self, monkeypatch, *, stream: bool):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_backend", "local")
        fake = _fake_db()
        import app.db
        with patch.object(app.db, "db", fake):
            tracer = local_tracer.LocalTracer("用户消息", "conv-cb").enter()
            model = GenericFakeChatModel(messages=iter(["模型回复内容"]))
            if stream:
                async for _ in model.astream("判断记忆相关度"):
                    pass
            else:
                await model.ainvoke("判断记忆相关度")
            tracer.close()
            await asyncio.sleep(0.1)
        return fake, tracer

    async def test_unary_call_persists_llm_run(self, monkeypatch):
        fake, tracer = await self._run_chat(monkeypatch, stream=False)

        # root create + llm run create
        assert fake.tracerun.create.await_count == 2
        llm_create = fake.tracerun.create.await_args_list[1].kwargs["data"]
        assert llm_create["runType"] == "llm"
        assert llm_create["status"] == "running"
        assert llm_create["traceId"] == tracer.trace_id
        # 顶层 langchain run 挂到合成 root
        assert llm_create["parentId"] == tracer.trace_id
        # inputs 是 dumpd 形态: messages[0][0].kwargs.content
        msg = _unjson(llm_create["inputsJson"])["messages"][0][0]
        assert msg["kwargs"]["content"] == "判断记忆相关度"

        # run 结束 upsert 带 outputs generations[0][0].text
        assert fake.tracerun.upsert.await_count == 1
        upsert = fake.tracerun.upsert.await_args.kwargs["data"]["create"]
        assert upsert["status"] == "success"
        assert _unjson(upsert["outputsJson"])["generations"][0][0]["text"] == "模型回复内容"

    async def test_stream_call_records_first_token(self, monkeypatch):
        fake, _ = await self._run_chat(monkeypatch, stream=True)

        upsert = fake.tracerun.upsert.await_args.kwargs["data"]
        assert upsert["update"]["firstTokenAt"] is not None
        # 事件被裁剪: 只保留 start / 首个 new_token / end
        event_names = [e["name"] for e in _unjson(upsert["update"]["eventsJson"])]
        assert event_names.count("new_token") == 1
        assert "start" in event_names and "end" in event_names

    async def test_enrich_extractors_read_stored_shapes(self, monkeypatch):
        """存储的 inputs/outputs 形态必须兼容 trace_enrich 的提取器."""
        from app.services.chat.trace_enrich import (
            _extract_first_user_message,
            _extract_output_text,
        )

        fake, _ = await self._run_chat(monkeypatch, stream=False)
        upsert = fake.tracerun.upsert.await_args.kwargs["data"]["create"]
        assert _extract_first_user_message(_unjson(upsert["inputsJson"])) == "判断记忆相关度"
        assert _extract_output_text(_unjson(upsert["outputsJson"])) == "模型回复内容"


class TestUsageExtraction:
    def test_usage_metadata_path(self):
        from app.services.chat.local_tracer import _extract_usage_from_outputs

        outputs = {
            "generations": [[{
                "text": "hi",
                "message": {"kwargs": {"usage_metadata": {
                    "input_tokens": 900,
                    "output_tokens": 50,
                    "total_tokens": 950,
                    "input_token_details": {"cache_read": 512},
                }}},
            }]],
        }
        usage = _extract_usage_from_outputs(outputs)
        assert usage["prompt_tokens"] == 900
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 950
        assert usage["prompt_token_details"] == {"cache_read": 512}

    def test_llm_output_token_usage_fallback(self):
        from app.services.chat.local_tracer import _extract_usage_from_outputs

        outputs = {
            "generations": [[{"text": "hi"}]],
            "llm_output": {"token_usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 4},
            }},
        }
        usage = _extract_usage_from_outputs(outputs)
        assert usage["total_tokens"] == 15
        assert usage["prompt_token_details"] == {"cache_read": 4}

    def test_no_usage_returns_none_fields(self):
        from app.services.chat.local_tracer import _extract_usage_from_outputs

        usage = _extract_usage_from_outputs({"generations": [[{"text": "x"}]]})
        assert usage["total_tokens"] is None
        assert usage["prompt_token_details"] is None


class TestRowToStepGolden:
    """行 → step 与 public_trace._normalize_step 键集对齐 (前端契约)."""

    def test_step_keys_match_normalize_step(self):
        from app.services.chat.local_tracer import _row_to_step
        from app.services.public_trace import _normalize_step

        langsmith_run = {
            "id": "r1", "name": "ChatOpenAI", "run_type": "llm", "status": "success",
            "parent_run_id": "root", "parent_run_ids": ["root"], "child_run_ids": [],
            "trace_id": "t1", "dotted_order": "x",
            "start_time": "2026-07-19T03:00:00+00:00",
            "end_time": "2026-07-19T03:00:01+00:00",
            "first_token_time": None,
            "inputs": {}, "outputs": {}, "error": None, "events": [],
            "extra": {}, "app_path": "/x",
            "total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0,
            "prompt_token_details": None, "completion_token_details": None,
        }
        expected_keys = set(_normalize_step(langsmith_run).keys())
        local_keys = set(_row_to_step(_row(run_id="r1")).keys())
        assert expected_keys == local_keys

    def test_step_field_values(self):
        from app.services.chat.local_tracer import _row_to_step

        start = datetime(2026, 7, 19, 3, 0, 0, tzinfo=timezone.utc)
        step = _row_to_step(_row(
            run_id="r1",
            parent_id="root-1",
            started_at=start,
            ended_at=start + timedelta(seconds=2),
            first_token_at=start + timedelta(milliseconds=350),
            prompt_token_details={"cache_read": 88},
        ))
        assert step["duration_ms"] == 2000
        assert step["first_token_ms"] == 350
        assert step["parent_id"] == "root-1"
        assert step["prompt_token_details"] == {"cache_read": 88}
        assert step["raw"]["source"] == "local"

    def test_running_row_maps_to_pending(self):
        from app.services.chat.local_tracer import _row_to_step

        step = _row_to_step(_row(run_id="r1", status="running", ended_at=None))
        assert step["status"] == "pending"


class TestTreeAndStale:
    def test_assign_tree_fields_builds_dotted_order_and_children(self):
        from app.services.chat.local_tracer import _assign_tree_fields, _row_to_step

        base = datetime(2026, 7, 19, 3, 0, 0, tzinfo=timezone.utc)
        root = _row_to_step(_row(
            run_id="root", run_type="chain", name="chat_request",
            parent_id=None, started_at=base,
        ))
        child_b = _row_to_step(_row(run_id="b", parent_id="root", started_at=base + timedelta(seconds=2)))
        child_a = _row_to_step(_row(run_id="a", parent_id="root", started_at=base + timedelta(seconds=1)))
        steps = [root, child_b, child_a]
        _assign_tree_fields(steps)

        assert root["child_ids"] == ["b", "a"]  # 填充顺序 = 输入顺序
        assert root["dotted_order"] and "." not in root["dotted_order"]
        assert child_a["dotted_order"].startswith(root["dotted_order"] + ".")
        assert child_b["dotted_order"].startswith(root["dotted_order"] + ".")
        # dotted_order 排序后 a (先开始) 在 b 之前
        ordered = sorted(steps, key=lambda s: s["dotted_order"])
        assert [s["id"] for s in ordered] == ["root", "a", "b"]

    def test_dangling_parent_treated_as_root(self):
        from app.services.chat.local_tracer import _assign_tree_fields, _row_to_step

        orphan = _row_to_step(_row(run_id="o1", parent_id="purged-parent"))
        _assign_tree_fields([orphan])
        assert orphan["dotted_order"]  # 不炸, 按 root 处理
        assert orphan["parent_id"] == "purged-parent"  # 展示字段保留

    def test_mark_stale_running(self):
        from app.services.chat.local_tracer import _mark_stale_running, _row_to_step

        now = datetime(2026, 7, 19, 4, 0, 0, tzinfo=timezone.utc)
        stale = _row_to_step(_row(
            run_id="s1", status="running", started_at=now - timedelta(minutes=30),
        ))
        fresh = _row_to_step(_row(
            run_id="s2", status="running", started_at=now - timedelta(minutes=1),
        ))
        _mark_stale_running([stale, fresh], now=now)
        assert stale["status"] == "cancelled"
        assert fresh["status"] == "pending"


class TestLoadLocalTrace:
    def _rows(self):
        base = datetime(2026, 7, 19, 3, 0, 0, tzinfo=timezone.utc)
        root = _row(
            run_id="trace-1", trace_id="trace-1", parent_id=None,
            name="chat_request", run_type="chain", status="success",
            started_at=base, ended_at=base + timedelta(seconds=8),
            model_name=None, prompt_tokens=None, completion_tokens=None,
            total_tokens=None,
            inputs={"message": "你好呀", "conversation_id": "conv-9"},
        )
        llm = _row(
            run_id="llm-1", trace_id="trace-1", parent_id="trace-1",
            started_at=base + timedelta(seconds=1),
            ended_at=base + timedelta(seconds=3),
            inputs={"messages": [[{"kwargs": {"content": "判断相关度"}}]]},
            outputs={"generations": [[{"text": "强"}]]},
        )
        return [root, llm]

    async def test_builds_detail_shape(self, monkeypatch):
        from app.services.chat import local_tracer

        fake = _fake_db()
        fake.tracerun.find_many = AsyncMock(return_value=self._rows())
        import app.db
        with patch.object(app.db, "db", fake):
            detail = await local_tracer.load_local_trace("trace-1")

        assert detail is not None
        trace = detail["trace"]
        assert trace["source"] == "local"
        assert trace["external_url"] is None
        assert trace["settled"] is True
        assert trace["conversation_id"] == "conv-9"
        assert trace["message"] == "你好呀"
        assert trace["step_count"] == 2
        assert trace["llm_step_count"] == 1
        assert trace["total_tokens"] == 120
        assert trace["duration_ms"] == 8000
        # enrich 已跑: 每个 step 都有 display_name / category
        for step in detail["steps"]:
            assert "display_name" in step and "category" in step

    async def test_unsettled_when_run_pending(self, monkeypatch):
        from app.services.chat import local_tracer

        rows = self._rows()
        rows[1].status = "running"
        rows[1].endedAt = None
        # 刚开始跑 1 分钟, 不到 stale 阈值
        rows[1].startedAt = datetime.now(timezone.utc) - timedelta(minutes=1)
        fake = _fake_db()
        fake.tracerun.find_many = AsyncMock(return_value=rows)
        import app.db
        with patch.object(app.db, "db", fake):
            detail = await local_tracer.load_local_trace("trace-1")
        assert detail["trace"]["settled"] is False

    async def test_returns_none_without_rows(self):
        from app.services.chat import local_tracer

        fake = _fake_db()
        import app.db
        with patch.object(app.db, "db", fake):
            assert await local_tracer.load_local_trace("nope") is None


class TestResolveLocalBranch:
    def _make_msg(self, *, metadata=None, user_id="u1"):
        msg = MagicMock()
        msg.id = "m1"
        msg.metadata = metadata or {"trace_id": "trace-1"}
        conv = MagicMock()
        conv.id = "conv1"
        conv.userId = user_id
        msg.conversation = conv
        return msg

    def _msg_db(self, msg):
        fake = MagicMock()
        fake.message.find_unique = AsyncMock(return_value=msg)
        fake.query_raw = AsyncMock(return_value=[])
        return fake

    async def test_local_rows_resolve_without_share(self, monkeypatch):
        from app.services.chat import tracing

        local_detail = {
            "trace": {"trace_id": "trace-1", "conversation_id": "conv1", "settled": True},
            "steps": [{"id": "s1"}, {"id": "s2"}],
        }
        share_calls = {"n": 0}

        async def fake_share(*a, **k):
            share_calls["n"] += 1
            return "unused"

        mirror_writes = []

        async def fake_write(*, detail, message_id):
            mirror_writes.append(message_id)
            return True

        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=2))
        monkeypatch.setattr(tracing, "load_local_trace", AsyncMock(return_value=local_detail))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=None))
        monkeypatch.setattr(tracing, "share_run_with_retry", fake_share)
        monkeypatch.setattr(tracing, "write_trace_mirror", fake_write)

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")

        assert result["trace_url"] is None
        assert result["detail"]["trace"]["trace_id"] == "trace-1"
        assert share_calls["n"] == 0
        assert mirror_writes == ["m1"]

    async def test_fresh_mirror_short_circuits(self, monkeypatch):
        from app.services.chat import tracing

        cached = {"trace": {"trace_id": "trace-1"}, "steps": [{"id": "a"}, {"id": "b"}]}
        load_calls = {"n": 0}

        async def fake_load_local(_):
            load_calls["n"] += 1
            return None

        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=2))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=cached))
        monkeypatch.setattr(tracing, "load_local_trace", fake_load_local)

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")

        assert result["detail"] is cached
        assert load_calls["n"] == 0

    async def test_stale_mirror_rebuilds_from_rows(self, monkeypatch):
        """后台 run 晚到: 行数 3 > 镜像步数 2 → 重建镜像."""
        from app.services.chat import tracing

        cached = {"trace": {"trace_id": "trace-1"}, "steps": [{"id": "a"}, {"id": "b"}]}
        rebuilt = {
            "trace": {"trace_id": "trace-1", "conversation_id": "conv1", "settled": True},
            "steps": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        }
        mirror_writes = []

        async def fake_write(*, detail, message_id):
            mirror_writes.append(message_id)
            return True

        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=3))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=cached))
        monkeypatch.setattr(tracing, "load_local_trace", AsyncMock(return_value=rebuilt))
        monkeypatch.setattr(tracing, "write_trace_mirror", fake_write)

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")

        assert len(result["detail"]["steps"]) == 3
        assert mirror_writes == ["m1"]

    async def test_unsettled_detail_skips_mirror_write(self, monkeypatch):
        from app.services.chat import tracing

        unsettled = {
            "trace": {"trace_id": "trace-1", "conversation_id": "conv1", "settled": False},
            "steps": [{"id": "a"}],
        }
        mirror_writes = []

        async def fake_write(*, detail, message_id):
            mirror_writes.append(message_id)
            return True

        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=1))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=None))
        monkeypatch.setattr(tracing, "load_local_trace", AsyncMock(return_value=unsettled))
        monkeypatch.setattr(tracing, "write_trace_mirror", fake_write)

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")

        assert result["detail"] is unsettled
        assert mirror_writes == []

    async def test_rows_exist_but_read_fails_raises_runtime_error(self, monkeypatch):
        """count>0 但行读取失败 (DB 瞬时故障) → 503 语义, 不误报 trace_expired."""
        from app.services.chat import tracing

        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=2))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=None))
        monkeypatch.setattr(tracing, "load_local_trace", AsyncMock(return_value=None))

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            with pytest.raises(RuntimeError, match="local_trace_read_failed"):
                await tracing.resolve_trace_for_message("m1", user_id="u1")

    async def test_no_rows_no_url_local_backend_raises_expired(self, monkeypatch):
        from app.services.chat import tracing

        monkeypatch.setattr(settings, "langsmith_tracing", False)
        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=0))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=None))

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            with pytest.raises(ValueError, match="trace_expired"):
                await tracing.resolve_trace_for_message("m1", user_id="u1")

    async def test_no_rows_but_mirror_survives_purge(self, monkeypatch):
        """行被保留期清理但曾查看过 (镜像在) → 仍可打开."""
        from app.services.chat import tracing

        cached = {"trace": {"trace_id": "trace-1"}, "steps": [{"id": "a"}]}
        monkeypatch.setattr(settings, "langsmith_tracing", False)
        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=0))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=cached))

        msg = self._make_msg()
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")
        assert result == {"trace_url": None, "detail": cached}

    async def test_legacy_langsmith_url_path_still_works(self, monkeypatch):
        """老 trace (metadata 带 trace_url, 本地无行) 走 legacy 公开 API 路径."""
        from app.services.chat import tracing

        loaded = {"trace": {"trace_id": "t-legacy"}, "steps": [{"id": "s"}]}
        monkeypatch.setattr(tracing, "count_local_trace_runs", AsyncMock(return_value=0))
        monkeypatch.setattr(tracing, "get_trace_mirror_by_message", AsyncMock(return_value=None))
        monkeypatch.setattr(tracing, "load_public_trace", AsyncMock(return_value=loaded))
        monkeypatch.setattr(tracing, "write_trace_mirror", AsyncMock(return_value=True))

        msg = self._make_msg(metadata={
            "trace_id": "t-legacy", "trace_url": "https://smith.langchain.com/public/x/r",
        })
        with patch.object(tracing, "db", self._msg_db(msg)):
            result = await tracing.resolve_trace_for_message("m1", user_id="u1")
        assert result["trace_url"] == "https://smith.langchain.com/public/x/r"
        assert result["detail"] is loaded


class TestPromptReplayEditorContract:
    """Trace 面板「单步提示词编辑 + 无副作用重跑」的数据契约 (本地采集端到端).

    前端 PromptReplayEditor 依赖链:
      step.inputs.messages[0][0].kwargs.content  = 渲染后的 system prompt 原文
        (extractPromptFromStep — 组件 span 的偏移基准, 不能有任何 trim/改写)
      step.prompt_key / prompt_components[].start/end  = apply_prompt_render_traces
        用渲染期记录的 prompt_hash 精确 join (码点偏移, 前端负责 UTF-16 归一)
      step.inputs 其余 messages  = 对话消息伪 section (历史 + 当前消息)
      detail.trace.conversation_id  = 「对话条数 N」从库补历史的入口
    本测试用生产同款机制 (record_prompt_render + _append_section 语义) 走完
    本地采集 → load_local_trace → _attach_message_trace_metadata 全链路.
    """

    _SECTION_BODIES = {
        "chat.personality_section": "你是小伴，25 岁，喜欢摄影 📷 和爬山。",
        "chat.response_instruction": "回复要求：口语化，禁止旁白。",
    }

    def _build_system_prompt_with_components(self):
        """用 prompt_builder._append_section 的生产逻辑拼 system prompt."""
        from app.services.chat.prompt_builder import _append_section

        sections: list[str] = []
        components: list[dict] = []
        sections.append("你在和用户微信聊天。")  # 无 prompt_key 的开头段
        _append_section(
            sections, components, "你的身份",
            self._SECTION_BODIES["chat.personality_section"],
            prompt_key="chat.personality_section",
        )
        _append_section(
            sections, components, "回复要求",
            self._SECTION_BODIES["chat.response_instruction"],
            prompt_key="chat.response_instruction",
        )
        return "\n\n".join(sections), components

    async def _collect_main_reply_step(self, monkeypatch):
        """模拟主回复调用: record_prompt_render + system/history 消息 LLM 调用."""
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from app.services.chat import local_tracer
        from app.services.prompting.trace_components import (
            record_prompt_render,
            reset_prompt_render_trace,
            snapshot_prompt_render_traces,
            start_prompt_render_trace,
        )

        monkeypatch.setattr(settings, "trace_backend", "local")
        system_prompt, components = self._build_system_prompt_with_components()

        prompt_token = start_prompt_render_trace()
        fake = _fake_db()
        captured_rows: list = []

        async def capture_create(*, data):
            captured_rows.append(("create", data))

        async def capture_upsert(*, where, data):
            captured_rows.append(("upsert", data["create"]))

        fake.tracerun.create = AsyncMock(side_effect=capture_create)
        fake.tracerun.upsert = AsyncMock(side_effect=capture_upsert)

        import app.db
        try:
            with patch.object(app.db, "db", fake):
                # 生产同款: build_system_prompt 尾部记录渲染溯源
                record_prompt_render(
                    system_prompt,
                    prompt_key="chat.system_base",
                    components=components,
                    source="chat.system_prompt",
                )
                tracer = local_tracer.LocalTracer("在干嘛", "conv-editor").enter()
                model = GenericFakeChatModel(messages=iter(["在整理照片呢"]))
                await model.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content="[07-19 10:00] 你好"),
                    AIMessage(content="嗨嗨"),
                    HumanMessage(content="[07-19 10:01] 在干嘛"),
                ])
                tracer.close()
                await asyncio.sleep(0.1)
            render_traces = snapshot_prompt_render_traces()
        finally:
            reset_prompt_render_trace(prompt_token)
        return tracer, captured_rows, render_traces, system_prompt

    def _rows_from_captured(self, tracer, captured_rows):
        """把捕获的写库 payload 还原成 TraceRun 行 (upsert 覆盖 create)."""
        merged: dict[str, dict] = {}
        for _, data in captured_rows:
            row = merged.setdefault(data["id"], {})
            row.update(data)
        rows = []
        for data in merged.values():
            rows.append(_row(
                run_id=data["id"],
                trace_id=data["traceId"],
                parent_id=data.get("parentId"),
                name=data["name"],
                run_type=data["runType"],
                status=data.get("status", "running"),
                started_at=data.get("startedAt"),
                ended_at=data.get("endedAt"),
                model_name=data.get("modelName"),
                prompt_tokens=data.get("promptTokens"),
                completion_tokens=data.get("completionTokens"),
                total_tokens=data.get("totalTokens"),
                inputs=_unjson(data.get("inputsJson")) if data.get("inputsJson") is not None else None,
                outputs=_unjson(data.get("outputsJson")) if data.get("outputsJson") is not None else None,
                events=_unjson(data.get("eventsJson")) if data.get("eventsJson") is not None else [],
            ))
        rows.sort(key=lambda r: (r.startedAt, r.id))
        return rows

    async def test_local_step_carries_editor_contract(self, monkeypatch):
        from app.services.chat import local_tracer
        from app.services.chat.tracing import _attach_message_trace_metadata
        from app.services.chat.trace_enrich import _extract_first_user_message

        tracer, captured_rows, render_traces, system_prompt = (
            await self._collect_main_reply_step(monkeypatch)
        )
        assert render_traces, "渲染溯源必须被记录"

        rows = self._rows_from_captured(tracer, captured_rows)
        fake = _fake_db()
        fake.tracerun.find_many = AsyncMock(return_value=rows)
        import app.db
        with patch.object(app.db, "db", fake):
            detail = await local_tracer.load_local_trace(tracer.trace_id)
        assert detail is not None

        # 生产路径: resolve 时把消息 metadata 的渲染溯源 join 到 steps 上
        _attach_message_trace_metadata(detail, {"prompt_render_traces": render_traces})

        llm_step = next(s for s in detail["steps"] if s["run_type"] == "llm")

        # 1) 待编辑 prompt 原文 = messages[0][0].kwargs.content, 逐字符一致
        #    (extractPromptFromStep 的取值路径; span 偏移全部以它为基准)
        assert _extract_first_user_message(llm_step["inputs"]) == system_prompt

        # 2) hash join 命中: prompt_key + 渲染来源 + admin 元数据
        assert llm_step["prompt_key"] == "chat.system_base"
        assert llm_step["prompt_render_source"] == "chat.system_prompt"
        assert llm_step["prompt_title"]

        # 3) 组件 span 精确切出 section 正文 (Python 码点语义 = 记录时语义)
        comps = llm_step["prompt_components"]
        by_key = {c["prompt_key"]: c for c in comps if "start" in c}
        assert set(by_key) == set(self._SECTION_BODIES)
        for key, body in self._SECTION_BODIES.items():
            comp = by_key[key]
            assert system_prompt[comp["start"]:comp["end"]] == body
            assert comp["editable"] is True
            assert comp["title"]  # _component_admin_meta 补全后台元数据

        # 4) 对话消息伪 section: system 之外的历史消息按原顺序完整保留
        messages = llm_step["inputs"]["messages"][0]
        roles = [m["id"][-1] for m in messages]
        assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
        assert messages[1]["kwargs"]["content"] == "[07-19 10:00] 你好"
        assert messages[3]["kwargs"]["content"] == "[07-19 10:01] 在干嘛"

        # 5) 「对话条数 N」入口依赖 conversation_id
        assert detail["trace"]["conversation_id"] == "conv-editor"

    async def test_astral_chars_keep_span_alignment(self, monkeypatch):
        """section 正文含 emoji (增补平面) 时 span 仍按码点对齐 —
        前端 normalizeComponentSpans 负责转 UTF-16, 后端保证码点正确."""
        tracer, captured_rows, render_traces, system_prompt = (
            await self._collect_main_reply_step(monkeypatch)
        )
        components = render_traces[0]["components"]
        photo_section = next(
            c for c in components if c["prompt_key"] == "chat.personality_section"
        )
        body = self._SECTION_BODIES["chat.personality_section"]
        assert "📷" in body
        assert system_prompt[photo_section["start"]:photo_section["end"]] == body


class TestRetention:
    async def test_purge_deletes_old_rows(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_retention_days", 30)
        fake = _fake_db()
        fake.tracerun.delete_many = AsyncMock(return_value=42)
        import app.db
        with patch.object(app.db, "db", fake):
            deleted = await local_tracer.purge_expired_trace_runs()

        assert deleted == 42
        where = fake.tracerun.delete_many.await_args.kwargs["where"]
        cutoff = where["createdAt"]["lt"]
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((cutoff - expected).total_seconds()) < 5

    async def test_purge_disabled_with_nonpositive_days(self, monkeypatch):
        from app.services.chat import local_tracer

        monkeypatch.setattr(settings, "trace_retention_days", 0)
        fake = _fake_db()
        import app.db
        with patch.object(app.db, "db", fake):
            assert await local_tracer.purge_expired_trace_runs() == 0
        assert fake.tracerun.delete_many.await_count == 0
