"""LangSmithTracer 单测：disabled / enabled / 幂等 close。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_tracer_disabled_no_op():
    """settings.langsmith_tracing=False → enter 后 trace_id=None，close 不报错。"""
    from app.services.chat import tracing

    with patch.object(tracing.settings, "langsmith_tracing", False):
        tracer = tracing.LangSmithTracer("hi", "conv1").enter()
        assert tracer.is_active is False
        assert tracer.trace_id is None
        # close 不应抛异常，且不调 share
        with patch.object(tracing, "_fire_background") as fbg:
            tracer.close()
            fbg.assert_not_called()


def test_tracer_enabled_returns_trace_id_and_fires_share():
    """settings.langsmith_tracing=True → trace_id 取自 ls_trace 的 run_tree.id；
    close() 触发后台 _fire_background(self._share())。"""
    from app.services.chat import tracing

    fake_run_tree = MagicMock(id="trace-xyz-123")
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_run_tree)
    fake_ctx.__exit__ = MagicMock(return_value=None)

    fake_ls_trace = MagicMock(return_value=fake_ctx)

    with patch.object(tracing.settings, "langsmith_tracing", True), \
         patch.dict("sys.modules", {"langsmith": MagicMock(trace=fake_ls_trace)}), \
         patch.object(tracing, "_fire_background") as fbg:
        tracer = tracing.LangSmithTracer("hi", "conv1").enter()
        assert tracer.trace_id == "trace-xyz-123"
        tracer.close()
        # 应 fire 一次后台 share 任务
        fbg.assert_called_once()


def test_tracer_close_is_idempotent():
    """重复调 close 不应导致 ctx.__exit__ 被多次调用或 share 被多次 fire。"""
    from app.services.chat import tracing

    fake_run_tree = MagicMock(id="t-1")
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_run_tree)
    fake_ctx.__exit__ = MagicMock(return_value=None)

    with patch.object(tracing.settings, "langsmith_tracing", True), \
         patch.dict("sys.modules", {"langsmith": MagicMock(trace=MagicMock(return_value=fake_ctx))}), \
         patch.object(tracing, "_fire_background") as fbg:
        tracer = tracing.LangSmithTracer("hi", "conv1").enter()
        tracer.close()
        tracer.close()  # 第二次 no-op
        tracer.close()
        assert fake_ctx.__exit__.call_count == 1
        assert fbg.call_count == 1


def test_run_tree_none_logs_warning(caplog):
    """is_active=True 但 ls_trace 返回的 run_tree 为 None (SDK 初始化失败) → 
    应 log warning 提示 trace 缺失, trace_id 仍为 None."""
    import logging
    from app.services.chat import tracing

    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=None)  # SDK init 失败
    fake_ctx.__exit__ = MagicMock(return_value=None)

    with patch.object(tracing.settings, "langsmith_tracing", True), \
         patch.dict("sys.modules", {"langsmith": MagicMock(trace=MagicMock(return_value=fake_ctx))}), \
         caplog.at_level(logging.WARNING, logger="app.services.chat.tracing"):
        tracer = tracing.LangSmithTracer("hi", "conv12345678").enter()
        assert tracer.trace_id is None
        # Warning 应包含诊断关键词
        warns = [r for r in caplog.records if "no run_tree" in r.message]
        assert warns, f"expected warning, got: {[r.message for r in caplog.records]}"


class TestShareRunRetry:
    """share_run 重试逻辑 (3 次指数退避 1s/2s/4s)."""

    def _build_tracer_with_share_outcome(self, share_side_effect):
        """构造一个 trace_id=已知, get_langsmith_client 返回 mock client 的 tracer."""
        from app.services.chat import tracing

        tracer = tracing.LangSmithTracer.__new__(tracing.LangSmithTracer)
        tracer._user_message = "hi"
        tracer._conversation_id = "conv-share-test"
        tracer._ctx = MagicMock()
        tracer._closed = False
        tracer.trace_id = "trace-abc"

        client = MagicMock()
        client.share_run = MagicMock(side_effect=share_side_effect)
        return tracer, client

    async def _invoke_share_run(self, tracer, client, monkeypatch):
        from app.services.chat import tracing
        monkeypatch.setattr(tracing, "get_langsmith_client", lambda: client)
        # 跳过 sleep 加快测试
        async def _instant_sleep(_): return None
        monkeypatch.setattr(tracing.asyncio, "sleep", _instant_sleep)
        return await tracing.share_run_with_retry(
            tracer.trace_id, conversation_id=tracer._conversation_id,
        )

    async def test_first_attempt_succeeds(self, monkeypatch):
        tracer, client = self._build_tracer_with_share_outcome(["https://smith/abc"])
        url = await self._invoke_share_run(tracer, client, monkeypatch)
        assert url == "https://smith/abc"
        assert client.share_run.call_count == 1

    async def test_succeeds_on_third_attempt(self, monkeypatch):
        outcomes = [RuntimeError("503"), RuntimeError("network"), "https://smith/ok"]
        tracer, client = self._build_tracer_with_share_outcome(outcomes)
        url = await self._invoke_share_run(tracer, client, monkeypatch)
        assert url == "https://smith/ok"
        assert client.share_run.call_count == 3

    async def test_all_retries_fail_raises(self, monkeypatch):
        import pytest
        outcomes = [RuntimeError("e1"), RuntimeError("e2"), RuntimeError("e3")]
        tracer, client = self._build_tracer_with_share_outcome(outcomes)
        with pytest.raises(RuntimeError, match="e3"):
            await self._invoke_share_run(tracer, client, monkeypatch)
        assert client.share_run.call_count == 3

    async def test_share_failure_logs_with_stack(self, monkeypatch, caplog):
        """share_run 全部失败 → _share 应 logger.exception (含 stack trace), 并触发
        _mark_trace_failed 让前端能显示重试按钮."""
        import logging
        from app.services.chat import tracing

        tracer, client = self._build_tracer_with_share_outcome(
            [RuntimeError("e1"), RuntimeError("e2"), RuntimeError("e3")]
        )
        monkeypatch.setattr(tracing, "get_langsmith_client", lambda: client)
        async def _instant_sleep(_): return None
        monkeypatch.setattr(tracing.asyncio, "sleep", _instant_sleep)
        # 截获 mark_trace_failed 调用, 不真打 DB.
        called: dict[str, object] = {}
        async def fake_mark_failed(conv_id, trace_id):
            called["conv"] = conv_id
            called["trace"] = trace_id
        monkeypatch.setattr(tracing, "_mark_trace_failed", fake_mark_failed)

        with caplog.at_level(logging.WARNING, logger="app.services.chat.tracing"):
            await tracer._share()

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors, "expected logger.exception (ERROR-level) for permanent failure"
        assert errors[0].exc_info is not None, "should include stack trace"
        assert called == {"conv": "conv-share-test", "trace": "trace-abc"}, \
            "permanent failure should trigger _mark_trace_failed"


import pytest as _pytest

# Mark async tests
TestShareRunRetry.test_first_attempt_succeeds = _pytest.mark.asyncio(TestShareRunRetry.test_first_attempt_succeeds)
TestShareRunRetry.test_succeeds_on_third_attempt = _pytest.mark.asyncio(TestShareRunRetry.test_succeeds_on_third_attempt)
TestShareRunRetry.test_all_retries_fail_raises = _pytest.mark.asyncio(TestShareRunRetry.test_all_retries_fail_raises)
TestShareRunRetry.test_share_failure_logs_with_stack = _pytest.mark.asyncio(TestShareRunRetry.test_share_failure_logs_with_stack)


class TestRetryShareForMessage:
    """retry_share_for_message: 重试单条消息 trace 分享 (供 /traces/retry endpoint).

    覆盖关键场景:
    - 消息不存在 → ValueError("message_not_found")
    - 跨用户访问 → PermissionError
    - 老消息没 trace_id → ValueError("no_trace_id")
    - 已有 trace_url → 直接返回 (幂等)
    - share_run 成功 → 更新 metadata + WS 通知
    """

    def _make_msg(self, *, msg_id="m1", user_id="u1", metadata=None):
        msg = MagicMock()
        msg.id = msg_id
        msg.metadata = metadata
        conv = MagicMock()
        conv.id = "conv1"
        conv.userId = user_id
        msg.conversation = conv
        return msg

    def _fake_db(self, msg):
        """Build a fake db whose `message.find_unique` returns the given msg."""
        from unittest.mock import AsyncMock
        fake = MagicMock()
        fake.message.find_unique = AsyncMock(return_value=msg)
        return fake

    @_pytest.mark.asyncio
    async def test_message_not_found(self):
        import pytest
        from app.services.chat import tracing
        with patch.object(tracing, "db", self._fake_db(None)):
            with pytest.raises(ValueError, match="message_not_found"):
                await tracing.retry_share_for_message("missing", user_id="u1")

    @_pytest.mark.asyncio
    async def test_cross_user_rejected(self):
        import pytest
        from app.services.chat import tracing
        msg = self._make_msg(user_id="other_user")
        with patch.object(tracing, "db", self._fake_db(msg)):
            with pytest.raises(PermissionError, match="not_your_message"):
                await tracing.retry_share_for_message("m1", user_id="u1")

    @_pytest.mark.asyncio
    async def test_no_trace_id_in_metadata(self):
        import pytest
        from app.services.chat import tracing
        msg = self._make_msg(metadata={"reply_index": 0})  # 老消息无 trace_id
        with patch.object(tracing, "db", self._fake_db(msg)):
            with pytest.raises(ValueError, match="no_trace_id"):
                await tracing.retry_share_for_message("m1", user_id="u1")

    @_pytest.mark.asyncio
    async def test_already_has_url_short_circuits(self, monkeypatch):
        from app.services.chat import tracing
        msg = self._make_msg(metadata={"trace_id": "t1", "trace_url": "https://existing"})
        called = {"share": 0}
        async def fake_share(*a, **k):
            called["share"] += 1
            return "should_not_be_called"
        monkeypatch.setattr(tracing, "share_run_with_retry", fake_share)

        with patch.object(tracing, "db", self._fake_db(msg)):
            result = await tracing.retry_share_for_message("m1", user_id="u1")
        assert result == {"trace_url": "https://existing"}
        assert called["share"] == 0

    @_pytest.mark.asyncio
    async def test_success_updates_metadata_and_pushes_ws(self, monkeypatch):
        from app.services.chat import tracing
        msg = self._make_msg(metadata={"trace_id": "t1", "trace_pending": True})

        async def fake_share(*a, **k): return "https://smith/new"
        monkeypatch.setattr(tracing, "share_run_with_retry", fake_share)

        captured = {}
        async def fake_patch(message_id, current, **patch_):
            captured["patch"] = patch_
            captured["msg_id"] = message_id
        monkeypatch.setattr(tracing, "_patch_message_metadata", fake_patch)

        ws_calls = []
        class FakeManager:
            async def send_event(self, conv_id, event, payload):
                ws_calls.append((conv_id, event, payload))
        from app.services.runtime import ws_manager
        monkeypatch.setattr(ws_manager, "manager", FakeManager())

        with patch.object(tracing, "db", self._fake_db(msg)):
            result = await tracing.retry_share_for_message("m1", user_id="u1")
        assert result == {"trace_url": "https://smith/new"}
        assert captured["patch"]["trace_url"] == "https://smith/new"
        assert captured["patch"]["trace_pending"] is False
        assert captured["patch"]["trace_failed"] is False
        assert ws_calls and ws_calls[0][1] == "trace_ready"
