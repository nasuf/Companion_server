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
