"""中/低耐心态也要接住道歉.

原来 `detect_apology` 只在 blocked 态调用, 于是用户在余波期说「对不起」会被当成
普通消息, 再走一遍余波模板。生产实录 (2026-07-31):

    用户「对不起」→ AI「没事啦 其实我还有点不开心 希望以后别再那样」

既说没事又说不开心, 而且对话再也出不来 —— 后面两轮它还在重复同一句。

道歉是用户主动修复关系的动作, 不接住比机械回复更伤; 也让「道歉能恢复耐心」这条
产品设定在被拉黑之前就成立, 而不是非得先闹到拉黑。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.chat import boundary_phase as bp


def _ctx(message: str):
    fired: list = []
    return SimpleNamespace(
        user_message=message,
        agent_id="a1",
        user_id="u1",
        conversation_id="c1",
        agent=SimpleNamespace(name="小伴"),
        tracer=SimpleNamespace(close=lambda: None, safe_trace_id=None),
        fire_background_fn=lambda coro: fired.append(coro),
        bg_memory_pipeline_fn=lambda *a, **k: ("mem", a, k),
        stopped=False,
        cached_patience=50,
        last_short_circuit_reply=None,
        _fired=fired,
    )


async def _drain(gen):
    return [e async for e in gen]


@pytest.mark.asyncio
class TestApologyAccepted:
    async def test_apology_short_circuits_with_acceptance(self, monkeypatch):
        monkeypatch.setattr(
            bp, "detect_apology",
            AsyncMock(return_value={"is_apology": True, "sincerity": 0.9}),
        )
        monkeypatch.setattr(bp, "handle_apology", AsyncMock(return_value=70))
        monkeypatch.setattr(bp, "apology_reply", AsyncMock(return_value="嗯，翻篇啦"))
        emitted: list = []

        async def fake_emit(ctx, reply, meta):
            emitted.append((reply, meta))
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        monkeypatch.setattr(bp, "_fire_memory_pipeline", lambda ctx, reply: None)

        ctx = _ctx("对不起，刚才是我不好")
        await _drain(bp._handle_residual_patience(ctx, "medium"))

        assert ctx.stopped is True, "接住道歉后必须短路, 否则还会再走一遍余波模板"
        assert emitted and emitted[0][0] == "嗯，翻篇啦"
        assert emitted[0][1]["apology_accepted"] is True

    async def test_patience_is_restored(self, monkeypatch):
        restore = AsyncMock(return_value=70)
        monkeypatch.setattr(
            bp, "detect_apology",
            AsyncMock(return_value={"is_apology": True, "sincerity": 0.8}),
        )
        monkeypatch.setattr(bp, "handle_apology", restore)
        monkeypatch.setattr(bp, "apology_reply", AsyncMock(return_value="好啦"))

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        monkeypatch.setattr(bp, "_fire_memory_pipeline", lambda ctx, reply: None)

        await _drain(bp._handle_residual_patience(_ctx("抱歉"), "medium"))
        restore.assert_awaited_once()

    async def test_apology_enters_the_memory_pipeline(self, monkeypatch):
        """道歉是有价值的关系事件, 该被记住 (与 blocked 态一致)."""
        monkeypatch.setattr(
            bp, "detect_apology",
            AsyncMock(return_value={"is_apology": True, "sincerity": 0.9}),
        )
        monkeypatch.setattr(bp, "handle_apology", AsyncMock(return_value=70))
        monkeypatch.setattr(bp, "apology_reply", AsyncMock(return_value="嗯"))

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        seen: list = []
        monkeypatch.setattr(bp, "_fire_memory_pipeline", lambda ctx, r: seen.append(r))

        await _drain(bp._handle_residual_patience(_ctx("对不起"), "medium"))
        assert seen == ["嗯"]


@pytest.mark.asyncio
class TestGating:
    async def test_no_llm_call_without_apology_keywords(self, monkeypatch):
        """余波轮本来就要调一次 LLM 生成回复; 无条件再加一次检测会让延迟翻倍,
        而绝大多数余波消息根本不是道歉。"""
        detect = AsyncMock(return_value={"is_apology": False})
        monkeypatch.setattr(bp, "detect_apology", detect)
        monkeypatch.setattr(
            bp, "generate_boundary_reply_llm", AsyncMock(return_value="嗯"),
        )

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        await _drain(bp._handle_residual_patience(_ctx("今天天气不错"), "medium"))
        detect.assert_not_awaited()

    async def test_keyword_present_triggers_detection(self, monkeypatch):
        detect = AsyncMock(return_value={"is_apology": False, "sincerity": 0.0})
        monkeypatch.setattr(bp, "detect_apology", detect)
        monkeypatch.setattr(
            bp, "generate_boundary_reply_llm", AsyncMock(return_value="嗯"),
        )

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        await _drain(bp._handle_residual_patience(_ctx("对不起啦"), "medium"))
        detect.assert_awaited_once()

    async def test_insincere_apology_falls_through(self, monkeypatch):
        """阈值不到就走原路径 —— 不能让一句敷衍的"对不起"直接重置耐心."""
        monkeypatch.setattr(
            bp, "detect_apology",
            AsyncMock(return_value={"is_apology": True, "sincerity": 0.2}),
        )
        restore = AsyncMock(return_value=70)
        monkeypatch.setattr(bp, "handle_apology", restore)
        monkeypatch.setattr(
            bp, "generate_boundary_reply_llm", AsyncMock(return_value="嗯"),
        )

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        await _drain(bp._handle_residual_patience(_ctx("对不起"), "medium"))
        restore.assert_not_awaited()

    async def test_detection_failure_does_not_break_the_turn(self, monkeypatch):
        """检测挂了要退回原路径, 不能让这一轮没有回复."""
        monkeypatch.setattr(bp, "detect_apology", AsyncMock(side_effect=RuntimeError("down")))
        gen = AsyncMock(return_value="嗯")
        monkeypatch.setattr(bp, "generate_boundary_reply_llm", gen)

        async def fake_emit(ctx, reply, meta):
            if False:
                yield {}

        monkeypatch.setattr(bp, "_emit_short_circuit", fake_emit)
        await _drain(bp._handle_residual_patience(_ctx("对不起"), "medium"))
        gen.assert_awaited_once()
