"""热路径调用方必须把外层超时传成调用期限。

外层 wait_for 只会在到点时取消, 取消落在重试退避或兜底模型中途; 期限让 resilience
层提前停下, 半开试探名额也能归还。期限耗尽与超时同样只是"少一句", 不能往上抛。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.llm import resilience
from app.services.llm.resilience import LLMDeadlineExceeded


def _capture(seen: list, *, result="ok", exc: Exception | None = None):
    async def _render(*_args, **_kwargs):
        seen.append(resilience._deadline.get())
        if exc is not None:
            raise exc
        return result

    return _render


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [None, LLMDeadlineExceeded("deadline")])
async def test_topic_continuation_generate_sets_deadline(exc):
    from app.services.chat import topic_continuation as tc

    seen: list = []
    with patch.object(tc, "render_prompt", side_effect=_capture(seen, result="嗯", exc=exc)):
        out = await tc._generate("chat.topic_continuation_return", {})

    assert seen and seen[0] is not None
    assert out == (None if exc else "嗯")
    assert resilience._deadline.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [None, LLMDeadlineExceeded("deadline")])
async def test_trending_pick_sets_deadline(exc):
    from app.services.proactive import trending_pick as tp

    seen: list = []
    with patch.object(tp, "render_prompt", side_effect=_capture(seen, result={"index": 1}, exc=exc)):
        out = await tp._run_pick("proactive.trending_pick_random", {})

    assert seen and seen[0] is not None
    assert out == (None if exc else {"index": 1})


@pytest.mark.asyncio
async def test_topic_judge_deadline_exhaustion_is_no_verdict():
    from app.services.proactive import followup as fu

    seen: list = []
    with patch.object(fu, "render_prompt",
                      side_effect=_capture(seen, exc=LLMDeadlineExceeded("deadline"))):
        verdict = await fu.judge_topic_completion(
            [], arm_reason="normal", now=datetime.now(timezone.utc),
        )

    assert seen and seen[0] is not None
    assert verdict is None


@pytest.mark.asyncio
async def test_followup_generation_deadline_exhaustion_sends_nothing():
    from app.services.proactive import followup as fu

    now = datetime.now(timezone.utc)
    seen: list = []
    policy = SimpleNamespace(optional_questions_allowed=True, prompt_value=lambda: "",
                             diagnostics=lambda: {})
    clock = SimpleNamespace(diagnostics=lambda: {})
    with (
        patch.object(fu, "_current_session", return_value=["turn"]),
        patch.object(fu, "_turn_messages", return_value=[]),
        patch.object(fu, "_last_time", return_value=""),
        patch.object(fu, "derive_followup_policy", return_value=policy),
        patch("app.services.time_domain.topic_store.resolve_policy",
              new_callable=AsyncMock, return_value=policy),
        patch.object(fu, "build_turn_clock", return_value=clock),
        patch.object(fu, "render_temporal_context", new_callable=AsyncMock, return_value=""),
        patch.object(fu, "db", SimpleNamespace(aiagent=SimpleNamespace(
            find_unique=AsyncMock(return_value=SimpleNamespace(id="a1"))))),
        patch("app.services.mbti.build_personality_brief", return_value="温和"),
        patch.object(fu, "render_prompt",
                     side_effect=_capture(seen, exc=LLMDeadlineExceeded("deadline"))),
    ):
        state = SimpleNamespace(user_id="u1", workspace_id="w1", agent_id="a1")
        out = await fu.generate_followup_message(state, ["turn"], now)

    assert seen and seen[0] is not None
    assert out is None
