"""普通回复的追问判断只管可选追问: 危机照护、兜底回复、必要确认轮次不进判断 (M02.20.05)。"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.chat import graph_phases
from app.services.time_domain import followup_decision

NOW = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


class _Stop(Exception):
    """Raised right after the follow-up block so the rest of the phase never runs."""


def _ctx(**overrides):
    fields = dict(
        services=SimpleNamespace(logger=SimpleNamespace(warning=lambda *a, **k: None)),
        response_diagnostics={"temporal_context": {}},
        continuation_lines=[], reply_is_fallback=False, crisis_care_turn=False,
        contradiction_inquiry=None, replies=["面试结果出来了吗？"], user_message="嗯",
        agent=None, agent_id="a1", location_context=None, ai_status=None, schedule=None,
        messages_dicts=[], classified_memories=[], time_memories=[], l3_memories=[],
        turn_clock=SimpleNamespace(now=NOW), followup_policy=object(), user_id="u1",
        workspace_id="w1", reply_emotion_pre=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    async def unchanged_persona(_agent, text, **_k):
        return text, False

    async def unchanged_temporal(text, **_k):
        return text

    async def guard(replies, **kwargs):
        seen.append(kwargs["current_user_message"])
        return replies, followup_decision.summarize([], status="checked")

    def stop(_ctx):
        raise _Stop

    monkeypatch.setattr("app.services.persona_grounding.guard_reply", unchanged_persona)
    monkeypatch.setattr("app.services.time_domain.guard.guard_temporal_reply", unchanged_temporal)
    monkeypatch.setattr(graph_phases, "guard_optional_questions", lambda replies, _p: (replies, 0))
    monkeypatch.setattr(followup_decision, "guard_reply_followups", guard)
    monkeypatch.setattr(graph_phases.repetition_review, "drop_restated", stop)
    return seen


async def test_ordinary_turn_runs_the_shared_followup_decision(calls):
    ctx = _ctx()
    with pytest.raises(_Stop):
        async for _ in graph_phases.normalize_reply(ctx):
            pass
    assert calls == ["嗯"]
    assert ctx.response_diagnostics["temporal_context"]["followup_decision"]["status"] == "checked"


@pytest.mark.parametrize("turn", [
    {"crisis_care_turn": True},
    {"reply_is_fallback": True},
    {"contradiction_inquiry": {"memory_id": "m1"}},  # 必要确认: 问用户哪条是对的
])
async def test_crisis_fallback_and_required_confirmation_never_reach_it(calls, turn):
    ctx = _ctx(**turn)
    with pytest.raises(_Stop):
        async for _ in graph_phases.normalize_reply(ctx):
            pass
    assert calls == [] and ctx.replies == ["面试结果出来了吗？"]
    assert "followup_decision" not in ctx.response_diagnostics["temporal_context"]
