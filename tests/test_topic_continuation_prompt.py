"""被动回复高级承接: 「话题接续」段注入 + 与重逢感知的互斥 (《主动聊天机制（新增）》)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.interaction.topic_continuity import (
    CUE_JUMP_KEY,
    CUE_RETURN_KEY,
    FOLLOWUP_TRIGGER_TYPE,
    ContinuationCue,
    ContinuityState,
    TopicVerdict,
)

UTC = timezone.utc


async def _prompt_text(key: str, **_kwargs) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


async def _build(**kwargs) -> str:
    from app.services.chat.prompt_builder import build_system_prompt

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text", AsyncMock(side_effect=_prompt_text)),
        patch(
            "app.services.chat.prompt_builder.get_prompt_text_or_default",
            AsyncMock(side_effect=_prompt_text),
        ),
    ):
        return await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_return_cue_replaces_short_reengagement():
    cue = ContinuationCue(CUE_RETURN_KEY, "周末去哪玩", gap_seconds=40 * 60)
    prompt = await _build(reengagement_gap_seconds=40 * 60, topic_continuation=cue)
    assert "## 话题接续" in prompt
    assert "过了约40 分钟" in prompt
    assert "「周末去哪玩」" in prompt
    assert "## 重逢感知" not in prompt  # 不叠两层"回来了怎么接"


@pytest.mark.asyncio
async def test_jump_cue_is_injected_without_return_phrase():
    cue = ContinuationCue(CUE_JUMP_KEY, "面试结果", gap_seconds=7 * 60)
    prompt = await _build(reengagement_gap_seconds=7 * 60, topic_continuation=cue)
    assert "## 话题接续" in prompt
    assert "你们刚才聊到「面试结果」" in prompt
    assert "忙完啦" not in prompt


@pytest.mark.asyncio
async def test_finished_topic_suppresses_reengagement_without_section():
    """spec: 上一轮已完结 → 无论隔多久都不带承接话术."""
    cue = ContinuationCue(None, "", gap_seconds=90 * 60)
    prompt = await _build(reengagement_gap_seconds=90 * 60, topic_continuation=cue)
    assert "## 话题接续" not in prompt
    assert "## 重逢感知" not in prompt


@pytest.mark.asyncio
async def test_no_cue_keeps_existing_reengagement():
    prompt = await _build(reengagement_gap_seconds=90 * 60)
    assert "## 重逢感知" in prompt
    assert "## 话题接续" not in prompt


@pytest.mark.asyncio
async def test_offering_turn_ignores_cue():
    """红包/礼物本身就是新话题."""
    cue = ContinuationCue(CUE_RETURN_KEY, "周末去哪玩", gap_seconds=40 * 60)
    prompt = await _build(
        topic_continuation=cue,
        red_packet_context={"offering_id": "rp-1", "amount": 5},
    )
    assert "## 话题接续" not in prompt


@pytest.mark.asyncio
async def test_disabled_template_drops_section(monkeypatch):
    from app.services.chat import prompt_builder

    monkeypatch.setattr(prompt_builder, "_get_optional_prompt", AsyncMock(return_value=None))
    section = await prompt_builder._build_topic_continuation_section(
        ContinuationCue(CUE_JUMP_KEY, "x", gap_seconds=60),
    )
    assert section is None


# ── orchestrator: 读判定 → cue ─────────────────────────────────────────

def _continuity(**kw) -> ContinuityState:
    base = dict(
        verdict=TopicVerdict("unfinished", "ai_question", "周末去哪玩"),
        anchor_message_id="ai-1",
        followup_sent_at=None,
        last_user_at=None,
        session_closed=False,
    )
    base.update(kw)
    return ContinuityState(**base)  # type: ignore[arg-type]


async def _cue(previous_assistant, continuity, *, offering_turn=False, replied_at=None,
               patience_low=False):
    import asyncio

    from app.services.chat.orchestrator import _resolve_topic_cue

    async def _load():
        return continuity

    diagnostics: dict = {}
    cue = await _resolve_topic_cue(
        asyncio.ensure_future(_load()),
        previous_assistant=previous_assistant,
        replied_at=replied_at,
        offering_turn=offering_turn,
        patience_low=patience_low,
        response_diagnostics=diagnostics,
    )
    return cue, diagnostics


def _assistant(minutes_ago: float, *, id_: str = "ai-1", metadata=None):
    return SimpleNamespace(
        id=id_,
        createdAt=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_orchestrator_builds_return_cue_from_previous_ai_message():
    cue, diagnostics = await _cue(_assistant(25), _continuity())
    assert cue.template_key == CUE_RETURN_KEY
    assert 24 * 60 < cue.gap_seconds < 26 * 60
    assert diagnostics["topic_continuation"] == CUE_RETURN_KEY


@pytest.mark.asyncio
async def test_orchestrator_recognises_reply_to_followup():
    previous = _assistant(40, id_="b-1", metadata={"trigger_type": FOLLOWUP_TRIGGER_TYPE})
    cue, diagnostics = await _cue(previous, _continuity())
    assert cue.template_key is None
    assert diagnostics["topic_continuation"] == "finished"


@pytest.mark.asyncio
async def test_orchestrator_cue_skipped_on_offering_and_load_failure():
    import asyncio

    from app.services.chat.orchestrator import _resolve_topic_cue

    cue, _ = await _cue(_assistant(25), _continuity(), offering_turn=True)
    assert cue is None

    async def _boom():
        raise RuntimeError("redis down")

    common = dict(replied_at=None, offering_turn=False, patience_low=False, response_diagnostics={})
    assert await _resolve_topic_cue(
        asyncio.ensure_future(_boom()), previous_assistant=_assistant(25), **common,
    ) is None
    assert await _resolve_topic_cue(None, previous_assistant=_assistant(25), **common) is None

    cancelled = asyncio.ensure_future(asyncio.sleep(10))
    cancelled.cancel()
    await asyncio.sleep(0)
    assert await _resolve_topic_cue(cancelled, previous_assistant=_assistant(25), **common) is None


@pytest.mark.asyncio
async def test_gap_is_measured_to_user_reply_not_generation_time():
    """聚合/延迟队列下生成比用户发消息晚: 8 分钟就回的不能被算成 >10 分钟."""
    previous = _assistant(12)
    replied_at = datetime.now(UTC) - timedelta(minutes=4)  # 用户 8 分钟时就回了
    cue, _ = await _cue(previous, _continuity(), replied_at=replied_at)
    assert cue.template_key == CUE_JUMP_KEY
    assert 7 * 60 < cue.gap_seconds < 9 * 60


@pytest.mark.asyncio
async def test_no_cue_while_ai_is_still_annoyed():
    cue, _ = await _cue(_assistant(25), _continuity(), patience_low=True)
    assert cue is None


def test_turn_started_at_takes_earliest_fragment():
    from app.services.chat.orchestrator import _turn_started_at

    early = datetime(2026, 9, 29, 4, 0, tzinfo=UTC)
    messages = [
        {"id": "old", "role": "user", "createdAt": (early - timedelta(hours=1)).isoformat()},
        {"id": "f1", "role": "user", "createdAt": early.isoformat()},
        {"id": "f2", "role": "user", "createdAt": (early + timedelta(seconds=3)).isoformat()},
    ]
    assert _turn_started_at(messages, {"f1", "f2"}) == early
    assert _turn_started_at(messages, set()) is None
