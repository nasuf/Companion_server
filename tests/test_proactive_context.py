"""Spec §3.2 + §4.2 主动记忆按 topic 过滤 + spec §2.1 stage 4 档单测."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _row(rid: str, text: str, importance: float = 0.5):
    return SimpleNamespace(
        id=rid, summary=text, content=text,
        mainCategory="生活", subCategory="日常",
        importance=importance,
    )


@pytest.mark.asyncio
async def test_load_proactive_memories_uses_rerank_when_topic_present():
    """topic_theme 非空 → utility model rerank 选 ≤3 条按返回顺序排."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "下雨天"), _row("m2", "看了部电影"), _row("m3", "新菜谱")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock, return_value={"ids": ["m2", "m1"]}),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="电影",
        )

    # 按 rerank 顺序: m2 在前
    assert ids == ["m2", "m1"]
    assert "看了部电影" in texts[0]


@pytest.mark.asyncio
async def test_load_proactive_memories_falls_back_to_importance_when_rerank_returns_none():
    """render_prompt 失败返回 None → 回退到 importance 倒排, 不阻塞主动消息发送."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "记忆 A"), _row("m2", "记忆 B")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock, return_value=None),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="任意",
        )

    assert ids == ["m1", "m2"]
    assert len(texts) == 2


@pytest.mark.asyncio
async def test_load_proactive_memories_filters_hallucinated_ids():
    """LLM 返回不在候选集里的 id (幻觉) → 仅保留 valid 子集."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "A"), _row("m2", "B")]
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt",
              new_callable=AsyncMock,
              return_value={"ids": ["m1", "ghost-id-not-in-candidates"]}),
    ):
        texts, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="任意",
        )

    assert ids == ["m1"]


@pytest.mark.asyncio
async def test_prepare_music_recommendation_fetches_track_when_not_co_listening():
    from app.services.proactive import sender
    from app.models.music import MusicTrack

    ctx = {}
    track = MusicTrack(id="track-1", title="Quiet Realm")
    with (
        patch(
            "app.services.music.get_open_co_listening",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.services.music.default_libraries", return_value=["focus"]),
        patch(
            "app.services.music.fetch_random_track",
            new_callable=AsyncMock,
            return_value=track,
        ) as fetch_track,
    ):
        source = await sender._prepare_music_recommendation_source(
            ctx,
            conversation_id="conv-1",
        )

    assert source == "music"
    assert ctx["music_track"] is track
    fetch_track.assert_awaited_once()


@pytest.mark.asyncio
async def test_prepare_music_recommendation_falls_back_when_already_co_listening():
    from app.services.proactive import sender

    ctx = {}
    with patch(
        "app.services.music.get_open_co_listening",
        new_callable=AsyncMock,
        return_value=object(),
    ):
        source = await sender._prepare_music_recommendation_source(
            ctx,
            conversation_id="conv-1",
        )

    assert source == "greeting"
    assert "music_track" not in ctx


@pytest.mark.asyncio
async def test_prepare_music_recommendation_skips_when_agent_not_idle():
    from app.services.proactive import sender

    ctx = {"schedule_status": {"status": "busy", "activity": "写报告"}}
    source = await sender._prepare_music_recommendation_source(
        ctx,
        conversation_id="conv-1",
    )

    assert source == "music_skip_not_idle"
    assert "music_track" not in ctx


@pytest.mark.asyncio
async def test_proactive_music_waits_for_user_before_marking_playback_active():
    from app.models.music import MusicTrack
    from app.services.proactive import sender

    class _Trace:
        safe_trace_id = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    def _discard_background_task(coro):
        coro.close()
        return None

    state = SimpleNamespace(
        id="state-1",
        workspace_id="ws-1",
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        stage="warming",
        followup_plan_type="normal",
        current_window_index=1,
    )
    track = MusicTrack(id="track-1", title="Quiet Realm")
    context = {
        "agent": SimpleNamespace(name="A"),
        "music_track": track,
        "schedule_status": {"status": "idle"},
    }
    start_co_listening = AsyncMock()
    emit_status = AsyncMock(return_value="status-1")

    with (
        patch(
            "app.services.runtime_config.bind_agent_context",
            new_callable=AsyncMock,
        ),
        patch.object(
            sender,
            "_check_send_eligibility",
            new_callable=AsyncMock,
            return_value=sender._SendPrep(
                conversation_id="conv-1",
                cooldown={},
                exclude_memory_ids=set(),
            ),
        ),
        patch.object(
            sender,
            "determine_proactive_stage",
            new_callable=AsyncMock,
            return_value="warming",
        ),
        patch.object(sender, "select_topic_theme", return_value="音乐"),
        patch.object(sender, "select_topic_source", return_value="greeting"),
        patch.object(sender, "_should_use_music_source", return_value=True),
        patch.object(
            sender,
            "build_proactive_context",
            new_callable=AsyncMock,
            return_value=context,
        ),
        patch.object(
            sender,
            "_prepare_music_recommendation_source",
            new_callable=AsyncMock,
            return_value="music",
        ),
        patch(
            "app.services.llm.usage_tracker.traced_usage_session",
            return_value=_Trace(),
        ),
        patch.object(
            sender,
            "_generate_message",
            new_callable=AsyncMock,
            return_value="一起听这首歌吧。",
        ),
        patch.object(
            sender,
            "emit_proactive_message",
            new_callable=AsyncMock,
            return_value="assistant-1",
        ),
        patch(
            "app.services.music.start_co_listening",
            new=start_co_listening,
        ),
        patch(
            "app.services.music_status.persist_and_emit_music_status",
            new=emit_status,
        ),
        patch.object(
            sender,
            "increment_proactive_count",
            new_callable=AsyncMock,
        ),
        patch.object(
            sender,
            "_persist_proactive_state",
            new_callable=AsyncMock,
        ),
        patch.object(
            sender.asyncio,
            "create_task",
            side_effect=_discard_background_task,
        ),
    ):
        sent = await sender.generate_and_send_proactive(
            state,
            trigger_type="silence_wakeup",
        )

    assert sent is True
    start_co_listening.assert_awaited_once()
    assert start_co_listening.await_args.kwargs["initiated_by"] == "agent"
    assert start_co_listening.await_args.kwargs["status"] == "active"
    assert start_co_listening.await_args.kwargs["is_playing"] is False
    emit_status.assert_awaited_once()
    assert emit_status.await_args.kwargs["actor"] == "agent"


@pytest.mark.asyncio
async def test_load_proactive_memories_skips_rerank_when_no_topic():
    """topic_theme 为空 → 不调 rerank, 直接走 importance 倒排."""
    from app.services.proactive.context import _load_proactive_memories

    rows = [_row("m1", "A")]
    rerank_call = AsyncMock()
    with (
        patch("app.services.proactive.context.memory_repo.find_many",
              new_callable=AsyncMock, return_value=rows),
        patch("app.services.proactive.context.render_prompt", rerank_call),
    ):
        _, ids = await _load_proactive_memories(
            user_id="u1", workspace_id="ws1", source="ai_l1",
            topic_theme="",
        )

    rerank_call.assert_not_awaited()
    assert ids == ["m1"]


@pytest.mark.asyncio
async def test_determine_proactive_stage_4_tier():
    """Spec §2.1 单维度 4 档: P1/P2/warming/intimate. 入参 float (Redis 写入侧类型)."""
    from app.services.proactive.state import determine_proactive_stage

    cases = [
        (0.0, "p1_cold"),    # 新 agent 无 Redis cache → cold start
        (5.0, "p1_cold"),
        (20.0, "p1_cold"),
        (21.0, "p2_cold"),
        (40.0, "p2_cold"),
        (41.0, "warming"),
        (80.0, "warming"),
        (81.0, "intimate"),
        (100.0, "intimate"),
    ]
    for intimacy, expected in cases:
        with patch(
            "app.services.proactive.state._load_topic_intimacy",
            new_callable=AsyncMock, return_value=intimacy,
        ):
            stage = await determine_proactive_stage("agent1", "user1")
        assert stage == expected, f"intimacy={intimacy} → expected={expected}, got={stage}"


def test_silence_prompts_carry_current_mood():
    """4 个 silence_* prompt 必须带 current_mood."""
    from app.services.proactive.sender import _format_prompt
    from app.services.prompting.defaults import (
        PROACTIVE_SILENCE_PLAIN_PROMPT,
        PROACTIVE_SILENCE_AI_MEMORY_PROMPT,
        PROACTIVE_SILENCE_USER_MEMORY_PROMPT,
        PROACTIVE_SILENCE_SCHEDULE_PROMPT,
    )

    prompt_by_key = {
        "proactive.silence_plain": PROACTIVE_SILENCE_PLAIN_PROMPT,
        "proactive.silence_ai_memory": PROACTIVE_SILENCE_AI_MEMORY_PROMPT,
        "proactive.silence_user_memory": PROACTIVE_SILENCE_USER_MEMORY_PROMPT,
        "proactive.silence_schedule": PROACTIVE_SILENCE_SCHEDULE_PROMPT,
    }

    emotion = {"emotion": "焦虑", "intensity": 80}
    for key, tpl in prompt_by_key.items():
        ctx = {
            "topic_theme": "天气",
            "proactive_memories": ["[生活/日常] 测试记忆"],
            "schedule_status": {"activity": "散步", "status": "idle"},
            "user_portrait": "测试用户",
            "recent_context": "(无)",
            "emotion": emotion,
            "__tpl": tpl,
        }
        out = _format_prompt(key, ctx, "温和")
        assert out is not None, f"{key} format returned None"
        assert "焦虑而紧绷" in out, f"{key} missing current_mood; got:\n{out}"
