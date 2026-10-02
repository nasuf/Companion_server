"""Shared status publication follows the session mutation, before slow replies."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.models.music import MusicCoListeningResponse, MusicTrack
from app.services import music, music_status


@pytest.mark.asyncio
async def test_joint_exit_is_published_even_when_reply_generation_fails(monkeypatch):
    ended = MusicCoListeningResponse(
        status="ended", initiated_by="user_joined",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
    )
    monkeypatch.setattr(music, "end_co_listening", AsyncMock(return_value=ended))
    publish = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", publish)

    async def fail_reply(*args, **kwargs):
        publish.assert_awaited_once()
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(music_status, "_render_exit_reply", fail_reply)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await music_status.end_co_listening_with_notice(
            user_id="user-1", agent_id="agent-1", conversation_id="conv-1",
            reason="ai_busy", prompt_key="music.busy_exit",
        )
    assert publish.await_args.kwargs["shared_session"] == ended


@pytest.mark.asyncio
async def test_final_waiting_exit_is_published_before_reply(monkeypatch):
    ended = MusicCoListeningResponse(
        status="ended", initiated_by="user_joined",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
    )
    monkeypatch.setattr(music, "end_agent_waiting_if_stale", AsyncMock(return_value=ended))
    monkeypatch.setattr(music_status, "_resolve_agent_name", AsyncMock(return_value="小伴"))
    publish = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", publish)

    async def fail_reply(*args, **kwargs):
        publish.assert_awaited_once()
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(music_status, "_render_exit_reply", fail_reply)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await music_status.end_agent_waiting_after_timeout(
            user_id="user-1", agent_id="agent-1", conversation_id="conv-1", seconds=0,
        )
    assert publish.await_args.kwargs["shared_session"] == ended


@pytest.mark.asyncio
async def test_agent_acceptance_is_published_before_reply(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent", initiated_by="user_pending", is_playing=True,
        track=MusicTrack(id="track-1", title="Quiet Realm"),
    )
    joined = pending.model_copy(update={"status": "active", "initiated_by": "user_joined"})
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "start_co_listening", AsyncMock(return_value=joined))
    publish = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", publish)

    async def fail_reply(*args, **kwargs):
        publish.assert_awaited_once()
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(music_status, "_emit_rendered_reply", fail_reply)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await music_status.reconcile_co_listening_for_status(
            user_id="user-1", agent_id="agent-1", conversation_id="conv-1",
            workspace_id=None, status_code="idle", activity="自由时间", ai_name="小伴",
        )
    assert publish.await_args.kwargs["shared_session"] == joined


@pytest.mark.asyncio
async def test_concurrent_end_requests_only_return_one_transition(monkeypatch):
    monkeypatch.setattr(music, "ensure_conversation_owner", AsyncMock())
    # Emulate UPDATE ... WHERE status IN (...) RETURNING *: the second caller
    # sees an ended row and receives no transition to publish.
    active = True

    async def update(query, *args):
        nonlocal active
        assert "RETURNING *" in query
        assert "status IN ('active', 'pending_agent', 'agent_waiting_user')" in query
        if not active:
            return []
        active = False
        return [{
            "status": "ended", "initiated_by": "user_joined",
            "track_external_id": "track-1", "title": "Quiet Realm",
            "is_playing": False, "ended_reason": args[3],
        }]

    monkeypatch.setattr(music.db, "query_raw", AsyncMock(side_effect=update))
    results = await asyncio.gather(*[
        music.end_co_listening(
            user_id="user-1", agent_id="agent-1", conversation_id="conv-1",
        ) for _ in range(2)
    ])
    assert sum(result is not None for result in results) == 1
