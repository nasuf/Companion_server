from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.api.public import agents as agents_api
from app.services.runtime.job_queue import RuntimeJobEnqueueUncertain, RuntimeJobOrphaned


@pytest.mark.asyncio
async def test_enqueue_agent_initialization_uses_runtime_job_queue():
    enqueue = AsyncMock(return_value="job-1")

    with patch.object(agents_api, "enqueue_runtime_job", enqueue):
        await agents_api._enqueue_agent_initialization(
            agent_id="agent-1",
            user_id="user-1",
            workspace_id="ws-1",
            personality={"warmth": 80},
        )

    enqueue.assert_awaited_once()
    args = enqueue.await_args.args
    kwargs = enqueue.await_args.kwargs
    assert args[0] == "agent_initialization"
    assert args[1]["agent_id"] == "agent-1"
    assert args[1]["workspace_id"] == "ws-1"
    assert kwargs["idempotency_key"] == "agent_initialization:agent-1"


@pytest.mark.asyncio
async def test_enqueue_agent_initialization_falls_back_to_local_background_task():
    background = []

    with (
        patch.object(agents_api, "enqueue_runtime_job", AsyncMock(side_effect=RuntimeError("redis down"))),
        patch.object(agents_api, "fire_background", side_effect=lambda coro: background.append(coro)),
    ):
        await agents_api._enqueue_agent_initialization(
            agent_id="agent-1",
            user_id="user-1",
            workspace_id="ws-1",
            personality={},
        )

    assert len(background) == 1
    background[0].close()


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeJobOrphaned('old-job'), RuntimeJobEnqueueUncertain('possible-job')])
async def test_unverifiable_initialization_is_visible_and_never_locally_replayed(error):
    progress = AsyncMock()
    with (
        patch.object(agents_api, 'enqueue_runtime_job', AsyncMock(side_effect=error)),
        patch.object(agents_api, 'set_progress', progress),
        patch.object(agents_api, 'fire_background') as background,
    ):
        await agents_api._enqueue_agent_initialization(
            agent_id='agent-1', user_id='user-1', workspace_id='ws-1', personality={},
        )
    background.assert_not_called()
    assert progress.await_args.args == ('agent-1', 'failed')
    assert progress.await_args.kwargs['expected_stage'] == 'queued'
