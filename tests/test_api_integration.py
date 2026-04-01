"""Integration tests for API endpoints.

These tests verify the API routes work correctly with mocked dependencies.
Run with: pytest tests/test_api_integration.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from fastapi.testclient import TestClient


@pytest.fixture
def mock_deps():
    """Mock all external dependencies."""
    with (
        patch("app.db.connect_db", new_callable=AsyncMock),
        patch("app.db.disconnect_db", new_callable=AsyncMock),
        patch("app.redis_client.get_redis", new_callable=AsyncMock),
        patch("app.redis_client.close_redis", new_callable=AsyncMock),
        patch("app.neo4j_client.get_driver", new_callable=AsyncMock),
        patch("app.neo4j_client.close_neo4j", new_callable=AsyncMock),
        patch("app.services.graph.schema.init_graph_schema", new_callable=AsyncMock),
        patch("jobs.scheduler.setup_scheduler"),
        patch("jobs.scheduler.shutdown_scheduler"),
        patch("app.middleware.configure_logging"),
        patch("app.middleware.configure_langsmith"),
    ):
        from app.main import app
        yield TestClient(app)


def test_health_endpoint(mock_deps):
    """Health endpoint returns 200."""
    client = mock_deps
    with (
        patch("app.api.public.health.db") as mock_db,
        patch("app.api.public.health.redis_health", new_callable=AsyncMock, return_value=True),
        patch("app.api.public.health.neo4j_health", new_callable=AsyncMock, return_value=True),
    ):
        mock_db.execute_raw = AsyncMock(return_value=None)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


def test_create_user(mock_deps):
    """POST /users creates a user."""
    client = mock_deps
    mock_user = SimpleNamespace(
        id="test-id",
        name="Test User",
        email="test@example.com",
        createdAt="2025-01-01T00:00:00",
        updatedAt="2025-01-01T00:00:00",
    )
    with patch("app.api.public.users.db") as mock_db:
        mock_db.user = MagicMock()
        mock_db.user.create = AsyncMock(return_value=mock_user)
        response = client.post("/users", json={"name": "Test User"})
        assert response.status_code == 200
        assert response.json()["name"] == "Test User"


def test_create_agent(mock_deps):
    """POST /agents creates an agent."""
    client = mock_deps
    mock_agent = SimpleNamespace(
        id="agent-id",
        name="TestBot",
        userId="user-id",
        personality={"openness": 0.8},
        background=None,
        values=None,
        gender=None,
        lifeOverview=None,
        createdAt="2025-01-01T00:00:00",
    )
    mock_workspace = SimpleNamespace(id="workspace-id")
    with (
        patch("app.api.public.agents.db") as mock_db,
        patch("app.api.public.agents.create_provisioning_workspace", new_callable=AsyncMock, return_value=mock_workspace),
        patch("app.api.public.agents.stage_active_workspaces_for_user", new_callable=AsyncMock, return_value=[]),
        patch("app.api.public.agents.activate_workspace", new_callable=AsyncMock, return_value=mock_workspace),
        patch("app.api.public.agents.finalize_archived_workspaces", new_callable=AsyncMock),
        patch("app.api.public.agents.generate_initial_self_memories", new_callable=AsyncMock),
        patch("app.api.public.agents.save_ai_emotion", new_callable=AsyncMock),
        patch("app.api.public.agents.generate_and_save_life_overview", new_callable=AsyncMock, return_value={"description": "overview"}),
        patch("app.api.public.agents.generate_daily_schedule", new_callable=AsyncMock),
    ):
        mock_db.aiagent = MagicMock()
        mock_db.aiagent.create = AsyncMock(return_value=mock_agent)
        mock_db.aiagent.update = AsyncMock(return_value=mock_agent)
        response = client.post("/agents", json={
            "name": "TestBot",
            "user_id": "user-id",
            "personality": {"openness": 0.8},
        })
        assert response.status_code == 200
        assert response.json()["name"] == "TestBot"


def test_list_memories(mock_deps):
    """GET /memories returns memory list."""
    client = mock_deps
    with patch("app.api.public.memories.memory_repo.find_many", new_callable=AsyncMock, return_value=[]):
        response = client.get("/memories", params={"user_id": "test-user"})
        assert response.status_code == 200
        assert response.json() == []


def test_memory_stats(mock_deps):
    """GET /memories/stats returns bucketed counts."""
    client = mock_deps
    memories = [
        SimpleNamespace(level=1, mainCategory="身份", subCategory="姓名"),
        SimpleNamespace(level=2, mainCategory="生活", subCategory="工作"),
        SimpleNamespace(level=2, mainCategory="生活", subCategory="工作"),
    ]
    with patch("app.api.public.memories.memory_repo.find_many", new_callable=AsyncMock, return_value=memories):
        response = client.get("/memories/stats", params={"user_id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["by_level"][0]["key"] == "L2"
        assert data["by_main_category"][0]["key"] == "生活"


def test_search_memories_forwards_workspace_and_taxonomy_filters(mock_deps):
    """POST /memories/search forwards workspace/category filters to retrieval."""
    client = mock_deps
    with patch("app.api.public.memories.retrieve_memories", new_callable=AsyncMock, return_value=[]) as mock_retrieve:
        response = client.post(
            "/memories/search?user_id=test-user",
            json={
                "query": "她最近工作怎么样",
                "top_k": 8,
                "workspace_id": "ws-1",
                "main_category": "生活",
                "sub_category": "工作",
            },
        )
        assert response.status_code == 200
        mock_retrieve.assert_awaited_once_with(
            "她最近工作怎么样",
            user_id="test-user",
            semantic_k=8,
            workspace_id="ws-1",
            main_category="生活",
            sub_category="工作",
        )


def test_admin_memory_overview(mock_deps):
    """GET /admin-api/users/memory-overview returns overview payload."""
    client = mock_deps
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt

    mock_workspace = SimpleNamespace(id="ws1", status="active")
    mock_user_memory = SimpleNamespace(
        level=1,
        mainCategory="身份",
        subCategory="姓名",
        workspaceId="ws1",
    )
    mock_ai_memory = SimpleNamespace(
        level=2,
        mainCategory="生活",
        subCategory="工作",
        workspaceId="ws1",
    )
    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        with patch("app.api.admin.users.db") as mock_db:
            mock_db.chatworkspace.find_many = AsyncMock(return_value=[mock_workspace])
            mock_db.usermemory.find_many = AsyncMock(
                side_effect=[[mock_user_memory], [mock_user_memory]]
            )
            mock_db.aimemory.find_many = AsyncMock(
                side_effect=[[mock_ai_memory], [mock_ai_memory]]
            )
            response = client.get("/admin-api/users/memory-overview")
            assert response.status_code == 200
            data = response.json()
            assert data["totals"]["workspaces"] == 1
            assert data["totals"]["memories"] == 2
            assert data["totals"]["recent_memories_7d"] == 2
            assert data["by_workspace_status"][0]["key"] == "active"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_admin_agent_proactive_detail(mock_deps):
    """GET /admin-api/users/{user_id}/agents/{agent_id}/proactive returns state/events/logs."""
    client = mock_deps
    from app.main import app
    from app.api.jwt_auth import require_admin_jwt

    mock_agent = SimpleNamespace(id="agent-1", userId="user-1")
    mock_workspace = SimpleNamespace(id="ws-1")
    mock_log = SimpleNamespace(
        message="你今天过得怎么样？",
        eventType="memory_proactive",
        createdAt="2026-04-01T08:00:00+00:00",
    )
    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        with patch("app.api.admin.users.db") as mock_db:
            mock_db.aiagent.find_unique = AsyncMock(return_value=mock_agent)
            mock_db.chatworkspace.find_first = AsyncMock(return_value=mock_workspace)
            mock_db.query_raw = AsyncMock(
                side_effect=[
                    [{
                        "status": "waiting_user",
                        "stage": "warming",
                        "silence_level_n": 2,
                        "followup_plan_type": "normal",
                        "remaining_forced_triggers": None,
                        "current_window_index": None,
                        "window_due_at": None,
                        "response_deadline_at": "2026-04-02T08:00:00+00:00",
                        "t0_at": "2026-04-01T08:00:00+00:00",
                        "last_proactive_at": "2026-04-01T08:00:00+00:00",
                        "last_user_reply_at": None,
                        "last_assistant_reply_at": "2026-04-01T07:30:00+00:00",
                        "stop_reason": None,
                        "metadata": {"reason": "conversation_end"},
                    }],
                    [{
                        "event_type": "message_sent",
                        "window_name": "2h-4h",
                        "trigger_type": "memory_proactive",
                        "payload": {"message": "你今天过得怎么样？"},
                        "created_at": "2026-04-01T08:00:00+00:00",
                    }],
                ]
            )
            mock_db.proactivechatlog.find_many = AsyncMock(return_value=[mock_log])
            response = client.get("/admin-api/users/user-1/agents/agent-1/proactive")
            assert response.status_code == 200
            data = response.json()
            assert data["workspace_id"] == "ws-1"
            assert data["state"]["status"] == "waiting_user"
            assert data["events"][0]["event_type"] == "message_sent"
            assert data["logs"][0]["message"] == "你今天过得怎么样？"
            mock_db.proactivechatlog.find_many.assert_awaited_once_with(
                where={"workspaceId": "ws-1"},
                order={"createdAt": "desc"},
                take=20,
            )
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_get_emotion(mock_deps):
    """GET /emotions/{agent_id}/current returns emotion state."""
    client = mock_deps
    mock_agent = SimpleNamespace(id="agent-id")
    with (
        patch("app.api.public.emotions.db") as mock_db,
        patch("app.api.public.emotions.get_ai_emotion", new_callable=AsyncMock,
              return_value={"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}),
    ):
        mock_db.aiagent = MagicMock()
        mock_db.aiagent.find_unique = AsyncMock(return_value=mock_agent)
        response = client.get("/emotions/agent-id/current")
        assert response.status_code == 200
        data = response.json()
        assert data["pleasure"] == 0.5
        assert "tone" in data
