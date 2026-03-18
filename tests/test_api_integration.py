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
        patch("app.api.health.db") as mock_db,
        patch("app.api.health.redis_health", new_callable=AsyncMock, return_value=True),
        patch("app.api.health.neo4j_health", new_callable=AsyncMock, return_value=True),
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
    with patch("app.api.users.db") as mock_db:
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
        createdAt="2025-01-01T00:00:00",
    )
    with patch("app.api.agents.db") as mock_db:
        mock_db.aiagent = MagicMock()
        mock_db.aiagent.create = AsyncMock(return_value=mock_agent)
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
    with patch("app.api.memories.memory_repo.find_many", new_callable=AsyncMock, return_value=[]):
        response = client.get("/memories", params={"user_id": "test-user"})
        assert response.status_code == 200
        assert response.json() == []


def test_get_emotion(mock_deps):
    """GET /emotions/{agent_id}/current returns emotion state."""
    client = mock_deps
    mock_agent = SimpleNamespace(id="agent-id")
    with (
        patch("app.api.emotions.db") as mock_db,
        patch("app.api.emotions.get_ai_emotion", new_callable=AsyncMock,
              return_value={"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}),
    ):
        mock_db.aiagent = MagicMock()
        mock_db.aiagent.find_unique = AsyncMock(return_value=mock_agent)
        response = client.get("/emotions/agent-id/current")
        assert response.status_code == 200
        data = response.json()
        assert data["pleasure"] == 0.5
        assert "tone" in data
