"""Auth wiring for the HTTP chat endpoints.

`POST /chat/{conversation_id}` and the proactive endpoints previously treated
conversation_id / user_id as capability tokens. They now require a JWT owner.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_auth_header as _hdr


@pytest.fixture
def client(api_client):
    return api_client


def _conv(user_id="u1"):
    return SimpleNamespace(
        id="c1", userId=user_id, isDeleted=False,
        agent=SimpleNamespace(id="a1", name="小樱", status="active"),
    )


class TestChatPostAuth:
    def test_no_token_401(self, client):
        r = client.post("/chat/c1", json={"message": "hi"})
        assert r.status_code == 401

    def test_wrong_owner_403(self, client):
        with patch("app.api.public.chat.db") as db_mock:
            db_mock.conversation.find_unique = AsyncMock(return_value=_conv("owner"))
            r = client.post("/chat/c1", headers=_hdr("intruder"), json={"message": "hi"})
        assert r.status_code == 403

    def test_owner_accepted(self, client):
        """owner → 通过鉴权 (进入聚合逻辑, 用 mock 拦住副作用即可)."""
        with (
            patch("app.api.public.chat.db") as db_mock,
            patch("app.api.public.chat.get_cached_schedule", new_callable=AsyncMock, return_value=None),
            patch("app.api.public.chat.generate_daily_schedule", new_callable=AsyncMock, return_value=None),
            patch("app.api.public.chat.build_reply_timing_context", new_callable=AsyncMock, return_value={}),
            patch("app.api.public.chat.plan_user_message_aggregation", new_callable=AsyncMock) as plan_mock,
            patch("app.api.public.chat._persist_user_message", new_callable=AsyncMock, return_value="m1"),
            patch("app.api.public.chat.enqueue_or_append_delayed", new_callable=AsyncMock),
        ):
            db_mock.conversation.find_unique = AsyncMock(return_value=_conv("u1"))
            plan_mock.return_value = SimpleNamespace(
                should_wait=False, metadata=None,
                final_message="hi", final_context={"delay_seconds": 0.0},
            )
            r = client.post("/chat/c1", headers=_hdr("u1"), json={"message": "hi"})
        assert r.status_code == 200

    def test_admin_bypass(self, client):
        with (
            patch("app.api.public.chat.db") as db_mock,
            patch("app.api.public.chat.get_cached_schedule", new_callable=AsyncMock, return_value=None),
            patch("app.api.public.chat.generate_daily_schedule", new_callable=AsyncMock, return_value=None),
            patch("app.api.public.chat.build_reply_timing_context", new_callable=AsyncMock, return_value={}),
            patch("app.api.public.chat.plan_user_message_aggregation", new_callable=AsyncMock) as plan_mock,
            patch("app.api.public.chat._persist_user_message", new_callable=AsyncMock, return_value="m1"),
            patch("app.api.public.chat.enqueue_or_append_delayed", new_callable=AsyncMock),
        ):
            db_mock.conversation.find_unique = AsyncMock(return_value=_conv("someone-else"))
            plan_mock.return_value = SimpleNamespace(
                should_wait=False, metadata=None,
                final_message="hi", final_context={"delay_seconds": 0.0},
            )
            r = client.post("/chat/c1", headers=_hdr("admin", role="admin"), json={"message": "hi"})
        assert r.status_code == 200


class TestProactiveAuth:
    def test_trigger_no_token_401(self, client):
        r = client.post("/chat/proactive/a1?user_id=u1")
        assert r.status_code == 401

    def test_trigger_wrong_user_403(self, client):
        r = client.post("/chat/proactive/a1?user_id=u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_trigger_owner_ok(self, client):
        with (
            patch("app.api.public.chat.resolve_workspace_id", new_callable=AsyncMock, return_value="w1"),
            patch(
                "app.api.public.chat.send_manual_or_triggered_proactive",
                new_callable=AsyncMock, return_value={"ok": True, "message": "hi"},
            ),
        ):
            r = client.post("/chat/proactive/a1?user_id=u1", headers=_hdr("u1"))
        assert r.status_code == 200

    def test_history_wrong_user_403(self, client):
        r = client.get("/chat/proactive/a1/history?user_id=u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_history_owner_ok(self, client):
        with (
            patch("app.api.public.chat.resolve_workspace_id", new_callable=AsyncMock, return_value="w1"),
            patch("app.api.public.chat.get_proactive_history", new_callable=AsyncMock, return_value=[]),
        ):
            r = client.get("/chat/proactive/a1/history?user_id=u1", headers=_hdr("u1"))
        assert r.status_code == 200
