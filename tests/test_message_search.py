"""Tests for the chat history search feature (Flutter「查找」).

Covers:
- app.services.chat.message_search core matching logic (text/card/image, the
  scope="all" preview + has_more shape, rank computation) with the DB layer
  mocked.
- Route wiring / ownership gating on GET /conversations/{id}/messages/search,
  same pattern as tests/test_public_endpoints_ownership.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat import message_search
from tests.conftest import make_auth_header as _hdr  # noqa: F401 — shared helper


def _message(
    id_: str,
    *,
    content: str = "",
    metadata: dict | None = None,
    conversation_id: str = "c1",
    role: str = "user",
):
    return SimpleNamespace(
        id=id_,
        conversationId=conversation_id,
        role=role,
        content=content,
        metadata=metadata,
        createdAt="2026-08-01T00:00:00+00:00",
    )


class TestSearchText:
    @pytest.mark.asyncio
    async def test_matches_content_case_insensitive(self):
        rows = [_message("m1", content="今天天气真好")]
        with patch.object(message_search, "db") as db_mock:
            db_mock.message.find_many = AsyncMock(return_value=rows)
            db_mock.query_raw = AsyncMock(return_value=[{"id": "m1", "rank": 3}])
            result = await message_search.search_messages(
                conversation_id="c1", q="天气", scope="text", limit=30, offset=0
            )
        assert len(result.text) == 1
        assert result.text[0].match_type == "text"
        assert result.text[0].rank == 3
        assert result.has_more_text is False
        where = db_mock.message.find_many.call_args.kwargs["where"]
        assert where["content"] == {"contains": "天气", "mode": "insensitive"}

    @pytest.mark.asyncio
    async def test_has_more_when_extra_row_returned(self):
        rows = [_message(f"m{i}") for i in range(31)]  # limit+1
        with patch.object(message_search, "db") as db_mock:
            db_mock.message.find_many = AsyncMock(return_value=rows)
            db_mock.query_raw = AsyncMock(return_value=[])
            result = await message_search.search_messages(
                conversation_id="c1", q=None, scope="text", limit=30, offset=0
            )
        assert len(result.text) == 30
        assert result.has_more_text is True


class TestSearchCards:
    @pytest.mark.asyncio
    async def test_filters_by_card_fields_in_python(self):
        rows = [
            {
                "id": "m1",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "time_capsule",
                        "title": "给未来的自己",
                        "subtitle": "",
                        "body": "",
                        "footer": "",
                    }
                },
                "created_at": "2026-08-01T00:00:00+00:00",
            },
            {
                "id": "m2",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "music_track",
                        "title": "晴天",
                        "subtitle": "周杰伦",
                        "body": "",
                        "footer": "",
                    }
                },
                "created_at": "2026-08-01T00:00:01+00:00",
            },
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(
                side_effect=[rows, [{"id": "m1", "rank": 0}, {"id": "m2", "rank": 1}]]
            )
            result = await message_search.search_messages(
                conversation_id="c1", q="未来", scope="card", limit=30, offset=0
            )
        assert len(result.cards) == 1
        assert result.cards[0].id == "m1"
        assert result.cards[0].match_type == "card"

    @pytest.mark.asyncio
    async def test_empty_query_browses_all_cards(self):
        rows = [
            {
                "id": "m1",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "checkin_habit",
                        "title": "喝水打卡",
                        "subtitle": "",
                        "body": "",
                        "footer": "",
                    }
                },
                "created_at": "2026-08-01T00:00:00+00:00",
            }
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(side_effect=[rows, [{"id": "m1", "rank": 0}]])
            result = await message_search.search_messages(
                conversation_id="c1", q=None, scope="card", limit=30, offset=0
            )
        assert len(result.cards) == 1

    @pytest.mark.asyncio
    async def test_matches_red_packet_blessing_even_though_card_text_is_boilerplate(
        self,
    ):
        # Every red packet card has the exact same title/body ("红包" /
        # "给你的一点心意") — only the joined user_offerings.blessing lets us
        # tell two red packets apart by what the sender actually wrote.
        rows = [
            {
                "id": "m1",
                "conversation_id": "c1",
                "role": "user",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "red_packet",
                        "title": "红包",
                        "subtitle": "",
                        "body": "给你的一点心意",
                        "footer": "点击查看",
                    }
                },
                "created_at": "2026-08-01T00:00:00+00:00",
                "offering_blessing": "生日快乐呀",
                "offering_ticket_amount": 50,
            },
            {
                "id": "m2",
                "conversation_id": "c1",
                "role": "user",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "red_packet",
                        "title": "红包",
                        "subtitle": "",
                        "body": "给你的一点心意",
                        "footer": "点击查看",
                    }
                },
                "created_at": "2026-08-01T00:00:01+00:00",
                "offering_blessing": "新年快乐",
                "offering_ticket_amount": 88,
            },
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(
                side_effect=[rows, [{"id": "m2", "rank": 0}]]
            )
            result = await message_search.search_messages(
                conversation_id="c1", q="新年", scope="card", limit=30, offset=0
            )
        assert len(result.cards) == 1
        assert result.cards[0].id == "m2"

    @pytest.mark.asyncio
    async def test_matches_red_packet_by_ticket_amount(self):
        rows = [
            {
                "id": "m1",
                "conversation_id": "c1",
                "role": "user",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "red_packet",
                        "title": "红包",
                        "subtitle": "",
                        "body": "给你的一点心意",
                        "footer": "点击查看",
                    }
                },
                "created_at": "2026-08-01T00:00:00+00:00",
                "offering_blessing": None,
                "offering_ticket_amount": 88,
            }
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(side_effect=[rows, [{"id": "m1", "rank": 0}]])
            result = await message_search.search_messages(
                conversation_id="c1", q="88", scope="card", limit=30, offset=0
            )
        assert len(result.cards) == 1

    @pytest.mark.asyncio
    async def test_card_scan_joins_user_offerings_by_message_id(self):
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(return_value=[])
            await message_search.search_messages(
                conversation_id="c1", q=None, scope="card", limit=30, offset=0
            )
        sql = db_mock.query_raw.call_args_list[0].args[0]
        assert "LEFT JOIN user_offerings" in sql
        assert "uo.message_id = m.id" in sql


class TestSearchImages:
    @pytest.mark.asyncio
    async def test_matches_vision_summary(self):
        # A single JOIN query returns message fields + the matched attachment
        # id directly — no separate message lookup step.
        joined_row = {
            "id": "m1",
            "conversation_id": "c1",
            "role": "user",
            "content": "",
            "metadata": None,
            "created_at": "2026-08-01T00:00:00+00:00",
            "matched_attachment_id": "att1",
        }
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(
                side_effect=[[joined_row], [{"id": "m1", "rank": 5}]]
            )
            result = await message_search.search_messages(
                conversation_id="c1", q="小猫", scope="image", limit=30, offset=0
            )
        assert len(result.images) == 1
        assert result.images[0].matched_attachment_id == "att1"
        assert result.images[0].match_type == "image"
        assert result.images[0].rank == 5

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty_without_rank_query(self):
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(return_value=[])
            result = await message_search.search_messages(
                conversation_id="c1", q="xyz", scope="image", limit=30, offset=0
            )
        assert result.images == []
        # image scan only — no matches means no ids to rank, so _ranks_for
        # short-circuits without a second query_raw call.
        assert db_mock.query_raw.call_count == 1

    @pytest.mark.asyncio
    async def test_query_wildcards_are_escaped_before_ilike(self):
        # A literal '%'/'_' in the query must not act as a SQL wildcard —
        # otherwise "50%" would match any vision_summary containing "50".
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(return_value=[])
            await message_search.search_messages(
                conversation_id="c1", q="50%_off", scope="image", limit=30, offset=0
            )
        first_call_args = db_mock.query_raw.call_args_list[0].args
        assert first_call_args[2] == "50\\%\\_off"


class TestEscapeLikePattern:
    def test_escapes_percent_underscore_and_backslash(self):
        assert message_search._escape_like_pattern("50%") == "50\\%"
        assert message_search._escape_like_pattern("a_b") == "a\\_b"
        assert message_search._escape_like_pattern("a\\b") == "a\\\\b"


class TestSearchAllPreview:
    @pytest.mark.asyncio
    async def test_previews_each_kind_independently(self):
        text_rows = [_message("t1", content="hello")]
        card_rows = [
            {
                "id": "c1m",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "time_capsule",
                        "title": "x",
                        "subtitle": "",
                        "body": "",
                        "footer": "",
                    }
                },
                "created_at": "2026-08-01T00:00:00+00:00",
            }
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.message.find_many = AsyncMock(return_value=text_rows)
            db_mock.query_raw = AsyncMock(
                side_effect=[
                    card_rows,  # card scan
                    [],  # image JOIN scan
                    [{"id": "t1", "rank": 0}, {"id": "c1m", "rank": 1}],  # rank batch
                ]
            )
            result = await message_search.search_messages(
                conversation_id="c1", q=None, scope="all", limit=30, offset=0
            )
        assert len(result.text) == 1
        assert len(result.cards) == 1
        assert result.images == []
        assert result.has_more_text is False
        assert result.has_more_cards is False


class TestSearchRouteOwnership:
    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_no_token_401(self, client):
        r = client.get("/conversations/c1/messages/search?q=hi")
        assert r.status_code == 401

    def test_wrong_owner_403(self, client):
        conv = SimpleNamespace(id="c1", userId="other-user", isDeleted=False)
        with patch("app.api.ownership.db") as db_mock:
            db_mock.conversation.find_unique = AsyncMock(return_value=conv)
            r = client.get("/conversations/c1/messages/search?q=hi", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_owner_200_returns_service_payload(self, client):
        from app.models.message import MessageSearchResponse

        conv = SimpleNamespace(id="c1", userId="u1", isDeleted=False)
        with (
            patch("app.api.ownership.db") as db_mock,
            patch(
                "app.api.public.conversations.message_search.search_messages",
                new_callable=AsyncMock,
            ) as search_mock,
        ):
            db_mock.conversation.find_unique = AsyncMock(return_value=conv)
            search_mock.return_value = MessageSearchResponse()
            r = client.get(
                "/conversations/c1/messages/search?q=hi&scope=text",
                headers=_hdr("u1"),
            )
        assert r.status_code == 200
        assert r.json() == {
            "text": [],
            "cards": [],
            "images": [],
            "has_more_text": False,
            "has_more_cards": False,
            "has_more_images": False,
        }
        search_mock.assert_called_once_with(
            conversation_id="c1", q="hi", scope="text", limit=30, offset=0
        )
