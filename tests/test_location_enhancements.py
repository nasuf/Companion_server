from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services import location_memory
from app.services.chat import message_search


def _location_card():
    return {
        "type": "location",
        "title": "上海市",
        "subtitle": "上海市黄浦区中山东一路",
        "body": "黄浦区",
        "footer": "刚刚",
        "accent": "#22C66B",
        "payload": {
            "latitude": 31.2304,
            "longitude": 121.4737,
            "address": "上海市黄浦区中山东一路",
            "city": "上海市",
            "region": "黄浦区",
            "country": "中国",
            "source": "device",
        },
    }


def test_build_location_memory_texts():
    from datetime import datetime, timezone
    texts = location_memory.build_location_memory_texts(_location_card(),
        accepted_at=datetime(2026, 10, 15, tzinfo=timezone.utc))
    for text in texts:
        assert "2026-10-15T00:00:00+00:00" in text
        assert "用户分享了位置点" in text
        assert "不代表常住地或Agent所在地" in text
        assert "31.23040" not in text
        assert "用户当前在" not in text


class TestLocationCardSearch:
    @pytest.mark.asyncio
    async def test_card_category_location(self):
        rows = [
            {
                "id": "m1",
                "conversation_id": "c1",
                "role": "user",
                "content": "用户分享了当前位置：上海市",
                "metadata": {"component_card": _location_card()},
                "created_at": "2026-08-01T00:00:00+00:00",
            },
            {
                "id": "m2",
                "conversation_id": "c1",
                "role": "user",
                "content": "",
                "metadata": {
                    "component_card": {
                        "type": "gift",
                        "title": "美式咖啡",
                        "subtitle": "",
                        "body": "",
                        "footer": "",
                    }
                },
                "created_at": "2026-08-01T00:00:01+00:00",
            },
        ]
        with patch.object(message_search, "db") as db_mock:
            db_mock.query_raw = AsyncMock(side_effect=[rows, [{"id": "m1", "rank": 0}]])
            result = await message_search.search_messages(
                conversation_id="c1",
                q=None,
                scope="card",
                limit=30,
                offset=0,
                card_category="location",
            )
        assert len(result.cards) == 1
        assert result.cards[0].id == "m1"
        assert result.cards[0].metadata["component_card"]["type"] == "location"
