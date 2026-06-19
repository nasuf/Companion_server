from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.api.public import daily_share
from app.services.daily_share import photos


@pytest.mark.asyncio
async def test_list_daily_share_photos_groups_user_message_images(monkeypatch):
    rows = [
        {
            "id": "att-evening",
            "message_id": "msg-1",
            "conversation_id": "conv-1",
            "name": "sunset.jpg",
            "mime": "image/jpeg",
            "size": 100,
            "width": 800,
            "height": 600,
            "url": "/chat/media/user_sunset.jpg",
            "vision_summary": "画面里有傍晚的晚霞和柔和光线。",
            "created_at": datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
        },
        {
            "id": "att-desk",
            "message_id": "msg-2",
            "conversation_id": "conv-1",
            "name": "desk.jpg",
            "mime": "image/jpeg",
            "size": 90,
            "width": 640,
            "height": 480,
            "url": "/chat/media/user_desk.jpg",
            "vision_summary": "桌面上有咖啡杯、书页和一副眼镜。",
            "created_at": datetime(2026, 6, 18, 11, 0, tzinfo=UTC),
        },
    ]
    query = AsyncMock(return_value=rows)
    monkeypatch.setattr(photos.db, "query_raw", query)

    response = await daily_share.list_daily_share_photos(
        user={"sub": "user-id", "role": "user"},
    )

    assert response.total == 2
    assert [group.id for group in response.groups] == [
        "evening-light",
        "desk-fragments",
    ]
    assert response.groups[0].photos[0].url == "/chat/media/user_sunset.jpg"
    query.assert_awaited_once()
    assert query.await_args.args[1] == "user-id"
    assert len(query.await_args.args) == 2


def test_daily_share_photo_unknowns_fall_back_to_recent():
    photo = photos._photo_from_row(
        {
            "id": "att-1",
            "message_id": "msg-1",
            "conversation_id": "conv-1",
            "mime": "image/jpeg",
            "size": 1,
            "url": "/chat/media/user_photo.jpg",
            "vision_summary": "一张难以归类的抽象画面。",
        }
    )

    grouped = photos._group_photos([photo])

    assert grouped[0].id == "recent-photos"
    assert grouped[0].count == 1


def test_daily_share_classification_scores_vision_summary_content():
    flower_field = photos._photo_from_row(
        {
            "id": "att-road",
            "message_id": "msg-1",
            "conversation_id": "conv-1",
            "mime": "image/jpeg",
            "size": 1,
            "url": "/chat/media/user_flower_field.jpg",
            "vision_summary": "户外自然风景，有一大片粉色花海和远处的道路。",
        }
    )
    potted_plant = photos._photo_from_row(
        {
            "id": "att-object",
            "message_id": "msg-2",
            "conversation_id": "conv-1",
            "mime": "image/jpeg",
            "size": 1,
            "url": "/chat/media/user_potted_plant.jpg",
            "vision_summary": "室内桌边有一个小盆栽特写，像随手拍的小物件。",
        }
    )

    grouped = photos._group_photos([flower_field, potted_plant])

    assert [group.id for group in grouped] == ["on-the-road", "little-things"]
    assert grouped[0].photos[0].id == "att-road"
    assert grouped[1].photos[0].id == "att-object"
