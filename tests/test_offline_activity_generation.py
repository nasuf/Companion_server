import base64
from unittest.mock import AsyncMock

from app.api.public import offline
from app.services.offline.activity_generation import (
    _fallback_card,
    _search_query,
    _usable_results,
)
from app.services.offline import activity_media_repo, repository as offline_repo
from app.services.offline.providers.search import SearchResult


def test_search_query_localizes_zhenjiang_for_chinese_sources():
    query = _search_query("Zhenjiang", ["音乐爱好者"])

    assert "江苏 镇江" in query
    assert "Zhenjiang" in query
    assert "音乐爱好者" in query


def test_tripadvisor_generic_review_source_is_not_usable_activity_source():
    result = SearchResult(
        title="THE BEST Free Things to Do in Zhenjiang (2026) - Tripadvisor",
        url="https://www.tripadvisor.com/Attractions-g297444-Activities-zft11292-Zhenjiang_Jiangsu.html",
        content=(
            "Highly rated activities with free entry in Zhenjiang. "
            "We had dinner at the Paulaner, which had live music and outdoor"
        ),
        score=0.5,
    )

    assert _usable_results([result], "Zhenjiang") == []


def test_fallback_card_does_not_promote_unverified_source_title_to_place():
    result = SearchResult(
        title="THE BEST Free Things to Do in Zhenjiang (2026) - Tripadvisor",
        url="https://www.tripadvisor.com/Attractions-g297444-Activities-zft11292-Zhenjiang_Jiangsu.html",
        content="We had dinner at the Paulaner, which had live music and outdoor",
    )

    card = _fallback_card("Zhenjiang", ["音乐爱好者"], [result])

    assert "Paulaner" not in card["title"]
    assert card["location_name"] == "镇江"


async def test_clear_user_activities_deletes_feedback_and_recommendations(monkeypatch):
    captured = []

    async def fake_query_raw(sql, user_id):
        captured.append((sql, user_id))
        if "offline_activity_feedback" in sql:
            return [{"id": "feedback-1"}, {"id": "feedback-2"}]
        return [{"id": "activity-1"}, {"id": "activity-2"}, {"id": "activity-3"}]

    monkeypatch.setattr(offline_repo.db, "query_raw", fake_query_raw)

    result = await offline_repo.clear_user_activities("user-1")

    assert result == {"deleted_activities": 3, "deleted_feedback": 2}
    assert [user_id for _, user_id in captured] == ["user-1", "user-1"]
    assert "DELETE FROM offline_activity_feedback" in captured[0][0]
    assert "DELETE FROM offline_activity_recommendations" in captured[1][0]


async def test_upload_offline_activity_image_saves_file_and_returns_media(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(offline.activity_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        offline.activity_media_storage,
        "storage_key_for",
        lambda _user_id, _mime: "user-1_activity.jpg",
    )
    monkeypatch.setattr(
        offline.activity_media_repo,
        "activity_belongs_to_user",
        AsyncMock(return_value=True),
    )
    created = activity_media_repo.OfflineActivityMedia(
        id="media-1",
        recommendation_id="activity-1",
        user_id="user-1",
        kind="image",
        name="done.jpg",
        mime="image/jpeg",
        size=5,
        width=800,
        height=600,
        storage_key="user-1_activity.jpg",
        url="/offline/media/user-1_activity.jpg",
        created_at=None,
    )
    create_media = AsyncMock(return_value=created)
    monkeypatch.setattr(offline.activity_media_repo, "create_media", create_media)

    response = await offline.upload_offline_activity_image(
        "activity-1",
        offline.OfflineActivityImageUpload(
            name="done.jpg",
            mime="image/jpeg",
            size=5,
            width=800,
            height=600,
            base64=base64.b64encode(b"image").decode("ascii"),
        ),
        user={"sub": "user-1", "role": "user"},
    )

    assert (tmp_path / "user-1_activity.jpg").read_bytes() == b"image"
    create_media.assert_awaited_once()
    assert response.id == "media-1"
    assert response.url == "/offline/media/user-1_activity.jpg"
