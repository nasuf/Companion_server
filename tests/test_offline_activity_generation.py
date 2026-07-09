import base64
from unittest.mock import AsyncMock

from app.api.public import offline
from app.services.offline import activity_service
from app.services.offline.activity_generation import (
    ACTIVITY_PLACE_CATEGORIES,
    _card_repeats_history,
    _card_has_concrete_place,
    _fallback_card,
    _filter_repeated_results,
    _search_query,
    _search_queries,
    _usable_results,
)
from app.services.offline.activity_images import _is_bad_image_url
from app.services.offline import activity_media_repo, repository as offline_repo
from app.services.offline.providers.search import SearchResult


def test_search_query_localizes_zhenjiang_for_chinese_sources():
    query = _search_query("Zhenjiang", ["音乐爱好者"])

    assert "江苏 镇江" in query
    assert "Zhenjiang" in query
    assert "音乐爱好者" in query
    assert "咖啡馆" in query
    assert "手作" in query
    assert "河边" in query
    assert "菜市场" in query
    assert "创意园" in query


def test_activity_place_categories_cover_small_city_options():
    category_names = {category.name for category in ACTIVITY_PLACE_CATEGORIES}

    assert {
        "阅读与文化",
        "咖啡与茶饮",
        "手作与小店",
        "公园与绿地",
        "水边散步",
        "山与轻户外",
        "街区与市集",
        "小吃与轻食",
        "城市观察",
    }.issubset(category_names)


def test_offline_activity_image_upload_limit_is_10mb():
    assert offline.activity_media_storage._MAX_IMAGE_BYTES == 10 * 1024 * 1024


def test_search_queries_push_recently_used_place_category_back():
    recent = [{"title": "镇江市图书馆常设展", "location_name": "镇江市图书馆"}]
    queries = _search_queries("Zhenjiang", ["音乐爱好者"], recent)

    assert "图书馆" not in queries[1]
    assert any("图书馆" in query for query in queries[2:])


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


def test_social_and_video_sources_are_not_usable_activity_sources():
    results = [
        SearchResult(
            title="《消费主张》江苏镇江",
            url="https://www.youtube.com/watch?v=SVi68CHJb4c",
            content="江苏镇江 半城山水半城诗",
            score=0.5,
        ),
        SearchResult(
            title="好玩江苏 镇江打卡墙",
            url="https://www.facebook.com/ExploreJiangsu/photos/example",
            content="猜猜看这些打卡墙都在镇江哪里",
            score=0.5,
        ),
    ]

    assert _usable_results(results, "Zhenjiang") == []


def test_fallback_card_does_not_promote_unverified_source_title_to_place():
    result = SearchResult(
        title="THE BEST Free Things to Do in Zhenjiang (2026) - Tripadvisor",
        url="https://www.tripadvisor.com/Attractions-g297444-Activities-zft11292-Zhenjiang_Jiangsu.html",
        content="We had dinner at the Paulaner, which had live music and outdoor",
    )

    card = _fallback_card("Zhenjiang", ["音乐爱好者"], [result])

    assert card is None


def test_activity_card_requires_a_concrete_place():
    assert (
        _card_has_concrete_place(
            {
                "title": "当前位置附近轻松散步小计划",
                "location_name": "当前位置附近",
                "address": "当前位置附近",
            },
            "Zhenjiang",
        )
        is False
    )
    assert (
        _card_has_concrete_place(
            {
                "title": "镇江博物馆常设展",
                "location_name": "镇江博物馆",
                "address": "镇江博物馆",
            },
            "Zhenjiang",
        )
        is True
    )


def test_activity_image_filters_map_and_icon_urls():
    assert _is_bad_image_url("https://map.qq.com/staticmap?markers=1") is True
    assert _is_bad_image_url("https://example.com/assets/logo.png") is True
    assert (
        _is_bad_image_url("https://example.com/zhenjiang-museum-hall-photo.jpg")
        is False
    )


def test_repeated_activity_search_results_are_filtered_by_recent_location():
    recent = [{"title": "镇江市图书馆常设展", "location_name": "镇江市图书馆"}]
    library = SearchResult(
        title="镇江市图书馆：在书页的伤痕里看见光阴",
        url="https://example.com/library",
        content="镇江市图书馆活动公告",
    )
    museum = SearchResult(
        title="镇江博物馆常设展",
        url="https://example.com/museum",
        content="镇江博物馆开放信息",
    )

    assert _filter_repeated_results([library, museum], recent) == [museum]


def test_repeated_activity_filter_ignores_old_place_in_source_snippet():
    recent = [{"title": "镇江市图书馆常设展", "location_name": "镇江市图书馆"}]
    cafe = SearchResult(
        title="镇江安静咖啡馆与茶室整理",
        url="https://example.com/cafe",
        content="这条整理也提到镇江市图书馆附近适合散步。",
    )

    assert _filter_repeated_results([cafe], recent) == [cafe]


def test_repeated_activity_search_results_return_empty_when_all_candidates_repeat():
    recent = [{"title": "镇江市图书馆常设展", "location_name": "镇江市图书馆"}]
    library = SearchResult(
        title="镇江市图书馆：在书页的伤痕里看见光阴",
        url="https://example.com/library",
        content="镇江市图书馆活动公告",
    )

    assert _filter_repeated_results([library], recent) == []


def test_activity_card_repeats_history_when_location_matches_recent_place():
    recent = [{"title": "镇江市图书馆常设展", "location_name": "镇江市图书馆"}]
    card = {
        "title": "镇江市图书馆阅读节",
        "location_name": "镇江市图书馆",
        "address": "镇江市图书馆",
    }

    assert _card_repeats_history(card, recent) is True


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
        lambda _user_id, _mime, kind="image": "user-1_activity.jpg",
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
        duration_seconds=None,
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


async def test_upload_offline_activity_audio_saves_file_and_returns_media(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(offline.activity_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        offline.activity_media_storage,
        "storage_key_for",
        lambda _user_id, _mime, kind="image": "user-1_voice.m4a",
    )
    monkeypatch.setattr(
        offline.activity_media_repo,
        "activity_belongs_to_user",
        AsyncMock(return_value=True),
    )
    created = activity_media_repo.OfflineActivityMedia(
        id="media-voice-1",
        recommendation_id="activity-1",
        user_id="user-1",
        kind="audio",
        name="voice.m4a",
        mime="audio/mp4",
        size=5,
        width=None,
        height=None,
        duration_seconds=12,
        storage_key="user-1_voice.m4a",
        url="/offline/media/user-1_voice.m4a",
        created_at=None,
    )
    create_media = AsyncMock(return_value=created)
    monkeypatch.setattr(offline.activity_media_repo, "create_media", create_media)

    response = await offline.upload_offline_activity_image(
        "activity-1",
        offline.OfflineActivityImageUpload(
            kind="audio",
            name="voice.m4a",
            mime="audio/mp4",
            size=5,
            duration_seconds=12,
            base64=base64.b64encode(b"voice").decode("ascii"),
        ),
        user={"sub": "user-1", "role": "user"},
    )

    assert (tmp_path / "user-1_voice.m4a").read_bytes() == b"voice"
    create_media.assert_awaited_once()
    assert response.kind == "audio"
    assert response.duration_seconds == 12
    assert response.url == "/offline/media/user-1_voice.m4a"


async def test_create_recommendation_requires_resolved_user_city(monkeypatch):
    generate = AsyncMock()
    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
                "user_location_latitude": 32.19,
                "user_location_longitude": 119.45,
                "user_location_city": None,
                "user_location_region": None,
            }
        ),
    )
    monkeypatch.setattr(activity_service, "generate_activity_card", generate)

    result = await activity_service.create_recommendation_for_user(
        user_id="user-1",
        workspace_id="workspace-1",
        source="manual",
    )

    assert result is None
    generate.assert_not_awaited()
