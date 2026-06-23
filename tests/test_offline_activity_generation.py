import base64
from unittest.mock import AsyncMock

from app.api.public import offline
from app.services.offline import activity_service
from app.services.offline.activity_generation import (
    _card_repeats_history,
    _fallback_card,
    _filter_repeated_results,
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


async def test_accept_activity_allows_reaccepting_ignored_activity(monkeypatch):
    activity = {
        "id": "activity-1",
        "status": "ignored",
        "title": "镇江博物馆常设展",
        "summary": "",
        "description": "",
        "workspace_id": "workspace-1",
        "created_at": "2026-06-21T10:00:00Z",
        "updated_at": "2026-06-21T10:00:00Z",
    }
    updated = {**activity, "status": "accepted"}
    feedback = AsyncMock()
    emit = AsyncMock()

    monkeypatch.setattr(
        activity_service.repo,
        "get_activity",
        AsyncMock(return_value=activity),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "update_activity_status",
        AsyncMock(return_value=updated),
    )
    monkeypatch.setattr(activity_service.repo, "create_activity_feedback", feedback)
    monkeypatch.setattr(
        activity_service.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
                "workspace_id": "workspace-1",
            }
        ),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "update_next_activity_due",
        AsyncMock(),
    )
    monkeypatch.setattr(activity_service, "emit_assistant", emit)
    monkeypatch.setattr(activity_service, "remember_user_event", lambda **_: None)

    result = await activity_service.accept_activity("user-1", "activity-1")

    assert result.status == "accepted"
    assert "重新接受" in feedback.await_args.kwargs["text"]
    assert emit.await_args.kwargs["trigger_type"] == "offline_activity_reaccepted"
