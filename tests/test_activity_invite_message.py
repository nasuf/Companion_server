"""The chat invitation sent with an activity card: grounded, reviewed, observable."""
import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.offline import activity_generation as generation
from app.services.offline import activity_message_context as context
from app.services.offline import activity_recommendation_message as note
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.trace_components import ManagedPromptText

TEXTS = {  # The r2 follow-up overrides the keys it republishes.
    entry["key"]: entry["content"]
    for name in ("20261017_activity_copy_grounding.json", "20261017_activity_copy_grounding_r2.json")
    for entry in json.loads((Path(__file__).parents[1] / "scripts/prompt_releases" / name).read_text())["prompts"]
}
NOW = datetime(2026, 10, 13, 23, 47, tzinfo=ZoneInfo("Asia/Shanghai"))
HISTORY = "[2026-10-10] 用户喜欢历史"
GOOD = "刚看到大慈恩寺，想到你之前说喜欢历史，要不要去转转？"


def activity(**metadata):
    return dict(title="大慈恩寺", location_name="大慈恩寺", address="慈恩路1号", category="寺庙",
                summary="想换条路线走走，可以把大慈恩寺作为这次出门的一站。", description="",
                discovery_metadata={"kind": "place", **metadata})


@pytest.fixture
def invite(monkeypatch):
    texts = dict(TEXTS)

    async def prompt(key):
        # Production returns managed prompt objects; keep that type in tests.
        return ManagedPromptText(texts[key], key)

    monkeypatch.setattr(generation, "get_prompt_text", prompt)
    monkeypatch.setattr(generation, "get_chat_model", lambda: object())
    monkeypatch.setattr(generation, "local_now", lambda: NOW)
    monkeypatch.setattr(note, "local_now", lambda: NOW)
    monkeypatch.setattr(generation.repo, "resolve_user_context", AsyncMock(return_value={"conversation_id": "c"}))
    monkeypatch.setattr(context, "message_context", AsyncMock(return_value={
        "agent_name": "小伴", "personality_brief": "温和",
        "dialogue_context": "[2026-10-13 07:44] AI：想着你平时对着代码，去看点绿色？",
    }))
    # The broad memory brief and rule tags are no longer invitation inputs.
    monkeypatch.setattr(generation.repo, "memory_brief", AsyncMock(side_effect=AssertionError("unused")))
    monkeypatch.setattr(generation.repo, "list_user_tags", AsyncMock(side_effect=AssertionError("unused")))
    write = AsyncMock(return_value=GOOD)
    review = AsyncMock(return_value={"supported": True, "relevant_indices": [0], "unsupported_claims": []})
    monkeypatch.setattr(generation, "invoke_text", write)
    monkeypatch.setattr(note, "invoke_json", review)
    monkeypatch.setattr(note, "get_utility_model", lambda: object())
    return write, review, texts


async def run(card):
    return await generation.generate_activity_invite_message(activity=card, user_id="u", workspace_id="w")


async def test_only_verified_dated_references_and_computed_dates_reach_the_writer(invite):
    write, review, _ = invite
    card = activity(user_relevance=[{"kind": "memory", "text": HISTORY}])
    assert await run(card) == GOOD
    prompt = write.await_args.args[1]
    assert HISTORY in prompt and "今天：2026-10-13（星期二）" in prompt
    assert "活动日期：固定地点，不对应某一天" in prompt
    assert re.search(r"\{[a-z_]+\}", prompt) is None  # Every placeholder was supplied.
    reviewed = review.await_args.args[1]
    assert "当前阶段：message" in reviewed and HISTORY in reviewed
    assert card["discovery_metadata"]["invite_message_status"] == "verified"
    assert card["discovery_metadata"]["invite_message_issues"] == []


async def test_claims_from_assistant_lines_are_repaired_then_fall_back(invite):
    write, review, _ = invite
    # Phrased so no deterministic guard decides it: the model review must.
    write.return_value = "写代码的人去大慈恩寺换换心情正合适？"
    review.return_value = {"supported": False, "relevant_indices": [],
                           "unsupported_claims": ["“你平时写代码”没有用户原文支持"]}
    card = activity()
    text = await run(card)
    assert text == "「大慈恩寺」要不要了解一下？"
    assert write.await_count == 2 and "没有用户原文支持" in write.await_args.args[1]
    meta = card["discovery_metadata"]
    assert meta["invite_message_status"] == "fallback_invalid"
    assert meta["invite_message_issues"] == ["“你平时写代码”没有用户原文支持"]


async def test_going_along_is_repaired_without_spending_a_review(invite):
    write, review, _ = invite
    write.side_effect = ["大慈恩寺挺有意思，到时候我们现场见！", "大慈恩寺挺有意思，有空去转转？"]
    review.return_value = {"supported": True, "relevant_indices": [], "unsupported_claims": []}
    card = activity()
    assert await run(card) == "大慈恩寺挺有意思，有空去转转？"
    assert review.await_count == 1
    assert card["discovery_metadata"]["invite_message_status"] == "verified_no_evidence"


async def test_no_evidence_invitation_citing_the_user_is_repaired(invite):
    """Production 10-17: "之前看你关注过这场" passed review with no user quote at all."""
    write, review, _ = invite
    write.side_effect = ["之前看你关注过大慈恩寺，那边挺清静，有空去转转？", "大慈恩寺挺有意思，有空去转转？"]
    review.return_value = {"supported": True, "relevant_indices": [], "unsupported_claims": []}
    card = activity()
    assert await run(card) == "大慈恩寺挺有意思，有空去转转？"
    feedback = write.await_args.args[1]
    assert "没有用户原文可依据" in feedback and "环境“清静”" in feedback
    assert review.await_count == 1
    assert card["discovery_metadata"]["invite_message_status"] == "verified_no_evidence"


async def test_unpublished_review_template_is_marked_unchecked(invite):
    write, review, texts = invite
    texts["offline.recommendation_message_check"] = PROMPT_DEFINITION_MAP["offline.recommendation_message_check"].default_text
    card = activity(user_relevance=[{"kind": "memory", "text": HISTORY}])
    assert await run(card) == GOOD
    review.assert_not_awaited()
    assert card["discovery_metadata"]["invite_message_status"] == "unchecked"


async def test_provider_failure_keeps_the_fallback_and_says_so(invite):
    write, _, _ = invite
    write.side_effect = TimeoutError()
    card = activity()
    card["discovery_metadata"] = None  # Callers may pass rows without metadata.
    assert await run(card) == "「大慈恩寺」要不要了解一下？"
    assert card["discovery_metadata"]["invite_message_status"] == "fallback_unavailable"


async def test_empty_drafts_and_unexpected_errors_keep_the_fallback(invite, monkeypatch):
    write, _, _ = invite
    write.return_value = "“”"
    card = activity()
    assert await run(card) == "「大慈恩寺」要不要了解一下？"
    assert card["discovery_metadata"]["invite_message_status"] == "fallback_invalid"
    monkeypatch.setattr(generation.repo, "resolve_user_context", AsyncMock(side_effect=RuntimeError("db")))
    card = activity()
    assert await run(card) == "「大慈恩寺」要不要了解一下？"
    assert card["discovery_metadata"]["invite_message_status"] == "fallback_unavailable"


@pytest.mark.parametrize("merge_fails", [False, True])
async def test_post_create_path_keeps_reasons_out_of_client_metadata(monkeypatch, merge_fails):
    from app.services.offline import activity_service as service
    card = dict(title="大慈恩寺", summary="可以去大慈恩寺走走。", description="", location_name="大慈恩寺",
                image_urls=[], discovery_metadata={"kind": "place"})
    created = {**card, "id": "activity-1", "status": "pending", "conversation_id": "conversation-1",
               "discovery_metadata": {"kind": "place"}}

    async def invite(*, activity, user_id, workspace_id):
        activity["discovery_metadata"].update(invite_message_status="fallback_invalid",
                                              invite_message_issues=["“你平时写代码”没有用户原文支持"])
        return "「大慈恩寺」要不要了解一下？"

    monkeypatch.setattr(service.repo, "resolve_user_context", AsyncMock(return_value={
        "conversation_id": "conversation-1", "agent_id": "agent-1", "workspace_id": "workspace-1",
        "user_location_latitude": 32.19, "user_location_longitude": 119.45,
        "user_location_city": "Zhenjiang", "user_location_region": "Jiangsu"}))
    monkeypatch.setattr(service, "generate_activity_card", AsyncMock(return_value=card))
    monkeypatch.setattr(service.repo, "create_activity", AsyncMock(return_value=created))
    monkeypatch.setattr(service, "generate_activity_invite_message", invite)
    merge = AsyncMock(side_effect=RuntimeError("db") if merge_fails else None)
    monkeypatch.setattr(service.repo, "merge_discovery_metadata", merge)
    emit = AsyncMock()
    monkeypatch.setattr(service, "emit_assistant", emit)
    monkeypatch.setattr(service, "emit_activity_card", AsyncMock())
    monkeypatch.setattr(service.repo, "update_next_activity_due", AsyncMock())
    await service.create_recommendation_for_user(user_id="user-1", workspace_id="workspace-1", source="manual")
    merge.assert_awaited_once_with("activity-1", "user-1", {
        "invite_message_status": "fallback_invalid",
        "invite_message_issues": ["“你平时写代码”没有用户原文支持"]})
    sent = emit.await_args.kwargs["extra_metadata"]  # Still delivered when recording fails.
    assert sent["invite_message_status"] == "fallback_invalid" and "invite_message_issues" not in sent


async def test_merge_adds_keys_without_replacing_the_document(monkeypatch):
    from app.services.offline import repository
    run = AsyncMock()
    monkeypatch.setattr(repository.db, "execute_raw", run)
    await repository.merge_discovery_metadata("a", "u", {})
    run.assert_not_awaited()
    await repository.merge_discovery_metadata("a", "u", {"invite_message_status": "verified"})
    sql, *params = run.await_args.args
    assert "discovery_metadata || $3::jsonb" in sql and "updated_at" not in sql
    assert params == ["a", "u", '{"invite_message_status": "verified"}']
