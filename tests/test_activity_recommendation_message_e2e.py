"""Web-published prompts → scoped evidence → final POI → HTTP → persisted note."""
import json
from unittest.mock import AsyncMock
from uuid import uuid4

from app.services.auth import create_jwt
from app.services.offline import activity_generation as generation
from app.services.offline import activity_recommendation_message as note
from app.services.offline import repository as repo, cleversee_discovery as discovery
from app.services.offline.discovery_facts import native_place
from tests.test_activity_recommendation_message import MEMORY, DIALOGUE, MESSAGE
from tests.test_cleversee_activity_e2e import native, journey, flow  # noqa: F401


async def seed(j):
    for table, text, workspace in (
        ("memories_user", MEMORY, j.workspace),
        ("memories_user", "我明天要处理报销。", j.workspace),
        ("memories_user", "另一会话的私密记忆", None),
        ("memories_ai", "AI经常去这家咖啡店。", j.workspace),
    ):
        await j.db.execute_raw(f"""INSERT INTO {table}
            (id,user_id,workspace_id,content,importance,level,main_category,sub_category,updated_at)
            VALUES ($1,$2,$3,$4,0.8,2,'偏好边界','喜好',CURRENT_TIMESTAMP)""",
            uuid4().hex, j.user, workspace, text)
    for role, text in (("user", DIALOGUE), ("assistant", "你喜欢这里的甜品。")):
        await j.db.message.create(data={"conversationId": j.conversation, "role": role, "content": text})
    await j.db.execute_raw("""INSERT INTO messages (id,conversation_id,role,content,metadata)
        VALUES ($1,$2,'user','用户确认到达咖啡馆', '{"component_card":{"type":"offline_activity"}}'::jsonb)""",
        uuid4().hex, j.conversation)


async def test_selected_relevance_web_versions_scoping_and_public_api_consistency(native, monkeypatch):
    j = native
    await seed(j)
    captured = []
    original = generation.invoke_text

    async def select(model, prompt):
        if "候选原始ID" not in prompt:
            return await original(model, prompt)
        captured.append(str(prompt))
        assert MEMORY in prompt and DIALOGUE in prompt
        for excluded in ("另一会话的私密记忆", "AI经常去这家", "你喜欢这里的甜品", "用户确认到达咖啡馆"):
            assert excluded not in prompt
        return json.dumps({"candidates": [{"candidate_id": "poi:" + j.raw["id"],
            "user_relevance": [MEMORY, "咖啡", DIALOGUE, "喜欢找咖啡馆看书", "编造的回忆"]}]}, ensure_ascii=False)

    monkeypatch.setattr(generation, "invoke_text", select)
    writing = AsyncMock(return_value=MESSAGE)
    monkeypatch.setattr(note, "invoke_text", writing)
    monkeypatch.setattr(note, "invoke_json", AsyncMock(return_value={"supported": True, "unsupported_claims": [], "relevant_indices": [0, 1, 2]}))
    result = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert result.status_code == 200, result.text
    public = result.json()
    assert public["recommendation_message"] == MESSAGE and len(public["image_urls"]) == 3
    assert len(captured) == 1 and writing.await_count == 1
    assert "我明天要处理报销" not in writing.call_args.args[1]
    stored = await repo.get_activity(public["id"], j.user)
    meta = stored["discovery_metadata"]
    assert [r["text"] for r in meta["user_relevance"]] == [MEMORY, "咖啡", DIALOGUE]
    assert meta["recommendation_message_status"] == "verified"
    assert meta["recommendation_evidence_status"] == {"memory": "collected", "preference": "collected", "dialogue": "collected"}
    path = "/offline/activities/" + public["id"]
    for response in (
        await j.client.get(path),
        await j.client.post(path + "/ignore"),
        await j.client.post(path + "/accept"),
        await j.client.post(path + "/arrive", json={"lat": 32.21, "lng": 119.43, "accuracy_m": 10}),
        await j.client.post(path + "/archive"),
        await j.client.get(path),
    ):
        assert response.status_code == 200, response.text
        assert response.json()["recommendation_message"] == MESSAGE
        assert "user_relevance" not in response.text and "discovery_metadata" not in response.text
        assert "报销" not in response.text and "编造的回忆" not in response.text
    listing = await j.client.get("/offline/activities", params={"workspace_id": j.workspace})
    assert listing.status_code == 200 and MESSAGE in listing.text
    assert "user_relevance" not in listing.text
    # New API field never grants access to another user's recommendation.
    foreign = await j.client.get(path, headers={"Authorization": "Bearer " + create_jwt(uuid4().hex, role="user")})
    assert foreign.status_code == 404
    # These are real Web publication numbers, not code-sync audit counts.
    for key in ("offline.activity_card", "offline.activity_recommendation_message", "offline.recommendation_message_check"):
        rows = await j.db.query_raw("""SELECT v.change_type, p.number FROM prompt_template_versions v
            JOIN prompt_publication_versions p ON p.version_id=v.id
            WHERE v.prompt_key=$1 ORDER BY p.number DESC LIMIT 1""", key)
        assert rows[0]["change_type"] == "manual_save" and rows[0]["number"] >= 1


async def test_same_category_image_substitution_discards_original_personalization(native, monkeypatch):
    j = native
    await seed(j)
    first = native_place({**j.raw, "images": []}, "镇江市")
    second = native_place({**j.raw, "id": "second-" + j.raw["id"], "name": "青石咖啡馆", "address": "伯先路18号"}, "镇江市")
    monkeypatch.setattr(discovery, "discover", AsyncMock(return_value=[first, second]))
    original = generation.invoke_text

    async def select(model, prompt):
        if "候选原始ID" in prompt:
            return json.dumps({"candidates": [{"candidate_id": first["candidate_id"], "user_relevance": [MEMORY]}]})
        return await original(model, prompt)

    monkeypatch.setattr(generation, "invoke_text", select)
    result = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert result.status_code == 200, result.text
    public = result.json()
    assert public["location_name"] == second["location_name"]
    prompt = note.invoke_text.call_args.args[1]
    assert "用户记忆库：[]" in prompt and MEMORY not in prompt
    stored = await repo.get_activity(public["id"], j.user)
    assert stored["discovery_metadata"]["user_relevance"] == []
    assert stored["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"


async def test_foreign_conversation_is_never_evidence(native):
    j = native
    await seed(j)
    from app.services.offline.activity_message_context import recommendation_dialogue
    assert await recommendation_dialogue(user_id="someone-else", workspace_id=j.workspace, conversation_id=j.conversation) == []
    assert await recommendation_dialogue(user_id=j.user, workspace_id=None, conversation_id=j.conversation) == []
    assert await recommendation_dialogue(user_id=j.user, workspace_id=j.workspace, conversation_id=j.conversation) == [DIALOGUE]


async def test_legacy_source_keeps_selected_place_relevance(native, monkeypatch):
    from app.config import settings
    from app.services.offline.providers.search import SearchResult
    from app.services.offline import activity_service as service
    j = native
    await seed(j)
    monkeypatch.setattr(settings, "offline_search_provider", "tavily")
    source = SearchResult(title=j.raw["name"], url="https://source.fixture.test/cafe",
        content="镇江市润州区伯先路12号，小岛咖啡(伯先路店)是一家咖啡馆。")
    monkeypatch.setattr(generation, "_search_activity_candidates", AsyncMock(return_value=([source], [source], "镇江市 咖啡")))
    monkeypatch.setattr(generation, "persist_activity_images", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "geocode_address", AsyncMock(return_value=None))
    original = generation.invoke_text

    async def select(model, prompt):
        if "候选原始ID" in prompt:
            return json.dumps({"candidates": [{"title": j.raw["name"], "location_name": j.raw["name"],
                "city": "镇江市", "official_url": source.url, "user_relevance": [MEMORY]}]})
        return await original(model, prompt)

    monkeypatch.setattr(generation, "invoke_text", select)
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value=MESSAGE))
    monkeypatch.setattr(note, "invoke_json", AsyncMock(return_value={"supported": True, "relevant_indices": [0], "unsupported_claims": []}))
    result = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert result.status_code == 200, result.text
    stored = await repo.get_activity(result.json()["id"], j.user)
    assert stored["location_name"] == j.raw["name"]
    assert stored["recommendation_message"] == MESSAGE
    assert stored["discovery_metadata"]["user_relevance"] == [{"kind": "memory", "text": MEMORY}]
