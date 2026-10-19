"""Web versions → paragraph decisions → real HTTP/PG/Redis → final Trace/API.

Only provider/model transports are controlled. The fixture refuses a nonlocal
database, and a scoped synthetic user is used for every recommendation.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

from app.services.auth import create_jwt
from app.services.offline import activity_generation as generation, activity_recommendation_message as note, copy_trace, repository as repo
from app.services.offline.providers.search import SearchResult
from app.services.offline import cleversee_discovery as discovery
from app.services.offline.discovery_facts import native_place
from app.services.prompting import store
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from tests.test_cleversee_activity_e2e import native, journey, flow  # noqa: F401
from tests.test_activity_detail_partial_review import DRAFT, NOW, PARTS, RELEASE, verdict
from tests.test_offline_opening_hours import MUSEUM_HOURS

NOTE = ("想把镇江博物馆留作你下次出门的一个选择，就在伯先路。"
        "喜欢看展的话，可以先看看介绍，挑自己感兴趣的内容。"
        "不必为了这次推荐特意挤时间，也不用给自己安排太满的路线。"
        "另找一个方便的开放时段去看看，出发前确认一下当天安排就好。"
        "你可以先挑自己感兴趣的展览了解一下，不一定要一次走完。"
        "若当天还有别的安排，也可以等方便的时候再决定，按自己的节奏就好。")


async def publish(j):
    for entry in RELEASE["prompts"]:
        key = entry["key"]
        definition = PROMPT_DEFINITION_MAP[key]
        await j.db.prompttemplate.upsert(where={"key": key}, data={
            "create": dict(key=key, stage=definition.stage, category=definition.category, title=definition.title,
                           description=definition.description, content=definition.default_text, defaultContent=definition.default_text),
            "update": dict(isEnabled=True)})
        before = await j.db.prompttemplate.find_unique(where={"key": key})
        response = await j.client.put("/admin-api/prompts/" + key, json={"content": entry["content"], "expected_updated_at": before.updatedAt.isoformat()})
        assert response.status_code == 200, response.text
        after = await j.db.prompttemplate.find_unique(where={"key": key})
        assert after.content == await j.redis.get(store._redis_key(key)) == entry["content"]
        assert after.defaultContent == before.defaultContent and after.isEnabled == before.isEnabled
        versions = await j.db.prompttemplateversion.find_many(where={"promptKey": key}, order={"createdAt": "desc"})
        assert versions[0].changeType == "manual_save"


async def test_partial_detail_matches_persistence_public_api_and_final_trace(native, monkeypatch):
    j = native
    await publish(j)
    j.raw.update(name="镇江博物馆", address="伯先路85号", types="科教文化服务|博物馆", metadata={"dailyOpeningHours": MUSEUM_HOURS})
    monkeypatch.setattr(discovery, "discover", AsyncMock(return_value=[native_place(j.raw, "镇江市")]))
    monkeypatch.setattr(generation, "local_now", lambda: NOW)
    monkeypatch.setattr(note, "local_now", lambda: NOW)
    monkeypatch.setattr(generation, "_native_context_sources", AsyncMock(return_value=[SearchResult(
        title="镇江博物馆", url="https://www.zj-museum.com.cn/", content="镇江博物馆的馆舍落成于1890年，占地70余亩。6号楼设有吴文化青铜器、陶瓷器、金银器和工艺品基本陈列；8号楼有《奔流——镇江近代历史陈列》。")]))
    async def writer(_model, prompt, **_kwargs):
        if "候选原始ID" in prompt:
            return json.dumps({"candidates": [{"candidate_id": "poi:" + j.raw["id"], "summary": "喜欢看展的话，可以另选开放时段了解这里的展览。"}]})
        return json.dumps({"text": DRAFT}, ensure_ascii=False)
    async def checker(_model, prompt, **_kwargs):
        if "推荐详情分段：" in prompt:
            payload, _ = json.JSONDecoder().raw_decode(prompt.split("推荐详情分段：", 1)[1])
            ids = [p["section_id"] for p in payload["sections"]]
            return verdict(ids, {2})
        return {"supported": True, "unsupported_claims": []}
    monkeypatch.setattr(generation, "invoke_text", writer)
    monkeypatch.setattr(generation, "invoke_json", checker)
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value=NOTE))
    monkeypatch.setattr(note, "invoke_json", AsyncMock(return_value={"supported": True, "relevant_indices": [], "unsupported_claims": []}))
    steps = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: steps.append(kwargs))
    response = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert response.status_code == 200, response.text
    public = response.json()
    stored = await repo.get_activity(public["id"], j.user)
    assert stored["discovery_metadata"]["copy_status"] == "verified_partial"
    assert PARTS[0] in public["description"] and PARTS[1] in public["description"]
    assert "国庆节" not in public["description"] and "10月8日" not in public["description"]
    assert public["description"] == stored["description"]
    assert public["recommendation_message"] == stored["recommendation_message"] == NOTE
    final = next(s for s in steps if s["name"] == "offline_copy_final")
    assert final["inputs"]["activity_id"] == public["id"]
    for field in ("summary", "description", "recommendation_message"):
        assert final["outputs"]["fields"][field]["text"] == public[field]
        assert final["outputs"]["fields"][field]["sha256"] == hashlib.sha256(public[field].encode()).hexdigest()
    fetched = await j.client.get("/offline/activities/" + public["id"])
    assert fetched.status_code == 200 and fetched.json()["description"] == public["description"]
    denied = await j.client.get("/offline/activities/" + public["id"], headers={"Authorization": "Bearer " + create_jwt("foreign-user", role="user")})
    assert denied.status_code == 404
    assert "discovery_metadata" not in public and "copy_kept_sections" not in public
    output = os.environ.get("ACTIVITY_COPY_E2E_PAYLOAD")
    if output:
        Path(output).write_text(json.dumps(public, ensure_ascii=False))
