"""Web publications → real PG/Redis/HTTP → approved summary/note/Trace/card."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

from app.services.offline import activity_generation as generation, activity_recommendation_message as note, copy_trace, repository as repo
from app.services.offline import cleversee_discovery as discovery
from app.services.offline.discovery_facts import native_place
from app.services.offline.providers.cleversee import WebSearchResults
from app.services.offline.providers.search import SearchResult
from tests.test_cleversee_activity_e2e import native, journey, flow  # noqa: F401
from tests.test_activity_copy_final_e2e import publish
from tests.test_activity_copy_fidelity import RELEASE, PROMPTS, TEA, EVIDENCE, relevance
from tests.test_activity_detail_partial_review import verdict

SUMMARY = "爱探店的话，可以了解常发广场这家花艺小店的作品展示和花艺沙龙。"
SOURCE = ("镇江常发广场里有一家春树•花艺，地址是黄山西路常发广场5号楼102。店铺介绍提到作品展示、花艺沙龙和蜡烛制作，具体项目可在出发前向店内了解。" * 2
          + "鲜花盲盒9.9元。店内有鹦鹉和小狗。店里随便拍都好看。")
DETAIL = ("🌿 常发广场的一家花艺小店\n" + SUMMARY + "\n\n"
          "✨ 介绍里可以了解什么\n店铺介绍提到作品展示、花艺沙龙和蜡烛制作，可以先了解具体内容，选自己感兴趣的部分。\n\n"
          "📍 位置与安排\n地点在黄山西路常发广场5号楼102，出发前确认当天开放和项目安排，按自己的节奏决定。")


async def test_sparse_flower_store_is_enriched_without_flower_tea_analogy_and_all_final_fields_match(native, monkeypatch):
    j = native
    # Use the actual admin save endpoint and latest six versions, preserving defaults.
    import tests.test_activity_copy_final_e2e as previous
    monkeypatch.setattr(previous, "RELEASE", RELEASE)
    await publish(j)
    # Restore the real publisher stubbed by the shared journey fixture: persist
    # and emit the actual component card, with only external WS transport mocked.
    from app.services.offline import chat_emit, activity_service
    monkeypatch.setattr(activity_service, "emit_activity_card", chat_emit.emit_activity_card)
    monkeypatch.setattr(chat_emit, "_notify_takeover_transcript", AsyncMock())
    j.raw.update(name="春树·花艺(常发广场店)", address="黄山西路常发广场5号楼102", types="购物服务|花鸟鱼虫市场")
    monkeypatch.setattr(discovery, "discover", AsyncMock(return_value=[native_place(j.raw, "镇江市")]))
    monkeypatch.setattr(generation, "evidence_inputs", AsyncMock(return_value=(EVIDENCE, {k: "collected" for k in EVIDENCE})))
    from app.services.offline import place_context
    monkeypatch.setattr(place_context.cleversee, "web_search", AsyncMock(return_value=WebSearchResults([
        SearchResult(title="镇江常发广场花店探店", url="https://example.com/spring", content=SOURCE, raw_content=SOURCE)], source_status="available")))
    captured = []
    detail_calls = 0
    async def writer(_model, prompt, **_kwargs):
        nonlocal detail_calls
        captured.append(prompt)
        if "候选原始ID" in prompt:
            repair = '"copy_revision_feedback":' in prompt
            return json.dumps({"candidates": [{"candidate_id": "poi:" + j.raw["id"],
                "summary": SUMMARY if repair else "你喜欢带花香的茶，和店里的花香适配。",
                "vibe": "花艺小店", "suitable": "了解作品与项目", "user_relevance": [TEA, "爱探店"]}]}, ensure_ascii=False)
        assert SUMMARY in prompt and "想换条路线走走，可以把" not in prompt
        assert "花艺沙龙" in prompt
        detail_calls += 1
        if detail_calls == 1:
            assert all(ad not in prompt for ad in ("9.9", "店内有鹦鹉", "随便拍都好看"))
            # The factual reviewer below approves everything. The code guard
            # must still repair this hallucinated old offer, keeping other parts.
            return json.dumps({"text": DETAIL.replace("选自己感兴趣的部分。", "选自己感兴趣的部分。鲜花盲盒9.9元。")}, ensure_ascii=False)
        assert "旧探店特价" in prompt
        return json.dumps({"text": DETAIL}, ensure_ascii=False)
    async def checker(_model, prompt, **_kwargs):
        if "当前阶段：relevance" in prompt:
            return relevance()
        if "推荐详情分段：" in prompt:
            obj, _ = json.JSONDecoder().raw_decode(prompt.split("推荐详情分段：", 1)[1])
            return verdict([r["section_id"] for r in obj["sections"]])
        return {"supported": True, "unsupported_claims": []}
    monkeypatch.setattr(generation, "invoke_text", writer)
    monkeypatch.setattr(generation, "invoke_json", checker)
    monkeypatch.setattr(note, "invoke_json", checker)
    accepted_note = note.generic_message({"title": j.raw["name"]})
    note_writer = AsyncMock(side_effect=["只" * 81, accepted_note])
    monkeypatch.setattr(note, "invoke_text", note_writer)
    steps = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: steps.append(kwargs))
    response = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert response.status_code == 200, response.text
    public = response.json();stored = await repo.get_activity(public["id"], j.user);metadata = stored["discovery_metadata"]
    assert public["summary"] == SUMMARY and public["description"] == DETAIL
    assert detail_calls == 2
    assert public["recommendation_message"] == accepted_note and 160 <= len(accepted_note) <= 260
    assert metadata["user_relevance"] == [{"kind": "preference", "text": "爱探店"}]
    assert metadata["context_source_status"] == "available" and metadata["context_source_count"] == 1
    assert metadata["context_filtered_sentence_count"] == 3
    assert metadata["detail_input_summary_status"] == "verified"
    assert TEA not in note_writer.await_args.args[1] and "160—260" in note_writer.await_args.args[1]
    assert metadata["recommendation_message_status"] == "verified"
    final = next(s for s in steps if s["name"] == "offline_copy_final")
    for field in ("summary", "description", "recommendation_message"):
        assert final["outputs"]["fields"][field]["text"] == public[field]
        assert final["outputs"]["fields"][field]["sha256"] == hashlib.sha256(public[field].encode()).hexdigest()
    assert final["outputs"]["writing_inputs"]["recommendation_message_length"] == len(accepted_note)
    assert final["outputs"]["writing_inputs"]["context_filtered_sentence_count"] == 3
    messages = await j.db.query_raw("SELECT content,metadata FROM messages WHERE conversation_id=$1 ORDER BY created_at DESC LIMIT 8", j.conversation)
    cards = [r["metadata"]["component_card"] for r in messages if (r.get("metadata") or {}).get("component_card")]
    assert next(c for c in cards if c["payload"].get("activity_id") == public["id"])["body"] == SUMMARY
    fetched = await j.client.get("/offline/activities/" + public["id"])
    assert fetched.status_code == 200 and fetched.json()["description"] == DETAIL
    output = os.getenv("ACTIVITY_FIDELITY_E2E_PAYLOAD")
    if output:
        Path(output).write_text(json.dumps(public, ensure_ascii=False))


async def test_fact_checked_short_note_is_persisted_and_returned_without_template_replacement(native, monkeypatch):
    j = native
    import tests.test_activity_copy_final_e2e as previous
    monkeypatch.setattr(previous, "RELEASE", RELEASE)
    await publish(j)
    draft = ("这家小岛咖啡可以给下次出门多留一个选择。"
             "如果想换条路线，可以先了解店里的具体内容，再决定要不要去看看。"
             "不用为了推荐打乱原有安排，时间合适时再考虑就好。"
             "出发前确认当天的开放安排，到了现场按自己的节奏决定停留多久。"
             "若已有别的安排，把这个去处留到之后再考虑也可以。")
    assert 120 <= len(draft) < 160
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value=draft))
    monkeypatch.setattr(note, "invoke_json", AsyncMock(return_value={"supported": True, "unsupported_claims": []}))
    steps = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: steps.append(kwargs))
    response = await j.client.post("/offline/activities/recommend", params={"workspace_id": j.workspace})
    assert response.status_code == 200, response.text
    public = response.json();stored = await repo.get_activity(public["id"], j.user)
    assert public["recommendation_message"] == stored["recommendation_message"] == draft
    meta = stored["discovery_metadata"]
    assert meta["recommendation_message_status"] == "verified_partial"
    assert meta["recommendation_message_length_status"] == "below_target_after_repair"
    assert "已保留事实核验通过" in meta["recommendation_message_issues"][0]
    final = next(s["outputs"] for s in steps if s["name"] == "offline_copy_final")
    assert final["fields"]["recommendation_message"]["text"] == draft
    assert final["statuses"]["note"] == "verified_partial"
    assert final["writing_inputs"]["recommendation_message_length_status"] == "below_target_after_repair"
    assert "below_target" not in response.text and "discovery_metadata" not in response.text
