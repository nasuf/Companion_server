"""Regression for user location -> schedule -> self-memory feedback loops."""
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from app.services import persona_grounding as g
from app.services.prompting import defaults
from app.services.schedule_domain import schedule as s


def agent(city="云南省普洱市思茅区南屏镇凤凰路社区"):
    return SimpleNamespace(id="agent", userId="owner", name="小伴", city=city,
                           age=22, occupation="客服员")


@pytest.fixture
def checker(monkeypatch):
    async def prompt(key):
        return {"persona.grounding_context": defaults.PERSONA_GROUNDING_CONTEXT_PROMPT,
                "persona.grounding_check": defaults.PERSONA_GROUNDING_CHECK_PROMPT}[key]
    monkeypatch.setattr(g, "get_prompt_text", prompt)
    monkeypatch.setattr(g, "get_grounding_model", lambda: object())
    call = AsyncMock(return_value={"verdicts": [{"index": 0, "allowed": False}]})
    monkeypatch.setattr(g, "invoke_json", call)
    return call


@pytest.mark.parametrize("text", [
    "我本来就住镇江", "我现在住在镇江凤凰路", "我家在凤凰路那边",
    "我一直居住在成都", "我家就在镇江", "我今年30岁", "我叫小雨", "I live in London",
])
def test_stable_self_facts_cannot_bypass_with_life_category(text):
    assert g.generated_stable_self_claim(text)


@pytest.mark.parametrize("text", [
    "用户向我分享了当前位置：镇江市润州区天桥支路",
    "用户告诉我住在镇江", "我记得用户住在镇江", "我记得用户说自己住在镇江", "用户的朋友住在成都",
    "今天处理了一个困难的客服投诉", "我和用户第一次认真聊了家庭问题",
    "我住的小公寓真的很省心", "我叫了外卖然后继续处理工作",
])
def test_real_interactions_and_user_subjects_are_not_new_persona(text):
    assert not g.generated_stable_self_claim(text)


@pytest.mark.parametrize("text", [
    "我刚走到润州街头，晚风扫过还挺舒服的。", "刚走到润州街头，晚风挺舒服",
    "清晨在镇江醒来", "我来这边玩呀", "我在上海外滩", "我们在上海外滩", "在上海外滩拍照", "I am in Paris", "我家在凤凰路那边", "I live in London",
])
def test_location_self_statements_enter_verification(text):
    assert g.needs_location_verification(text)


async def test_plain_reply_has_no_extra_model_call(checker):
    assert await g.guard_reply(agent(), "听起来今天挺开心呀") == ("听起来今天挺开心呀", False)
    checker.assert_not_awaited()


@pytest.mark.parametrize("question", ["你在忙吗？", "你在干嘛？", "你在吃饭吗？", "你现在在想什么？"])
async def test_activity_questions_do_not_add_geography_model_call(checker, question):
    text = "今天处理工作有点累，想先休息一会儿"
    assert await g.guard_reply(agent(), text, question=question) == (text, False)
    checker.assert_not_awaited()


@pytest.mark.parametrize("question", ["你在镇江吗？", "你不是在普洱吗？", "你在忙吗？你在镇江吗？"])
async def test_location_confirmation_still_checks_short_answers(checker, question):
    _, corrected = await g.guard_reply(agent(), "对呀", question=question)
    assert corrected
    checker.assert_awaited_once()


async def test_wrong_implicit_proactive_place_is_caught(checker):
    text, corrected = await g.guard_reply(agent(), "刚走到润州街头，晚风挺舒服")
    assert corrected and "普洱" in text and "来玩" not in text
    assert checker.await_count == 1


async def test_denial_of_correct_city_is_checked_with_question(checker):
    text, corrected = await g.guard_reply(agent(), "我没说过呀，是不是你记混了", question="你不是在普洱吗？")
    assert corrected and "普洱" in text
    assert "你不是在普洱吗" in checker.await_args.args[1]


async def test_subject_preserving_user_location_share_is_kept(checker):
    text = "用户向我分享了当前位置：镇江市润州区天桥支路"
    assert await g.verify_generated_locations(agent(), [text], kind="memory") == []
    checker.assert_not_awaited()


@pytest.mark.parametrize("verdict", [
    {}, {"verdicts": []}, {"verdicts": [{"index": True, "allowed": True}]},
    {"verdicts": [{"index": 0, "allowed": "true"}]},
    {"verdicts": [{"index": 5, "allowed": True}]},
])
async def test_malformed_verdict_cannot_silently_pass(checker, verdict):
    checker.return_value = verdict
    with pytest.raises(g.GroundingUnavailable):
        await g.verify_generated_locations(agent(), ["我住镇江"], kind="memory")


async def test_checker_failure_cannot_send_unverified_claim(checker):
    checker.side_effect = TimeoutError()
    text, corrected = await g.guard_reply(agent(), "我住镇江")
    assert corrected and "普洱" in text


async def test_accepted_other_city_discussion_is_preserved(checker):
    checker.return_value = {"verdicts": [{"index": 0, "allowed": True}]}
    text = "我听说镇江的锅盖面不错，你想去尝尝吗？"
    assert await g.guard_reply(agent(), text) == (text, False)


async def test_schedule_filters_user_scope_archive_and_identity(monkeypatch):
    from app.services.memory.storage import repo
    query = AsyncMock(return_value=[SimpleNamespace(content="用户喜欢阅读")])
    monkeypatch.setattr(repo, "find_many", query)
    assert await s._get_user_memory_summary("owner", workspace_id="exact-workspace") == "用户喜欢阅读"
    assert query.await_args.kwargs["where"] == {
        "userId": "owner", "workspaceId": "exact-workspace", "level": 1,
        "isArchived": False, "mainCategory": "偏好",
    }
    query.reset_mock()
    assert await s._get_user_memory_summary("owner") == ""
    query.assert_not_awaited()


async def test_old_poisoned_schedule_is_neutralized_before_current_state(checker):
    slots = [{"start": "18:30", "end": "20:00", "event": "漫步润州街头", "status": "空闲"}]
    clean = await s._ground_schedule(agent(), slots)
    assert s.get_current_status(clean, datetime(2026, 10, 9, 19, 7))["event"] == "日常活动"
    assert slots[0]["event"] == "漫步润州街头"  # source evidence retained
    assert clean[0]["start"] == "18:30"


async def test_unknown_schedule_verdict_preserves_timing_and_safe_activity(checker):
    checker.side_effect = TimeoutError()
    slots = [{"start": "09:00", "end": "10:00", "event": "处理邮件", "status": "忙碌"},
             {"start": "18:00", "end": "19:00", "activity": "去镇江润州街头", "type": "leisure"}]
    clean = await s._ground_schedule(agent(), slots)
    assert clean[0] == slots[0]
    assert clean[1]["activity"] == "日常活动"


async def test_cached_schedule_validation_is_bound_to_profile_and_content(checker, monkeypatch):
    data = [{"start": "18:00", "end": "19:00", "event": "润州街头", "status": "空闲"}]
    cache = {"schedule:agent:20261009": json.dumps(data)}
    redis = SimpleNamespace(get=AsyncMock(side_effect=lambda key: cache.get(key)),
                            set=AsyncMock(side_effect=lambda key, value, **kw: cache.update({key: value})))
    monkeypatch.setattr(s, "get_redis", AsyncMock(return_value=redis))
    profile = agent()
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=profile))))
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert (await s.get_cached_schedule("agent", day))[0]["event"] == "日常活动"
    await s.get_cached_schedule("agent", day)
    assert checker.await_count == 1
    profile.city = "江苏省镇江市润州区"
    checker.return_value = {"verdicts": [{"index": 0, "allowed": True}]}
    assert (await s.get_cached_schedule("agent", day))[0]["event"] == "润州街头"
    assert checker.await_count == 2


async def test_storage_gate_runs_before_embeddings_and_classification(monkeypatch):
    from app.services.memory.storage import persistence as p
    embedding = AsyncMock()
    monkeypatch.setattr(p, "generate_embedding", embedding)
    for main, sub in [("生活", "居住"), ("生活", "其他"), ("情绪", "高兴")]:
        assert await p.store_memory("owner", "我本来就住镇江", main_category=main,
                                    sub_category=sub, source="ai", provenance="ai_authored") is None
    embedding.assert_not_awaited()


async def test_workspace_agent_rejects_other_owner_and_archived_scope(monkeypatch):
    import app.db as db_module
    find = AsyncMock(return_value=None)
    monkeypatch.setattr(db_module, "db", SimpleNamespace(chatworkspace=SimpleNamespace(find_first=find)))
    with pytest.raises(g.GroundingUnavailable):
        await g.workspace_agent("owner", "archived")
    find.return_value = SimpleNamespace(agentId="agent", agent=SimpleNamespace(id="agent", userId="someone-else"))
    with pytest.raises(g.GroundingUnavailable):
        await g.workspace_agent("owner", "wrong")
    assert find.await_args.kwargs["where"] == {"id": "wrong", "userId": "owner", "status": "active"}


async def test_mixed_subject_cannot_bypass_location_gate(checker):
    text = "用户向我分享了当前位置：镇江，我也在润州街头"
    assert g.needs_location_verification(text, implicit_self=True)
    assert await g.verify_generated_locations(agent(), [text], kind="memory") == [0]
    assert g.generated_stable_self_claim("用户告诉我住在镇江，我也住镇江")


async def test_unknown_canonical_city_keeps_safe_text_and_blocks_risky_claim(checker):
    assert await g.guard_reply(agent(None), "今天挺开心") == ("今天挺开心", False)
    text, corrected = await g.guard_reply(agent(None), "我现在住镇江")
    assert corrected and "镇江" not in text
    checker.assert_not_awaited()


async def test_reconciliation_cannot_reintroduce_wrong_location(monkeypatch):
    from app.services.memory.storage import persistence as p
    from app.services.memory.storage.reconciliation import ReconciliationDecision
    existing = SimpleNamespace(id="old", level=3, importance=.4, content="旧生活总结")
    monkeypatch.setattr(p, "generate_embedding", AsyncMock(return_value=[0.0]))
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=ReconciliationDecision(
        action="merge_existing", existing_id="old", existing_record=existing,
        merged_content="我本来就住镇江，今天聊得很开心",
    )))
    update, vector = AsyncMock(), AsyncMock()
    monkeypatch.setattr(p.memory_repo, "update", update)
    monkeypatch.setattr(p, "store_embedding", vector)
    assert await p.store_memory("owner", "今天聊得很开心", level=3, importance=.4,
        main_category="生活", sub_category="其他", source="ai", provenance="ai_authored", workspace_id="scope") is None
    update.assert_not_awaited()
    vector.assert_not_awaited()


def test_grounding_trace_explains_interception():
    from app.services.chat.trace_enrich import _label_persona_grounding
    assert _label_persona_grounding('{"verdicts":[{"index":0,"allowed":false}]}') == "拦截 1 项地点冲突"
    assert _label_persona_grounding('{"verdicts":[{"index":0,"allowed":true}]}') == "角色地点一致"
    assert _label_persona_grounding('{"verdicts":[{"allowed":"false"}]}') is None


async def test_short_circuit_guard_precedes_persistence_and_voice(checker, monkeypatch):
    from app.services.chat import multi_intent as m
    from app.services.speech_output import delivery, policy
    save, voice = AsyncMock(return_value="message"), AsyncMock(return_value=None)
    monkeypatch.setattr(policy, "should_generate_voice", AsyncMock(return_value=True))
    monkeypatch.setattr(delivery, "prepare_voice_output", voice)
    def background(coro):
        coro.close()
    monkeypatch.setattr(m, "_fire_background", background)
    events = await m.short_circuit_reply("刚走到润州街头", "conv", "agent", "owner", save,
        agent=agent(), voice_context=object(), defer_turn_finalization=True)
    assert "普洱" in voice.await_args.kwargs["text"]
    assert "润州" not in voice.await_args.kwargs["text"]
    assert "普洱" in save.await_args.args[1][0]["text"]
    assert "普洱" in json.loads(events[0]["data"])["text"]


@pytest.mark.parametrize("retry_failure", [False, True])
async def test_proactive_anchor_and_final_guard_cover_retry(checker, monkeypatch, retry_failure):
    from app.services.proactive import sender, recent_messages
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    async def prompt(key):
        return PROMPT_DEFINITION_MAP[key].default_text
    monkeypatch.setattr(sender, "get_prompt_text", prompt)
    monkeypatch.setattr(sender, "build_personality_brief", lambda _: "小伴")
    monkeypatch.setattr(sender, "get_chat_model", lambda: object())
    calls = AsyncMock(side_effect=["刚走到润州街头，晚风很舒服", TimeoutError()] if retry_failure else None,
                      return_value="刚走到润州街头，晚风很舒服")
    monkeypatch.setattr(sender, "invoke_text", calls)
    monkeypatch.setattr(recent_messages, "is_repeat_of_recent", AsyncMock(return_value=retry_failure))
    ctx = {"agent": agent(), "trigger_type": "silence", "is_decay_final": True, "workspace_id": "scope"}
    assert await sender._generate_message(ctx) is None
    assert ctx["_skip_reason_detail"] == "persona_location_conflict_or_unverified"
    assert "普洱" in calls.await_args_list[0].args[1]
    assert checker.await_count == 1


async def test_real_recording_failure_keeps_watermark_for_retry(checker, monkeypatch):
    from app.services.memory.recording import pipeline as p
    from app.services.chat import post_process as post
    from datetime import UTC
    monkeypatch.setattr(p, "resolve_workspace_id", AsyncMock(return_value="scope"))
    monkeypatch.setattr(p, "should_extract_memory", lambda _: True)
    monkeypatch.setattr(p, "should_memorize", AsyncMock(return_value=True))
    monkeypatch.setattr(p, "extract_memories", AsyncMock(return_value={"memories": [{
        "content": "清晨在镇江醒来", "importance": .4, "main_category": "生活", "sub_category": "其他",
    }]}))
    monkeypatch.setattr(g, "workspace_agent", AsyncMock(return_value=agent()))
    checker.side_effect = TimeoutError()
    monkeypatch.setattr(post, "get_watermark", AsyncMock(return_value=None))
    watermark = AsyncMock()
    monkeypatch.setattr(post, "set_watermark", watermark)
    monkeypatch.setattr(post, "process_memory_pipeline", p.process_memory_pipeline)
    await post._do_memory_pipeline("owner", [{"role": "assistant", "content": "清晨在镇江醒来", "createdAt": datetime.now(UTC).isoformat()}], "conv", "scope")
    watermark.assert_not_awaited()
    assert checker.await_count == 1
    # Confirmed invalid content is intentionally rejected, then consumption can
    # advance; an unavailable verifier must not silently consume the batch.
    checker.side_effect = None
    await post._do_memory_pipeline("owner", [{"role": "assistant", "content": "清晨在镇江醒来", "createdAt": datetime.now(UTC).isoformat()}], "conv", "scope")
    watermark.assert_awaited_once()


@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_real_chat_executor_delivers_and_records_same_correction(checker, monkeypatch, executor):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.agent.city = agent().city
    monkeypatch.setattr(settings, "chat_executor", executor)
    monkeypatch.setattr(chat, "_memory_weak_reply", AsyncMock(return_value="我本来就住镇江"))
    events = [evt async for evt in chat.stream_chat_response("c-1", "今天怎么样", io.agent, "u-1",
              reply_context={"received_at": io.now.isoformat()})]
    replies = [json.loads(evt["data"])["text"] for evt in events if evt["event"] == "reply"]
    assert replies and all("普洱" in text and "镇江" not in text for text in replies)
    assert "普洱" in chat._background_post_process.call_args.kwargs["full_response"]
    assert chat._save_replies.await_count == chat.finish_assistant_turn.await_count == 1
    assert io.db.message.create.await_count == 1
    assert sum(evt["event"] == "done" for evt in events) == 1


async def test_status_db_fallback_cannot_expose_old_scene(checker, monkeypatch):
    from app.api.public import conversations as api
    monkeypatch.setattr(api, "get_cached_schedule", AsyncMock(side_effect=ConnectionError()))
    slots = [{"start": "00:00", "end": "23:59", "event": "漫步润州街头", "status": "空闲"}]
    monkeypatch.setattr(api, "db", SimpleNamespace(
        aidailyschedule=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(scheduleData=slots))),
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent())),
    ))
    result = await api._current_ai_status("agent")
    assert result["ai_activity"] == "日常活动"
    assert result["ai_status"] == "idle"


@pytest.mark.parametrize("text", [
    "我在这儿陪着你。可以告诉我现在是什么让你这么难吗？",
    "我在目前能看到的记忆里，没有看到你提过喜欢的电影。",
    "我这里没有看到你跟我说过喜欢的电影，不能乱猜。",
    "我在工作室打磨齿轮", "我在吃晚饭", "我在，先到安全的地方",
])
async def test_virtual_support_and_generic_activity_are_not_physical_claims(checker, text):
    assert await g.guard_reply(agent(), text) == (text, False)
    checker.assert_not_awaited()


async def test_virtual_support_cannot_hide_real_wrong_city(checker):
    _, corrected = await g.guard_reply(agent(), "我在这儿陪着你，我现在住在镇江")
    assert corrected
    checker.assert_awaited_once()


async def test_crisis_conflict_fallback_keeps_care_instead_of_location_apology(checker, monkeypatch):
    from app.services.chat.intent_handlers import ShortCircuitCtx, _CRISIS_STATIC_FALLBACK
    from app.services.chat import multi_intent as m
    save = AsyncMock()
    monkeypatch.setattr(m, "_fire_background", lambda coro: coro.close())
    ctx = ShortCircuitCtx(conversation_id="conv", agent_id="agent", user_id="owner", agent=agent(),
        reply_context=None, tracer=MagicMock(safe_trace_id=None), save_replies_fn=save,
        pending_sub_fragments={}, sub_intent_mode=False, reply_index_offset=0, cached_patience=100,
        defer_turn_finalization=True)
    events = [e async for e in ctx.finalize("我在镇江陪着你，你现在安全吗？", kind="crisis")]
    assert json.loads(events[0]["data"])["text"] == _CRISIS_STATIC_FALLBACK
    assert ctx.last_short_circuit_reply == _CRISIS_STATIC_FALLBACK
    assert checker.await_count == 1
