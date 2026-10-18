"""Regression for user location -> schedule -> self-memory feedback loops."""
import asyncio
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
    assert s.get_current_status(clean, datetime(2026, 10, 9, 19, 7))["event"] == "自由时间"
    assert slots[0]["event"] == "漫步润州街头"  # source evidence retained
    assert clean[0]["start"] == "18:30"


async def test_unknown_schedule_verdict_preserves_timing_and_safe_activity(checker):
    checker.side_effect = TimeoutError()
    slots = [{"start": "09:00", "end": "10:00", "event": "处理邮件", "status": "忙碌"},
             {"start": "18:00", "end": "19:00", "activity": "去镇江润州街头", "type": "leisure"}]
    clean = await s._ground_schedule(agent(), slots)
    assert clean[0] == slots[0]
    assert clean[1]["activity"] == "自由时间"


async def test_cached_schedule_validation_is_bound_to_profile_and_content(checker, monkeypatch):
    data = [{"start": "18:00", "end": "19:00", "event": "润州街头", "status": "空闲"}]
    cache = {"schedule:agent:20261009": json.dumps(data)}
    redis = SimpleNamespace(get=AsyncMock(side_effect=lambda key: cache.get(key)),
                            set=AsyncMock(side_effect=lambda key, value, **kw: cache.update({key: value})))
    monkeypatch.setattr(s, "get_redis", AsyncMock(return_value=redis))
    policy = AsyncMock(return_value=defaults.PERSONA_GROUNDING_CHECK_PROMPT)
    monkeypatch.setattr(s, "get_prompt_text", policy)
    profile = agent()
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=profile))))
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert (await s.get_cached_schedule("agent", day))[0]["event"] == "自由时间"
    await s.get_cached_schedule("agent", day)
    assert checker.await_count == 1
    assert redis.set.await_args.kwargs["ex"] == 86400
    profile.city = "江苏省镇江市润州区"
    checker.return_value = {"verdicts": [{"index": 0, "allowed": True}]}
    assert (await s.get_cached_schedule("agent", day))[0]["event"] == "润州街头"
    assert checker.await_count == 2
    policy.return_value += "\nA new Web policy version"
    await s.get_cached_schedule("agent", day)
    assert checker.await_count == 3
    assert "A new Web policy version" in checker.await_args.args[1]
    data[0]["event"] = "在凤凰路社区附近散步"
    cache["schedule:agent:20261009"] = json.dumps(data)
    await s.get_cached_schedule("agent", day)
    assert checker.await_count == 4
    checker.side_effect = TimeoutError()
    policy.return_value += "\nRetry policy"
    assert (await s.get_cached_schedule("agent", day))[0]["event"] == "自由时间"
    assert redis.set.await_args.kwargs["ex"] == 300


async def test_disabled_schedule_policy_cannot_reuse_a_previous_verdict(checker, monkeypatch):
    from app.services.prompting.store import PromptDisabledError
    data = [{"start": "18:00", "end": "19:00", "event": "润州街头", "status": "空闲"}]
    redis = SimpleNamespace(get=AsyncMock(return_value=json.dumps(data)), set=AsyncMock())
    monkeypatch.setattr(s, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent()))))
    monkeypatch.setattr(s, "get_prompt_text", AsyncMock(side_effect=PromptDisabledError("persona.grounding_check")))
    assert (await s.get_cached_schedule("agent", datetime(2026, 10, 9)))[0]["event"] == "自由时间"
    assert redis.get.await_count == 1  # no old derived verdict read
    redis.set.assert_not_awaited()


def _batch_verifier(*, reject=(), fail=()):
    """Fake verifier: judges each submitted item, times out a batch containing `fail`."""
    def verdicts(model, prompt, **kw):
        items = json.loads(prompt.split("待审核项：", 1)[1].splitlines()[0])
        if any(word in item["text"] for item in items for word in fail):
            raise TimeoutError()
        return {"verdicts": [{"index": item["index"], "reason": "",
                              "allowed": not any(word in item["text"] for word in reject)}
                             for item in items]}
    return verdicts


def _submitted(checker) -> list[list[str]]:
    return [[item["text"] for item in json.loads(call.args[1].split("待审核项：", 1)[1].splitlines()[0])]
            for call in checker.await_args_list]


async def test_schedule_is_verified_in_small_batches_and_a_failure_stays_local(checker):
    # One large batch outlived the verifier timeout and erased every gated slot,
    # including generic places the verifier would have allowed.
    events = [("下班回家", "空闲"), ("在车间抽检毛茶", "很忙碌"), ("去凤凰路的面馆吃早点", "空闲"),
              ("在实验室做检测", "很忙碌"), ("晚饭后在小区散步", "空闲"), ("回家做饭", "忙碌"),
              ("处理邮件", "忙碌")]
    slots = [{"start": f"{8 + i:02d}:00", "end": f"{9 + i:02d}:00", "event": e, "status": st}
             for i, (e, st) in enumerate(events)]
    checker.side_effect = _batch_verifier(reject=("凤凰路",), fail=("散步",))
    clean, verified = await s._ground_schedule_result(agent(), slots)
    assert sorted(_submitted(checker)) == sorted([
        ["下班回家", "在车间抽检毛茶"], ["去凤凰路的面馆吃早点", "在实验室做检测"], ["晚饭后在小区散步", "回家做饭"],
    ])
    assert [slot["event"] for slot in clean] == [
        "下班回家", "在车间抽检毛茶", "自由时间", "在实验室做检测", "自由时间", "忙手头的事", "处理邮件",
    ]
    assert [slot["start"] for slot in clean] == [slot["start"] for slot in slots]
    assert not verified
    assert slots[2]["event"] == "去凤凰路的面馆吃早点"  # source evidence retained


async def test_rejected_busy_slot_keeps_its_workload_in_current_state(checker):
    slots = [{"start": "09:00", "end": "12:00", "event": "去凤凰路的写字楼开会", "status": "很忙碌"}]
    clean = await s._ground_schedule(agent(), slots)
    status = s.get_current_status(clean, datetime(2026, 10, 9, 10, 0))
    context = s.format_schedule_context(status)
    assert status["status"] == "very_busy"
    assert "最忙" in context and "凤凰路" not in context and "日常活动" not in context


def test_neutral_replacements_never_need_verification_again():
    # A replacement that re-entered the gate would be re-verified on every read.
    for text in s._DEFAULT_ACTIVITY.values():
        assert not g.needs_location_verification(text, implicit_self=True)


async def test_schedule_without_canonical_city_neutralizes_only_gated_slots(checker):
    slots = [{"start": "09:00", "end": "12:00", "event": "处理客户工单", "status": "很忙碌"},
             {"start": "18:00", "end": "19:00", "event": "下班回家", "status": "空闲"}]
    clean, verified = await s._ground_schedule_result(agent(None), slots)
    assert [slot["event"] for slot in clean] == ["处理客户工单", "自由时间"]
    assert not verified
    checker.assert_not_awaited()


async def test_oversized_schedule_is_not_fanned_out(checker):
    slots = [{"start": "18:00", "end": "19:00", "event": "下班回家", "status": "空闲"}] * 51
    clean, verified = await s._ground_schedule_result(agent(), slots)
    assert not verified and {slot["event"] for slot in clean} == {"自由时间"}
    checker.assert_not_awaited()


async def test_new_schedule_verdict_is_reused_by_the_read_path(checker, monkeypatch):
    # Generation runs in the background; the first read is usually a chat turn.
    cache = {}
    redis = SimpleNamespace(get=AsyncMock(side_effect=lambda key: cache.get(key)),
                            set=AsyncMock(side_effect=lambda key, value, **kw: cache.update({key: value})))
    monkeypatch.setattr(s, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(s, "get_prompt_text", AsyncMock(return_value=defaults.PERSONA_GROUNDING_CHECK_PROMPT))
    profile = agent()
    upsert = AsyncMock()
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=profile)),
                                                 aidailyschedule=SimpleNamespace(upsert=upsert)))
    checker.side_effect = _batch_verifier(reject=("凤凰路",))
    budgets = []
    verify = g.verify_generated_locations
    async def spy(*args, **kwargs):
        budgets.append(kwargs["timeout"])
        return await verify(*args, **kwargs)
    monkeypatch.setattr(g, "verify_generated_locations", spy)
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    slots = [{"start": "18:00", "end": "19:00", "event": "下班回家", "status": "空闲"},
             {"start": "19:00", "end": "20:00", "event": "去凤凰路吃米线", "status": "空闲"}]
    clean = await s._ground_and_cache_new_schedule("agent", day, profile, slots)
    assert [slot["event"] for slot in clean] == ["下班回家", "自由时间"]
    assert upsert.await_args.kwargs["data"]["update"]["scheduleData"].data == clean
    assert checker.await_count == 1
    assert await s.get_cached_schedule("agent", day) == clean
    assert checker.await_count == 1
    profile.city = "江苏省镇江市润州区"  # a profile change still forces a fresh verdict
    await s.get_cached_schedule("agent", day)
    assert checker.await_count == 2
    # Background generation may wait; the chat-path read keeps the hot-path budget.
    assert budgets == [s._GENERATION_GROUNDING_TIMEOUT, g.VERIFY_TIMEOUT_S]
    assert s._GENERATION_GROUNDING_TIMEOUT > g.VERIFY_TIMEOUT_S


def _generation_store(monkeypatch, *, set_side_effect=None):
    cache = {}

    async def store(key, value, **kw):
        if set_side_effect:
            set_side_effect(key)
        cache[key] = value

    redis = SimpleNamespace(get=AsyncMock(side_effect=lambda key: cache.get(key)),
                            set=AsyncMock(side_effect=store))
    monkeypatch.setattr(s, "get_redis", AsyncMock(return_value=redis))
    upsert = AsyncMock()
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent())),
                                                 aidailyschedule=SimpleNamespace(upsert=upsert)))
    return cache, upsert


async def test_new_schedule_without_gated_slots_needs_no_policy_or_verifier(checker, monkeypatch):
    cache, upsert = _generation_store(monkeypatch)
    policy = AsyncMock(return_value=defaults.PERSONA_GROUNDING_CHECK_PROMPT)
    monkeypatch.setattr(s, "get_prompt_text", policy)
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    slots = [{"start": "09:00", "end": "18:00", "event": "处理客服工单", "status": "忙碌"}]
    assert await s._ground_and_cache_new_schedule("agent", day, agent(), slots) == slots
    policy.assert_not_awaited()
    checker.assert_not_awaited()
    assert upsert.await_count == 1
    assert s._grounded_cache_key("agent", day, agent(), slots, "") in cache


async def test_new_schedule_without_readable_policy_is_still_neutralized(checker, monkeypatch):
    """Web 策略停用/读不到: 照常核验落库, 但不能把结论记成可复用的。"""
    cache, upsert = _generation_store(monkeypatch)
    monkeypatch.setattr(s, "get_prompt_text", AsyncMock(side_effect=RuntimeError("disabled")))
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    slots = [{"start": "19:00", "end": "20:00", "event": "漫步润州街头", "status": "空闲"}]
    clean = await s._ground_and_cache_new_schedule("agent", day, agent(), slots)
    assert [slot["event"] for slot in clean] == ["自由时间"]
    assert upsert.await_args.kwargs["data"]["update"]["scheduleData"].data == clean
    assert not any(":grounded:" in key for key in cache)


async def test_recording_the_verdict_is_best_effort(checker, monkeypatch, caplog):
    def fail_on_verdict(key):
        if ":grounded:" in key:
            raise ConnectionError("redis down")

    _, upsert = _generation_store(monkeypatch, set_side_effect=fail_on_verdict)
    monkeypatch.setattr(s, "get_prompt_text", AsyncMock(return_value=defaults.PERSONA_GROUNDING_CHECK_PROMPT))
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    slots = [{"start": "09:00", "end": "18:00", "event": "处理客服工单", "status": "忙碌"}]
    with caplog.at_level("WARNING"):
        assert await s._ground_and_cache_new_schedule("agent", day, agent(), slots) == slots
    assert upsert.await_count == 1
    assert "Failed to record schedule grounding" in caplog.text


async def test_generated_schedule_goes_through_grounding_and_cache(monkeypatch):
    """LLM 生成的作息必须经过核验+落库这一个出口, 不能绕开它直接写缓存。"""
    import app.services.runtime_config as runtime_config
    slots = [{"start": f"{h:02d}:00", "end": f"{h + 1:02d}:00", "event": "处理客服工单", "status": "忙碌"}
             for h in range(9, 14)]
    monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock())
    monkeypatch.setattr(s, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent()))))
    monkeypatch.setattr(s, "is_holiday", lambda _d: None)
    monkeypatch.setattr(s, "is_workday_swap", lambda _d: False)
    monkeypatch.setattr(s, "classify_day_kind", lambda *a, **k: "工作日")
    monkeypatch.setattr(s, "get_prompt_text", AsyncMock(return_value=defaults.DAILY_SCHEDULE_PROMPT))
    monkeypatch.setattr(s, "get_utility_model", lambda: object())
    monkeypatch.setattr(s, "invoke_json", AsyncMock(return_value=slots))
    monkeypatch.setattr(s, "_apply_carried_adjustment", AsyncMock(side_effect=lambda _a, _d, sch: sch))
    grounded = [{**slot, "event": "忙手头的事"} for slot in slots]
    seal = AsyncMock(return_value=grounded)
    monkeypatch.setattr(s, "_ground_and_cache_new_schedule", seal)
    day = datetime(2026, 10, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    result = await s.generate_daily_schedule("agent", "小伴", None, life_overview="客服员的一天", date=day)
    assert result == grounded
    assert seal.await_args.args[3] == slots


async def test_expired_verification_budget_is_unverified_not_allowed(checker):
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
        return {"verdicts": [{"index": 0, "allowed": True}]}
    checker.side_effect = slow
    with pytest.raises(g.GroundingUnavailable):
        await g.verify_generated_locations(agent(), ["下班回家"], kind="schedule", timeout=0.01)


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
    # Keep the real conversation-scope check; this unit fixture owns the
    # synthetic conversation and does not connect the global Prisma client.
    monkeypatch.setattr(post, "db", SimpleNamespace(conversation=SimpleNamespace(
        find_unique=AsyncMock(return_value=SimpleNamespace(
            id="conv", userId="owner", workspaceId="scope", isDeleted=False)))))
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
    results = await post._do_memory_pipeline("owner", [{"role": "assistant", "content": "清晨在镇江醒来", "createdAt": datetime.now(UTC).isoformat()}], "conv", "scope")
    assert results[1].status == "failed" and results[1].memory_count is None
    assert results[1].progress == "unchanged"
    watermark.assert_not_awaited()
    assert checker.await_count == 1
    # Confirmed invalid content is intentionally rejected, then consumption can
    # advance; an unavailable verifier must not silently consume the batch.
    checker.side_effect = None
    results = await post._do_memory_pipeline("owner", [{"role": "assistant", "content": "清晨在镇江醒来", "createdAt": datetime.now(UTC).isoformat()}], "conv", "scope")
    assert results[1].status == "completed" and results[1].memory_count == 0
    assert results[1].progress == "advanced"
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
    assert result["ai_activity"] == "自由时间"
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
